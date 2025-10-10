import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from src.preprocess.DataScaler import TableScaler


class TextDatasetGeneration(Dataset):
    def __init__(self, data, answer_column, tokenizer=None, encoder_tokenizer=None, system_prompt=None, dataset_json=None, add_features=False):
        """
        Dataset for sequence generation training.
        Unlike the classification version, this uses full sequence loss instead of just class tokens.
        """
        self.raw_data = data.to_dict('records')
        self.answer_column = answer_column
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.system_prompt = system_prompt
        self.dataset_json = dataset_json
        self.add_features = add_features

        if add_features and encoder_tokenizer is None:
            raise ValueError("encoder_tokenizer must be provided when add_features is True")

        if not add_features and tokenizer is None:
            raise ValueError("tokenizer must be provided when add_features is False")

        if add_features and system_prompt is not None:
            print("Warning: Ignoring system prompt when add_features is True")
            system_prompt = None

        self.scaler = TableScaler(dataset_json)

        data_vec = data.drop(columns=[answer_column, 'formatted_text', 'prompt_to_llm', 'response_text']).to_numpy()
        self.data_dummy = self.scaler.transform(data_vec)
        self.data_dummy = torch.tensor(self.data_dummy, dtype=torch.float32)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.add_features:
            row = self.raw_data[idx]
            row_copy = row.copy()
            for key in [self.answer_column, 'formatted_text', 'prompt_to_llm', 'response_text']:
                if key in row_copy:
                    del row_copy[key]

            def format_value(v):
                if isinstance(v, float) and v.is_integer():
                    return int(v)
                return v

            data_vec_str = ' '.join([f'{k}: {format_value(v)}' for k, v in row_copy.items()])
            # For generation, we include the full response in the training
            prompt = f"{row['formatted_text']} Features: {data_vec_str} Answer: {row['response_text']}{self.encoder_tokenizer.eos_token}"
            tokens = self.encoder_tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
            attention_mask = torch.ones_like(tokens)

            # Create token_labels for loss calculation - only compute loss on the answer part
            token_labels = tokens.clone()
            query_input = f"{row['formatted_text']} Features: {data_vec_str} Answer:"
            query_tokens = self.encoder_tokenizer(query_input, return_tensors="pt").input_ids.squeeze(0)
            token_labels[:query_tokens.size(0)] = -100

            return {"tokens": tokens, "attention_mask": attention_mask, "token_labels": token_labels, "features": self.data_dummy[idx]}

        else:
            row = self.raw_data[idx]

            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": row['formatted_text']},
                    {"role": "assistant", "content": row['response_text']}
                ]
            else:
                messages = [
                    {"role": "user", "content": row['formatted_text']},
                    {"role": "assistant", "content": row['response_text']}
                ]

            tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt").squeeze(0)
            attention_mask = (tokens != self.tokenizer.pad_token_id).long()

            # Create labels for sequence generation - only compute loss on assistant response
            token_labels = tokens.clone()
            if self.system_prompt is not None:
                user_messages = [
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": row['formatted_text']}
                ]
            else:
                user_messages = [{"role": "user", "content": row['formatted_text']}]
            
            user_tokens = self.tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, return_tensors="pt").squeeze(0)
            token_labels[:user_tokens.size(0)] = -100

            if self.encoder_tokenizer is not None:
                prompt = f"{row['formatted_text']} Features:"
                encoder_tokens = self.encoder_tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
                encoder_attention_mask = torch.ones_like(encoder_tokens)
                return {"tokens": tokens, "attention_mask": attention_mask, "token_labels": token_labels,
                        "features": self.data_dummy[idx],
                        "encoder_tokens": encoder_tokens, "encoder_attention_mask": encoder_attention_mask}
            else:
                return {"tokens": tokens, "attention_mask": attention_mask, "token_labels": token_labels, "features": self.data_dummy[idx]}

    @property
    def n_dummy_features(self):
        return self.data_dummy.shape[1]


def get_collate_fn_generation(pad_token_id, add_features=False):
    def collate_fn(batch):
        batch_tokens = [item['tokens'] for item in batch]
        batch_attention_mask = [item['attention_mask'] for item in batch]
        batch_token_labels = [item['token_labels'] for item in batch]

        features = torch.stack([item['features'] for item in batch])

        max_length = max(tokens.size(0) for tokens in batch_tokens)

        padded_tokens = torch.full((len(batch_tokens), max_length), pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(batch_tokens), max_length), dtype=torch.long)
        padded_token_labels = torch.full((len(batch_tokens), max_length), -100, dtype=torch.long)

        for i, (tokens, attention_mask, token_labels) in enumerate(
                zip(batch_tokens, batch_attention_mask, batch_token_labels)):
            seq_length = tokens.size(0)
            padded_tokens[i, max_length - seq_length:] = tokens
            padded_attention_mask[i, max_length - seq_length:] = attention_mask
            padded_token_labels[i, max_length - seq_length:] = token_labels

        result = {"tokens": padded_tokens, "attention_mask": padded_attention_mask,
                "token_labels": padded_token_labels, "features": features}

        if 'encoder_tokens' in batch[0]:
            batch_encoder_tokens = [item['encoder_tokens'] for item in batch]
            batch_encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
            max_encoder_length = max(tokens.size(0) for tokens in batch_encoder_tokens)

            padded_encoder_tokens = torch.full((len(batch_encoder_tokens), max_encoder_length), pad_token_id, dtype=torch.long)
            padded_encoder_attention_mask = torch.zeros((len(batch_encoder_tokens), max_encoder_length), dtype=torch.long)

            for i, (tokens, attention_mask) in enumerate(zip(batch_encoder_tokens, batch_encoder_attention_mask)):
                seq_length = tokens.size(0)
                padded_encoder_tokens[i, max_encoder_length - seq_length:] = tokens
                padded_encoder_attention_mask[i, max_encoder_length - seq_length:] = attention_mask

            result.update({"encoder_tokens": padded_encoder_tokens, "encoder_attention_mask": padded_encoder_attention_mask})

        return result

    return collate_fn


class TextDatasetForEvaluationGeneration(Dataset):
    def __init__(self, data, answer_column, dataset_json):
        self.raw_data = data.to_dict('records')
        self.answer_column = answer_column
        self.scaler = TableScaler(dataset_json)
        data_vec = data.drop(columns=[answer_column, 'formatted_text', 'prompt_to_llm', 'response_text']).to_numpy()
        self.data_dummy = self.scaler.transform(data_vec)
        self.data_dummy = torch.tensor(self.data_dummy, dtype=torch.float32)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        row = self.raw_data[idx]
        # For evaluation, we return the expected response for comparison
        return {"text": row['formatted_text'], "expected_response": row['response_text'], "features": self.data_dummy[idx]}


def get_dataset_generation(dataset_type, data_file, tokenizer=None, encoder_tokenizer=None, dataset_json=None, n_instances=None, system_prompt=None, add_features=False):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    dataset_info_json = f"src/dataset/dataset_info_generation.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    answer_column = dataset_info_json[dataset_type]["answer_column"]

    return TextDatasetGeneration(
        pd.read_csv(data_file).head(n_instances) if n_instances is not None else pd.read_csv(data_file),
        answer_column, tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer,
        system_prompt=system_prompt, dataset_json=dataset_json, add_features=add_features)


def get_dataset_for_evaluation_generation(dataset_type, data_file, dataset_json=None, n_instances=None):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    dataset_info_json = f"src/dataset/dataset_info_generation.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    answer_column = dataset_info_json[dataset_type]["answer_column"]

    return TextDatasetForEvaluationGeneration(
        pd.read_csv(data_file).head(n_instances) if n_instances is not None else pd.read_csv(data_file),
        answer_column, dataset_json) 