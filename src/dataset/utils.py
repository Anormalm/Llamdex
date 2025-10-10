import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from src.preprocess.DataScaler import TableScaler
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN, MISTRAL_ALPHABET_TOKENS
from src.utils.constants import LLAMA_YES_TOKEN, LLAMA_NO_TOKEN, LLAMA_ALPHABET_TOKENS

class TextDataset(Dataset):
    def __init__(self, data, answer_column, tokenizer=None, encoder_tokenizer=None, system_prompt=None, dataset_json=None, add_features=False):
        # add_features is a flag to add features to the input, tailored for the encoder training
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

        data_vec = data.drop(columns=[answer_column, 'formatted_text', 'prompt_to_llm']).to_numpy()
        self.data_dummy = self.scaler.transform(data_vec)
        self.data_dummy = torch.tensor(self.data_dummy, dtype=torch.float32)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        def letter_to_number(letter):
            if isinstance(letter, int):
                return letter
            if isinstance(letter, str):
                if letter.isdigit():
                    return int(letter)
                if letter.isalpha() and len(letter) == 1:
                    return ord(letter.upper()) - ord('A')
            return letter

        if self.add_features:
            row = self.raw_data[idx]
            row_copy = row.copy()
            for key in [self.answer_column, 'formatted_text', 'prompt_to_llm']:
                if key in row_copy:
                    del row_copy[key]

            def format_value(v):
                if isinstance(v, float) and v.is_integer():
                    return int(v)
                return v

            data_vec_str = ' '.join([f'{k}: {format_value(v)}' for k, v in row_copy.items()])
            prompt = f"{row['formatted_text']} Features: {data_vec_str}{self.encoder_tokenizer.eos_token}"
            tokens = self.encoder_tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
            attention_mask = torch.ones_like(tokens)
            label = torch.tensor(letter_to_number(row[self.answer_column]), dtype=torch.long)

            # Create token_labels for loss calculation
            token_labels = tokens.clone()
            query_input = f"{row[self.answer_column]} Features:"
            query_tokens = self.encoder_tokenizer(query_input, return_tensors="pt").input_ids.squeeze(0)
            token_labels[:query_tokens.size(0)] = -100

            return {"tokens": tokens, "attention_mask": attention_mask, "token_labels": token_labels, "labels": label, "features": self.data_dummy[idx]}

        else:
            row = self.raw_data[idx]

            if self.system_prompt is not None:
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": row['formatted_text']}]
            else:
                messages = [{"role": "user", "content": row['formatted_text']}]

            tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt").squeeze(0)
            attention_mask = (tokens != self.tokenizer.pad_token_id).long()
            label = torch.tensor(letter_to_number(row[self.answer_column]), dtype=torch.long)

            if self.encoder_tokenizer is not None:
                prompt = f"{row['formatted_text']} Features:"
                encoder_tokens = self.encoder_tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
                encoder_attention_mask = torch.ones_like(encoder_tokens)
                return {"tokens": tokens, "attention_mask": attention_mask, "labels": label,
                        "features": self.data_dummy[idx],
                        "encoder_tokens": encoder_tokens, "encoder_attention_mask": encoder_attention_mask}
            else:
                return {"tokens": tokens, "attention_mask": attention_mask, "labels": label, "features": self.data_dummy[idx]}

    @property
    def n_dummy_features(self):
        return self.data_dummy.shape[1]

    @property
    def n_dummy_features(self):
        return self.data_dummy.shape[1]


def get_collate_fn(pad_token_id, add_features=False):
    def collate_fn(batch):
        batch_tokens = [item['tokens'] for item in batch]
        batch_attention_mask = [item['attention_mask'] for item in batch]

        labels = torch.stack([item['labels'] for item in batch])
        features = torch.stack([item['features'] for item in batch])

        max_length = max(tokens.size(0) for tokens in batch_tokens)

        padded_tokens = torch.full((len(batch_tokens), max_length), pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(batch_tokens), max_length), dtype=torch.long)

        if add_features:
            batch_token_labels = [item['token_labels'] for item in batch]
            padded_token_labels = torch.full((len(batch_tokens), max_length), -100, dtype=torch.long)
            for i, (tokens, attention_mask, token_labels) in enumerate(
                    zip(batch_tokens, batch_attention_mask, batch_token_labels)):
                seq_length = tokens.size(0)
                padded_tokens[i, max_length - seq_length:] = tokens
                padded_attention_mask[i, max_length - seq_length:] = attention_mask
                padded_token_labels[i, max_length - seq_length:] = token_labels

            return {"tokens": padded_tokens, "attention_mask": padded_attention_mask,
                    "token_labels": padded_token_labels, "labels": labels, "features": features}

        else:
            for i, (tokens, attention_mask) in enumerate(zip(batch_tokens, batch_attention_mask)):
                seq_length = tokens.size(0)
                padded_tokens[i, max_length - seq_length:] = tokens
                padded_attention_mask[i, max_length - seq_length:] = attention_mask

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

                return {"tokens": padded_tokens, "attention_mask": padded_attention_mask, "labels": labels,
                        "features": features, "encoder_tokens": padded_encoder_tokens, "encoder_attention_mask": padded_encoder_attention_mask}
            else:
                return {"tokens": padded_tokens, "attention_mask": padded_attention_mask, "labels": labels,
                        "features": features}

    return collate_fn


class TextDatasetForEvaluation(Dataset):
    def __init__(self, data, answer_column, dataset_json):
        self.raw_data = data.to_dict('records')
        self.answer_column = answer_column
        self.scaler = TableScaler(dataset_json)
        data_vec = data.drop(columns=[answer_column, 'formatted_text', 'prompt_to_llm']).to_numpy()
        self.data_dummy = self.scaler.transform(data_vec)
        self.data_dummy = torch.tensor(self.data_dummy, dtype=torch.float32)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        def letter_to_number(letter):
            if isinstance(letter, int):
                return letter
            if isinstance(letter, str):
                if letter.isdigit():
                    return int(letter)
                if letter.isalpha() and len(letter) == 1:
                    return ord(letter.upper()) - ord('A')
            return letter

        row = self.raw_data[idx]
        label = torch.tensor(letter_to_number(row[self.answer_column]), dtype=torch.long)

        return {"text": row['formatted_text'], "labels": label, "features": self.data_dummy[idx]}


def get_collate_fn_for_evaluation(tokenizer, expert_list, system_prompt=None, gt_features=False, device=torch.device('cuda'), dtype=torch.bfloat16):
    # expert_list is a list of (encoder, expert, expert_description) tuples
    def collate_fn(batch):
        labels = torch.stack([item['labels'] for item in batch])
        results_list = []
        for i, (encoder, expert, expert_description) in enumerate(expert_list):
            if not gt_features:
                batch_encoder_tokens = [encoder.tokenizer(f"{item['text']} Features:", return_tensors="pt").input_ids.squeeze(0) for item in batch]
                batch_encoder_attention_mask = [torch.ones_like(encoder_tokens) for encoder_tokens in batch_encoder_tokens]
                max_encoder_length = max(tokens.size(0) for tokens in batch_encoder_tokens)

                padded_encoder_tokens = torch.full((len(batch_encoder_tokens), max_encoder_length), encoder.tokenizer.pad_token_id,
                                                   dtype=torch.long)
                padded_encoder_attention_mask = torch.zeros((len(batch_encoder_tokens), max_encoder_length),
                                                            dtype=torch.long)

                for i, (tokens, attention_mask) in enumerate(zip(batch_encoder_tokens, batch_encoder_attention_mask)):
                    seq_length = tokens.size(0)
                    padded_encoder_tokens[i, max_encoder_length - seq_length:] = tokens
                    padded_encoder_attention_mask[i, max_encoder_length - seq_length:] = attention_mask

                encoder_tokens = padded_encoder_tokens.to(device)
                encoder_attn_mask = padded_encoder_attention_mask.to(device)

                features = encoder(encoder_tokens, attention_mask=encoder_attn_mask)
            else:
                features = torch.stack([item['features'] for item in batch])

            features = features.to(device).to(dtype)
            results = expert(features)
            results_list.append(results.detach().cpu())

        tokens_list = []
        attention_mask_list = []
        for i, instance in enumerate(batch):
            text = instance['text']
            prompt = text
            if len(expert_list) > 0:
                prompt += "\nSystem: Expert predictions:"
            for j in range(len(expert_list)):
                results = results_list[j]
                prompt += f"\n{expert_list[j][2]}: {results[i].item() * 100}%"
                # prompt += f"\n{expert_list[j][2]}: {'Yes' if results[i].item() > 0.5 else 'No'}"

            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
            tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").squeeze(0)
            attention_mask = (tokens != tokenizer.pad_token_id).long()
            tokens_list.append(tokens)
            attention_mask_list.append(attention_mask)

        max_length = max(tokens.size(0) for tokens in tokens_list)
        padded_tokens = torch.full((len(tokens_list), max_length), tokenizer.pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(tokens_list), max_length), dtype=torch.long)
        for i, (tokens, attention_mask) in enumerate(zip(tokens_list, attention_mask_list)):
            seq_length = tokens.size(0)
            padded_tokens[i, max_length - seq_length:] = tokens
            padded_attention_mask[i, max_length - seq_length:] = attention_mask

        return {"tokens": padded_tokens, "attention_mask": padded_attention_mask, "labels": labels}

    return collate_fn

class TextDatasetForBaseline(Dataset):
    def __init__(self, data, answer_column, tokenizer=None, dataset_json=False, syn_mode=False):
        self.raw_data = data.to_dict('records')
        self.answer_column = answer_column
        self.tokenizer = tokenizer
        self.syn_mode = syn_mode
        if isinstance(dataset_json, str):
            with open(dataset_json, 'r') as f:
                dataset_json = json.load(f)
        self.dataset_json = dataset_json
        self.is_label_binary = dataset_json['y']['type'] == "bool"

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        def convert_label_to_text(label):
            if self.syn_mode:
                try:
                    int_label = int(float(label))
                    if self.is_label_binary:
                        if int_label == 0:
                            return "No"
                        elif int_label == 1:
                            return "Yes"
                        else:
                            return str(int_label)
                    else:
                        return chr(int_label + ord('A'))
                except ValueError:
                    return str(label)
            else:
                if label == 0:
                    return "No"
                elif label == 1:
                    return "Yes"
                else:
                    return str(label)

        def format_value(v):
            if isinstance(v, float):
                if self.syn_mode:
                    return f"{v:.4f}".rstrip('0').rstrip('.')
                else:
                    return str(v)
            return str(v)

        def process_dummy_features(row):
            feature_dict = {}
            new_row = row.copy()
            for k, v in row.items():
                if '_' in k:
                    base_feature, category = k.rsplit('_', 1)
                    del(new_row[k])
                    if base_feature not in feature_dict or v > feature_dict[base_feature][1]:
                        feature_dict[base_feature] = (category, v)

            for base_feature, (category, value) in feature_dict.items():
                new_row[base_feature] = category

            new_row = {k: new_row[k] for k in sorted(new_row)}

            return new_row

        row = self.raw_data[idx]
        if self.syn_mode:
            row = process_dummy_features(row.copy())

        query = " ".join([f"{k}: {format_value(v)}" for k, v in row.items() if k not in [self.answer_column, 'formatted_text', 'prompt_to_llm']]) + " Answer:"
        prompt = query + " " + convert_label_to_text(row[self.answer_column])

        tokens = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()

        token_labels = tokens.clone()
        query_tokens = self.tokenizer.encode(query, return_tensors="pt").squeeze(0)
        token_labels[:query_tokens.size(0)] = -100

        return {"tokens": tokens, "attention_mask": attention_mask, "token_labels": token_labels}


def get_collate_fn_for_baseline(pad_token_id):
    def collate_fn(batch):
        batch_tokens = [item['tokens'] for item in batch]
        batch_attention_mask = [item['attention_mask'] for item in batch]
        batch_token_labels = [item['token_labels'] for item in batch]

        max_length = max(tokens.size(0) for tokens in batch_tokens)

        padded_tokens = torch.full((len(batch_tokens), max_length), pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(batch_tokens), max_length), dtype=torch.long)
        padded_token_labels = torch.full((len(batch_tokens), max_length), -100, dtype=torch.long)

        for i, (tokens, attention_mask, token_labels) in enumerate(zip(batch_tokens, batch_attention_mask, batch_token_labels)):
            seq_length = tokens.size(0)
            padded_tokens[i, max_length - seq_length:] = tokens
            padded_attention_mask[i, max_length - seq_length:] = attention_mask
            padded_token_labels[i, max_length - seq_length:] = token_labels

        return {"tokens": padded_tokens, "attention_mask": padded_attention_mask, "token_labels": padded_token_labels}

    return collate_fn


def get_dataset(dataset_type, data_file, tokenizer=None, encoder_tokenizer=None, dataset_json=None, n_instances=None, system_prompt=None, add_features=False):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    dataset_info_json = f"src/dataset/dataset_info.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    answer_column = dataset_info_json[dataset_type]["answer_column"]

    return TextDataset(
        pd.read_csv(data_file).head(n_instances) if n_instances is not None else pd.read_csv(data_file),
        answer_column, tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer,
        system_prompt=system_prompt, dataset_json=dataset_json, add_features=add_features)


def get_dataset_for_evaluation(dataset_type, data_file, dataset_json=None, n_instances=None):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    dataset_info_json = f"src/dataset/dataset_info.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    answer_column = dataset_info_json[dataset_type]["answer_column"]

    return TextDatasetForEvaluation(
        pd.read_csv(data_file).head(n_instances) if n_instances is not None else pd.read_csv(data_file),
        answer_column, dataset_json)


def get_dataset_for_baseline(dataset_type, data_file, dataset_json=None, tokenizer=None, n_instances=None, syn_mode=False):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    dataset_info_json = f"src/dataset/dataset_info.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    answer_column = dataset_info_json[dataset_type]["answer_column"]

    if syn_mode:
        csv_file = pd.read_csv(data_file, index_col=0).head(n_instances) if n_instances is not None else pd.read_csv(data_file, index_col=0)
    else:
        csv_file = pd.read_csv(data_file).head(n_instances) if n_instances is not None else pd.read_csv(data_file)

    return TextDatasetForBaseline(csv_file, answer_column, tokenizer, dataset_json, syn_mode)


def get_class_tokens(dataset_type, model="mistral"):
    if model == "mistral":
        if dataset_type in ['adult', 'titanic', 'bank_marketing']:
            class_tokens = MISTRAL_NO_TOKEN + MISTRAL_YES_TOKEN
        elif dataset_type == 'wine_quality':
            class_tokens = MISTRAL_ALPHABET_TOKENS[:11]  # A to K
        elif dataset_type == 'abalone':
            class_tokens = MISTRAL_ALPHABET_TOKENS[:3] # A to C
        elif dataset_type == "nursery":
            class_tokens = MISTRAL_ALPHABET_TOKENS[:4] # A to D
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return class_tokens
    elif model == "llama2":
        if dataset_type in ['adult', 'titanic', 'bank_marketing']:
            class_tokens = LLAMA_NO_TOKEN + LLAMA_YES_TOKEN
        elif dataset_type == 'wine_quality':
            class_tokens = LLAMA_ALPHABET_TOKENS[:11] # A to K
        elif dataset_type == 'abalone':
            class_tokens = LLAMA_ALPHABET_TOKENS[:3] # A to C
        elif dataset_type == "nursery":
            class_tokens = LLAMA_ALPHABET_TOKENS[:4] # A to D
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return class_tokens
    else:
        raise ValueError(f"Invalid model: {model}")
