from typing import Callable
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import sys
import os

from src.dataset.synthetic import interpret_to_human_readable, check_contradictions_with_llm


def generate_text_batch(batch, tokenizer, model, query_prefix: str, label_name: str, max_new_tokens: int = 3000,
                        check_contradiction=False):
    if check_contradiction:
        contradictions = check_contradictions_with_llm(batch, tokenizer, model, label_name)
    else:
        contradictions = [False] * len(batch)

    valid_batch = [row for row, is_contradiction in zip(batch, contradictions) if not is_contradiction]
    valid_indices = [i for i, is_contradiction in enumerate(contradictions) if not is_contradiction]

    if not valid_batch:
        return [], [], []

    batched_tokens = []
    batched_attention_mask = []
    queries = []

    for row in valid_batch:
        items = list(row.items())
        random.shuffle(items)
        query_str = query_prefix + " ".join(
            f'#{key}: {value}' for key, value in items if (key != label_name and not pd.isnull(value)))
        messages = [{"role": "user", "content": query_str}]
        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        attention_mask = (tokens != tokenizer.pad_token_id).long()

        queries.append(query_str)
        batched_tokens.append(tokens)
        batched_attention_mask.append(attention_mask)

    max_length = max(t.size(-1) for t in batched_tokens)

    padded_tokens = torch.full((len(batched_tokens), max_length), tokenizer.pad_token_id, dtype=torch.long).to("cuda")
    padded_attention_mask = torch.zeros((len(batched_tokens), max_length), dtype=torch.long).to("cuda")

    for i, (tokens, attention_mask) in enumerate(zip(batched_tokens, batched_attention_mask)):
        seq_length = tokens.size(-1)
        padded_tokens[i, -seq_length:] = tokens
        padded_attention_mask[i, -seq_length:] = attention_mask

    with torch.no_grad():
        generated_ids = model.generate(input_ids=padded_tokens, attention_mask=padded_attention_mask,
                                       max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    results = []
    for i in range(len(batched_tokens)):
        generated_id = generated_ids[i, padded_tokens.shape[-1]:]
        result = tokenizer.decode(generated_id.tolist(), skip_special_tokens=True)
        results.append(result)
    return queries, results, valid_indices


def get_label_name_from_dataset(dataset: str, dataset_json_path: str = None) -> str:
    """
    Get the label name from the dataset json file. If json file not specified, use the default path
    Args:
        dataset_json_path:
        dataset: name of the dataset
        dataset_json: path to the dataset json file
    Returns: label name

    """
    if dataset_json_path is None:
        dataset_json_path = os.path.join(os.path.dirname(__file__), f"../../dataset/{dataset}/{dataset}.json")
    dataset_json = pd.read_json(dataset_json_path)
    label_name = dataset_json['y']['name']
    return label_name


def tabular_to_text(dataset: str, input_file: str, output_file_path: str, llm_path: str, llm_id: str, query_prefix: str,
                    max_new_tokens: int = 200, seed: int = 0, save_period: int = 100, batch_size: int = 8,
                    check_contradiction=False, n_repeat: int = 1):
    """
    Convert tabular data to text using a language model
    Args:
        dataset:
        input_file:
        output_file_path:
        llm_path:
        llm_id:
        query_prefix:
        max_new_tokens:
        seed:
        save_period:
        batch_size:
        check_contradiction:
        n_repeat: number of times to repeatedly generate text for each row

    Returns:

    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(llm_id, padding_side='left', cache_dir=llm_path, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(llm_id, cache_dir=llm_path, device_map='auto')

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(input_file)
    new_df = pd.DataFrame(columns=df.columns.tolist() + ['prompt_to_llm'] + ['formatted_text'])

    df_readable = interpret_to_human_readable(df.copy(), dataset)
    label_name = get_label_name_from_dataset(dataset)
    for _ in range(n_repeat):
        for i in tqdm(range(0, len(df_readable), batch_size)):
            readable_batch = df_readable.iloc[i:i + batch_size].to_dict('records')
            queries, formatted_texts, valid_indices = generate_text_batch(readable_batch, tokenizer, model,
                                                                          query_prefix=query_prefix,
                                                                          label_name=label_name,
                                                                          max_new_tokens=max_new_tokens,
                                                                          check_contradiction=check_contradiction)
            batch = df.iloc[i:i + batch_size]
            for j, idx in enumerate(valid_indices):
                new_df.loc[len(new_df)] = list(batch.iloc[idx]) + [queries[j], formatted_texts[j]]

            if i // batch_size % save_period == 0:
                new_df.to_csv(output_file_path, index=False)

    new_df.to_csv(output_file_path, index=False)
