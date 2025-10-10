#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers.utils import logging
import time


# ------------------------------
# Helper: string -> bool for argparse
# ------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# ------------------------------
# Custom collate: keep raw text, stack tensors
# ------------------------------
def custom_collate(batch):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    features = torch.stack(
        [torch.tensor(item["features"], dtype=torch.float32) for item in batch]
    )
    return {"text": texts, "labels": labels, "features": features}


# ------------------------------
# Dataset for evaluation
# ------------------------------
class TextDatasetForEvaluation(Dataset):
    def __init__(self, data: pd.DataFrame, answer_column: str, dataset_json=None):
        self.raw_data = data.to_dict("records")
        self.answer_column = answer_column
        # drop answer + formatted_text + prompt_to_llm
        features = data.drop(columns=[answer_column, "formatted_text", "prompt_to_llm"])
        features_numeric = pd.get_dummies(features)
        self.data_dummy = features_numeric.to_numpy().astype(np.float32)

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
        label = torch.tensor(
            letter_to_number(row[self.answer_column]), dtype=torch.long
        )
        return {
            "text": row["formatted_text"],
            "labels": label,
            "features": self.data_dummy[idx]
        }


# ------------------------------
# Load dataset from CSV or DataFrame
# ------------------------------
def get_dataset_for_evaluation(dataset_type, data_source, dataset_json=None, n_instances=None):
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
    info_path = f"src/dataset/dataset_info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    if dataset_type not in info:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    answer_column = info[dataset_type]["answer_column"]

    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    else:
        df = data_source.copy()
    if n_instances is not None:
        df = df.head(n_instances)
    return TextDatasetForEvaluation(df, answer_column, dataset_json)


# ------------------------------
# LimitedDomain mechanism (Gumbel noise)
# ------------------------------
def limited_domain(h, k=1, k_bar=10, epsilon=1.0, delta=1e-5,
                   Delta_0=1, Delta_infty=1):
    sorted_items = sorted(h.items(), key=lambda x: x[1], reverse=True)
    base = sorted_items[k_bar][1] if len(sorted_items) >= k_bar + 1 else 0
    h_perp = base + 1 + (2 * math.log(min(Delta_0, k_bar - k) / delta)) / epsilon
    v_perp = h_perp + np.random.gumbel(loc=0, scale=(2 * Delta_infty / epsilon))

    v_cands = []
    tok_cands = []
    for token_id, score in sorted_items[:k]:
        v = score + np.random.gumbel(loc=0, scale=(2 * Delta_infty / epsilon))
        v_cands.append(v)
        tok_cands.append(token_id)

    if not v_cands:
        return -1
    max_v = max(v_cands)
    if max_v > v_perp:
        return tok_cands[v_cands.index(max_v)]
    return -1


# ------------------------------
# vLLM helper: single-request generation
# ------------------------------
def vllm_generate(engine: LLM, prompt_texts: list,
                  max_new_tokens: int, temperature: float,
                  repetition_penalty: float) -> list:
    sampling_params = SamplingParams(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens
    )
    # result = engine.generate([prompt_text], sampling_params, use_tqdm=False)[0].outputs[0].text
    result = engine.generate(prompt_texts, sampling_params, use_tqdm=(len(prompt_texts) > 1))
    return result


# ------------------------------
# Instruct-based next token candidate (uses vLLM)
# ------------------------------
def get_next_token_candidate(
    local_engine: LLM,
    local_tokenizer: AutoTokenizer,
    demo_subset, current_prompt: str, current_output: str,
    gen_temp: float, rep_penalty: float, num_demos: int
) -> int:
    demos = demo_subset[:num_demos]
    demo_texts = ""
    for inp, gt, pred in demos:
        demo_texts += f"Input: {inp}\nCorrect Answer: {gt}\nLLM Prediction: {pred}\n\n"

    messages = [
        # {"role": "system", "content": current_prompt},
        {"role": "user",
         "content": f"Optimize this prompt for the task: {current_prompt}\n"
                    f"Below are some demonstrations of the task with LLM's current predictions under this prompt:\n"
                    f"Demonstrations:\n{demo_texts}"
                    f"Now, please optimize the prompt to improve the LLM's predictions' accuracy.\n"
                    f"Recall that your prompt to be optimized is:\n{current_prompt}\n"
                    f"You should reiterate in your prompt that the answer should be enclosed in \\boxed{{...}} with correct format mentioned in the above prompt, otherwise the LLM's prediction would be considered incorrect.\n"
                    f"Directly provide the optimized prompt without any additional text. Your response should be concise (not more than 100 words).\n"},
        {"role": "assistant",
         "content": current_output}
    ]
    # tokenize + apply chat template
    prompt_str = local_tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False,
        truncation=True
    )

    if prompt_str.endswith("<|eot_id|>"):
        prompt_str = prompt_str[:-len("<|eot_id|>")]

    # print(f"\nPrompt:\n{prompt_str}")
    # ask vLLM for one token of continuation
    out = vllm_generate(
        local_engine, [prompt_str],
        max_new_tokens=1,
        temperature=gen_temp,
        repetition_penalty=rep_penalty
    )
    # print(f"\nPrompt:\n{prompt_str}")
    out_text = out[0].outputs[0].text
    # print(f"Output: {out_text}")
    toks = local_tokenizer.encode(out_text, add_special_tokens=False)
    # print(f"Tokenized output: {toks}")
    return toks[0] if toks else -1


# ------------------------------
# DP-EnsGen: generate a sequence of tokens under DP
# ------------------------------
def dp_ensgen(
    local_engine: LLM, local_tokenizer: AutoTokenizer,
    predicted_data, current_prompt: str,
    epsilon: float, delta: float, q: float,
    max_new_tokens: int, ensemble_num: int,
    gen_temp: float, rep_penalty: float, num_demos: int
):
    generated_tokens = []
    l = 0
    progress_bar = tqdm(total=max_new_tokens, desc="DP-EnsGen", unit="token")
    while l < max_new_tokens:
        token_candidate = -1
        failure_count = 0
        patience = 20
        while (token_candidate == -1 and failure_count <= patience):
            S = [e for e in predicted_data if np.random.rand() < q]
            if not S:
                print(f"Warning: Poisson sample empty at step {l}, re-sampling...")
                continue
            np.random.shuffle(S)
            group_size = max(1, math.ceil(len(S) / ensemble_num))
            S_groups = [S[i:i+group_size] for i in range(0, len(S), group_size)]

            h = {}
            for group in S_groups:
                candidate = get_next_token_candidate(
                    local_engine, local_tokenizer,
                    group, current_prompt,
                    local_tokenizer.decode(torch.tensor(generated_tokens)) if generated_tokens else "",
                    gen_temp, rep_penalty, num_demos
                )
                h[candidate] = h.get(candidate, 0) + 1

            if epsilon < float("inf"):
                token_candidate = limited_domain(h, k=1, k_bar=10, epsilon=epsilon, delta=delta)
            else:
                token_candidate = max(h, key=h.get)

            if token_candidate == -1:
                failure_count += 1
                continue

            decoded_token = local_tokenizer.decode([token_candidate])

            if decoded_token == "<|eot_id|>":
                break

            generated_tokens.append(token_candidate)
            l += 1
            # print(f"Token {l}: {local_tokenizer.decode([token_candidate])}")
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"Token: {local_tokenizer.decode([token_candidate])}")

        if failure_count > patience:
            print(f"Warning: DP-EnsGen failed to generate token at step {l}")
            break


    return generated_tokens


# ------------------------------
# DP-Argmax: pick best prompt under DP
# ------------------------------
def dp_argmax(
    candidate_prompts, validation_dataset: Dataset,
    local_engine: LLM, local_tokenizer: AutoTokenizer,
    epsilon: float, delta: float,
    gen_temp: float, rep_penalty: float, max_new_tokens: int, chance_prob: float
):
    accuracies = []
    val_size = 0
    for batch in validation_dataset:
        val_size += len(batch["text"])

    for prompt in candidate_prompts:
        acc = evaluate(
            prompt, validation_dataset,
            local_engine, local_tokenizer,
            gen_temp, rep_penalty, chance_prob
        )
        accuracies.append(acc)

    accuracies = np.array(accuracies)

    if epsilon == float("inf"):
        best_index = np.argmax(accuracies)
        return candidate_prompts[best_index]
    else:
        # Pr[prompt_i is best] \propto exp(epsilon * accuracy_i * val_size)
        scores = epsilon * accuracies * val_size
        max_score = np.max(scores)
        stable_scores = scores - max_score  # for numerical stability
        probabilities = np.exp(stable_scores)
        probabilities /= np.sum(probabilities)
        best_index = np.random.choice(len(candidate_prompts), p=probabilities)
        return candidate_prompts[best_index]


# ------------------------------
# Evaluate a prompt on a dataset
# ------------------------------
def evaluate(
    prompt: str, val_dataset: Dataset,
    server_engine: LLM, server_tokenizer: AutoTokenizer,
    gen_temp: float, rep_penalty: float, chance_prob: float,
) -> float:
    correct, total = 0, 0
    loader = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate)
    # for batch in tqdm(loader, desc="Evaluating", unit="batch"):
    #     for i, input_text in enumerate(batch["text"]):
    #         gt = str(batch["labels"][i].item())
    #         messages = [
    #             {"role": "system", "content": prompt},
    #             {"role": "user", "content": input_text}
    #         ]
    #         prompt_str = server_tokenizer.apply_chat_template(
    #             messages, tokenize=False,
    #             add_generation_prompt=True,
    #             truncation=True
    #         )
    #         continuation = vllm_generate(
    #             server_engine, prompt_str,
    #             max_new_tokens=max_new_tokens,
    #             temperature=gen_temp,
    #             repetition_penalty=rep_penalty
    #         )
    #         # extract answer from \boxed{...}
    #         m = re.search(r"\\boxed\{(.+?)\}", continuation)
    #         ans = m.group(1).strip() if m else continuation.strip()
    #         if m and gt.lower() == ans.lower():
    #             correct += 1
    #         total += 1

    print("Evaluating...")
    all_prompts = []
    for batch in loader:
        for i, input_text in enumerate(batch["text"]):
            gt = str(batch["labels"][i].item())
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ]
            prompt_str = server_tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                truncation=True
            )
            # print(f"\nPrompt:\n{prompt_str}")
            all_prompts.append(prompt_str)

    continuation = vllm_generate(
        server_engine, all_prompts,
        max_new_tokens=256,
        temperature=gen_temp,
        repetition_penalty=rep_penalty
    )

    continuation_pointer = 0

    for batch in loader:
        for i, input_text in enumerate(batch["text"]):
            gt = str(batch["labels"][i].item())
            output = continuation[continuation_pointer].outputs[0].text
            continuation_pointer += 1
            answer_found = False
            ans = None

            boxed_matches = re.findall(r"\\boxed\{(.+?)\}", output)
            if boxed_matches:
                ans = boxed_matches[-1].strip()
                answer_found = True

            if not answer_found:
                ans = output.strip()
                # If no boxed answer is found, we add chance level accuracy evaluation.
                if random.random() < chance_prob:
                    correct += 1
                total += 1
                continue

            norm_gt = re.sub(r"\s+", "", gt.lower())
            norm_ans = re.sub(r"\s+", "", ans.lower())

            if norm_gt == norm_ans:
                correct += 1

            total += 1

    return correct / total if total > 0 else 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Main DP-OPT workflow
# ------------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="DP-OPT: Differentially-Private Prompt Optimization with vLLM"
    )
    # generation hyperparams
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument('--num_prompt', default=5, type=int)
    parser.add_argument('--num_demos', default=5, type=int)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--ensemble_num', default=205, type=int)
    parser.add_argument('--local_batch_size', default=32, type=int)
    parser.add_argument('--server_batch_size', default=32, type=int)
    parser.add_argument('--gen_temp', default=0.7, type=float)
    parser.add_argument('--rep_penalty', default=1.2, type=float)
    parser.add_argument('--tokenwise_gen', default=True, type=str2bool)
    # DP hyperparams
    parser.add_argument('--dp_eps', required=True, type=str)
    parser.add_argument('--dp_delta', default=None, type=float)
    parser.add_argument('--target_eps', default=None, type=float)
    parser.add_argument('--q', default=0.1, type=float)
    # data + model
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, required=True)
    parser.add_argument('--n_train_instances', type=int, default=None)
    parser.add_argument('--n_test_instances', type=int, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--local_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--server_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.dp_eps.lower() == 'inf':
        args.dp_eps = float('inf')
    else:
        try:
            args.dp_eps = float(args.dp_eps)
        except ValueError:
            raise ValueError(
                "Invalid value for --dp_eps. It should be a number or 'inf'."
            )

    start_time = time.perf_counter()

    # load tokenizers only
    local_tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    if local_tokenizer.pad_token_id is None:
        local_tokenizer.pad_token = local_tokenizer.eos_token
    server_tokenizer = AutoTokenizer.from_pretrained(args.server_model)
    if server_tokenizer.pad_token_id is None:
        server_tokenizer.pad_token = server_tokenizer.eos_token


    local_engine_kwargs = {
        "model": args.local_model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_num_seqs": args.local_batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }

    server_engine_kwargs = {
        "model": args.server_model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_num_seqs": args.server_batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }

    local_engine = LLM(**local_engine_kwargs)
    # local_engine.sleep(level=2)
    server_engine = LLM(**server_engine_kwargs)
    # server_engine.sleep(level=2)

    # default file paths
    if args.train_file is None:
        args.train_file = f"data/{args.dataset_type}/text/{args.dataset_type}_train_text.csv"
    if args.test_file is None:
        args.test_file = f"data/{args.dataset_type}/text/{args.dataset_type}_test_text.csv"

    # load + split train/val
    train_df = pd.read_csv(args.train_file)
    if args.n_train_instances:
        train_df = train_df.head(args.n_train_instances)
    val_df = train_df.sample(frac=args.val_ratio, random_state=42)
    train_df_rem = train_df.drop(val_df.index)

    # build datasets
    train_dataset = get_dataset_for_evaluation(args.dataset_type, train_df_rem)
    val_dataset = get_dataset_for_evaluation(args.dataset_type, val_df)
    test_dataset = get_dataset_for_evaluation(
        args.dataset_type, args.test_file, n_instances=args.n_test_instances
    )

    if args.dp_delta is None:
        args.dp_delta = 1 / len(train_dataset)

    # initial prompt by dataset
    if args.dataset_type in ["titanic", "bank_marketing"]:
        current_prompt = (
            "Answer the following question. Your answer MUST be either 0 (No) or 1 (Yes). Enclose ONLY the integer in \\boxed{...}."
        )
        chance_prob = 0.5
    elif args.dataset_type == "wine_quality":
        current_prompt = (
            "Answer the following question. Your answer MUST be an integer between 0 and 10 (inclusive), where a larger integer indicates better wine quality. Enclose ONLY the integer in \\boxed{...}."
        )
        chance_prob = 1.0 / 11
    elif args.dataset_type == "nursery":
        # current_prompt = (
        #     "Answer the following question. Your answer MUST be an integer (0, 1, 2, or 3) corresponding to one of these categories: "
        #     "{0: 'spec_prior', 1: 'priority', 2: 'very_recom', 3: 'not_recom'}. Enclose your answer in \\boxed{...}."
        # )
        current_prompt = (
            "Answer the following question. Your answer MUST be an integer (0, 1, 2, or 3) corresponding to one of these categories: 0 – special priority (e.g. veterans, siblings), 1 – priority (e.g. staff children, local), 2 – very recommended (strong applicants), 3 – not recommended (weak applicants). Enclose ONLY the integer in \\boxed{...}."
        )
        chance_prob = 0.25
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    # iterative prompt optimization
    for step in range(args.steps):
        # local_engine.wake_up()
        # local_engine = LLM(**local_engine_kwargs)
        print(f"\n=== Step {step + 1}/{args.steps} ===")
        # 1) generate predictions on training split
        predicted_data = []
        train_loader = DataLoader(
            train_dataset, batch_size=args.local_batch_size,
            shuffle=True, collate_fn=custom_collate
        )
        all_prompts = []

        # for batch in train_loader, desc="Generating preds", unit="batch"):
        for batch in train_loader:
            for i, input_text in enumerate(batch["text"]):
                gt = str(batch["labels"][i].item())
                messages = [
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": input_text}
                ]
                prompt_str = local_tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                    truncation=True
                )
                # pred = vllm_generate(
                #     local_engine, prompt_str,
                #     max_new_tokens=50,
                #     temperature=args.gen_temp,
                #     repetition_penalty=args.rep_penalty
                # )
                # predicted_data.append((input_text, gt, pred))
                all_prompts.append(prompt_str)

        preds = vllm_generate(
            local_engine, all_prompts,
            max_new_tokens=50,
            temperature=0.0,
            repetition_penalty=args.rep_penalty
        )

        preds_pointer = 0

        for batch in train_loader:
            for i, input_text in enumerate(batch["text"]):
                gt = str(batch["labels"][i].item())
                messages = [
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": input_text}
                ]
                prompt_str = local_tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                    truncation=True
                )
                pred = preds[preds_pointer].outputs[0].text
                predicted_data.append((input_text, gt, pred))
                preds_pointer += 1


        # 2) generate candidate prompts
        candidate_prompts = []

        for i in range(args.num_prompt):
            if args.tokenwise_gen:
                ids = dp_ensgen(
                    local_engine, local_tokenizer,
                    predicted_data, current_prompt,
                    args.dp_eps, args.dp_delta, args.q,
                    args.max_new_tokens, args.ensemble_num,
                    args.gen_temp, args.rep_penalty, args.num_demos
                )
                cand = local_tokenizer.decode(
                    torch.tensor(ids), skip_special_tokens=True
                )
            else:
                messages = [
                    {"role": "system", "content": current_prompt},
                    {"role": "user",
                     "content": "Please enhance or optimize the instruction for the above task."}
                ]
                prompt_str = local_tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                    truncation=True
                )
                cand = prompt_str + vllm_generate(
                    local_engine, [prompt_str],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.gen_temp,
                    repetition_penalty=args.rep_penalty
                )[0].outputs[0].text
            print(f"\nCandidate {i+1}:\n{cand}")
            candidate_prompts.append(cand)

        # del local_engine
        # torch.cuda.empty_cache()
        # server_engine = LLM(**server_engine_kwargs)

        # 3) DP-argmax on validation split
        best = dp_argmax(
            candidate_prompts, val_dataset,
            local_engine, local_tokenizer,
            args.dp_eps, args.dp_delta,
            0.0, args.rep_penalty, args.max_new_tokens, chance_prob
        )
        print(f"\nSelected prompt:\n{best}")
        current_prompt = best

        # server_engine.sleep(level=2)
        # del server_engine
        # torch.cuda.empty_cache()

    # server_engine = LLM(**server_engine_kwargs)

    # final test eval
    acc = evaluate(
        current_prompt, test_dataset,
        server_engine, server_tokenizer,
        0.0, args.rep_penalty, chance_prob,
    )
    print(f"\nFinal test accuracy: {acc*100:.2f}%")

    # server_engine.sleep(level=2)
    # del server_engine
    # torch.cuda.empty_cache()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()