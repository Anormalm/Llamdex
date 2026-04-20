import argparse
import csv
import json
import os
import re
from typing import Dict, List

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.data.cifar_qa import CIFARSingleImageQADataset, collate_single_image
from src.multimodal.task_matrix.datasets import GroundedGenerationDataset, collate_task_batch
from src.multimodal.task_matrix.metrics import keyword_consistency_score
from src.multimodal.utils.repro import set_seed


def _device(name: str) -> torch.device:
    return torch.device(name if torch.cuda.is_available() and name.startswith("cuda") else "cpu")


def _extract_answer_rationale(text: str):
    t = str(text)
    ans_matches = re.findall(r"(?im)\banswer\s*:\s*([^\n\.]+)", t)
    rat_matches = re.findall(r"(?ims)\brationale\s*:\s*(.+?)(?=\banswer\s*:|$)", t)
    answer_text = ans_matches[-1].strip() if ans_matches else ""
    rationale_text = rat_matches[-1].strip() if rat_matches else ""
    if not answer_text:
        m = re.search(r"([A-Za-z0-9!@#$%^&*()\[\]{}<>?/|])", t)
        answer_text = m.group(1) if m else ""
    if not rationale_text:
        rationale_text = t.strip()
    return answer_text, rationale_text


def _normalize_code_text(s: str) -> str:
    m = re.search(r"[A-Za-z0-9!@#$%^&*()\\[\\]{}<>?/|]", str(s))
    return m.group(0) if m else ""


def _build_model_and_tokenizer(args, dev: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.server_models_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = DomainQwenForCausalLM.from_pretrained_qwen(
        args.model_name,
        cache_dir=args.server_models_path,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        trust_remote_code=True,
        tokenizer=tokenizer,
    )
    for p in model.parameters():
        p.requires_grad = False

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.base_model = get_peft_model(model.base_model, lcfg)
    model = model.to(dev)
    if dev.type == "cuda":
        model = model.to(torch.bfloat16)
    return model, tokenizer


def _train_lora(model, loader, dev: torch.device, train_steps: int, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    step = 0
    while step < train_steps:
        for batch in loader:
            tokens = batch["prompt_tokens"].to(dev)
            mask = batch["prompt_mask"].to(dev)
            targets = batch["target_token_id"].to(dev)
            out = model(tokens, attention_mask=mask, use_cache=False)
            logits = out.logits[:, -1, :]
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= train_steps:
                break


def _eval_grounded(model, tokenizer, loader, dev: torch.device, max_new_tokens: int) -> Dict[str, float]:
    model.eval()
    total = 0
    fmt_ok = 0
    rationale_nonempty = 0
    faithful = 0
    halluc = 0
    answer_valid = 0
    answer_correct = 0
    kw_scores: List[float] = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["prompt_tokens"].to(dev)
            mask = batch["prompt_mask"].to(dev)
            targets = batch["target_token_id"].to(dev)
            kws_all = batch.get("rationale_keywords", [[] for _ in range(tokens.size(0))])

            for i in range(tokens.size(0)):
                inp = tokens[i : i + 1]
                attn = mask[i : i + 1]
                gen = model.base_model.generate(
                    input_ids=inp,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                )
                cont = gen[0, inp.size(1) :]
                text = tokenizer.decode(cont.tolist(), skip_special_tokens=True)
                ans_text, rat_text = _extract_answer_rationale(text)
                if ans_text and rat_text:
                    fmt_ok += 1
                if len(rat_text) >= 12:
                    rationale_nonempty += 1

                pred_code = _normalize_code_text(ans_text)
                true_code = _normalize_code_text(tokenizer.decode([int(targets[i].item())], skip_special_tokens=True))
                if pred_code:
                    answer_valid += 1
                if pred_code and pred_code == true_code:
                    answer_correct += 1

                kw_score = keyword_consistency_score(rat_text or text, kws_all[i])
                kw_scores.append(float(kw_score))
                if kw_score <= 0.0:
                    halluc += 1
                if kw_score >= 0.6 and pred_code == true_code and len(rat_text) >= 12:
                    faithful += 1
                total += 1

    return {
        "samples": total,
        "format_compliance": fmt_ok / max(1, total),
        "rationale_nonempty_rate": rationale_nonempty / max(1, total),
        "answer_code_valid_rate": answer_valid / max(1, total),
        "answer_code_accuracy": answer_correct / max(1, total),
        "rationale_consistency": (sum(kw_scores) / max(1, len(kw_scores))) if kw_scores else 0.0,
        "rationale_faithful_rate": faithful / max(1, total),
        "rationale_hallucination_rate": halluc / max(1, total),
    }


def parse_args():
    p = argparse.ArgumentParser(description="LoRA grounded-rationale evaluator")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_steps", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_new_tokens", type=int, default=28)
    p.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/lora_grounded_generation.remote.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/lora_grounded_generation.remote.json")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dev = _device(args.device)
    model, tokenizer = _build_model_and_tokenizer(args, dev)

    train_ds = CIFARSingleImageQADataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=True,
        dataset_name="dtd",
        mode="vision",
        qa_type="label_code",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_single_image)
    _train_lora(model, train_loader, dev, args.train_steps, args.lr)

    eval_ds = GroundedGenerationDataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=False,
        max_samples=args.max_eval_samples,
        dataset_name="dtd",
    )
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_task_batch)
    m = _eval_grounded(model, tokenizer, eval_loader, dev, args.max_new_tokens)

    row = {
        "benchmark_family": "architecture",
        "benchmark_source": "lora_finetune",
        "baseline_name": "lora_qwen35_9b",
        "task": "grounded_generation",
        "dataset": "dtd",
        "backbone": args.model_name,
        "fusion_policy": "none",
        "status": "ok",
        "seed": args.seed,
        **m,
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump([row], f, indent=2)

    print("CSV:", args.out_csv)
    print("JSON:", args.out_json)
    print("faithful=", m["rationale_faithful_rate"], "halluc=", m["rationale_hallucination_rate"])


if __name__ == "__main__":
    main()
