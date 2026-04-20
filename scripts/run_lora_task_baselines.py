import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.task_matrix.metrics import macro_f1_from_ints, mean_absolute_error
from src.multimodal.tasks.parsing import parse_population_answer, population_bin_to_fraction_midpoint
from src.multimodal.utils.repro import set_seed


@dataclass
class RunSpec:
    dataset_name: str
    task_family: str
    qa_type: str


def _device(name: str) -> torch.device:
    return torch.device(name if torch.cuda.is_available() and name.startswith("cuda") else "cpu")


def _single_token_ids(tokenizer, text: str) -> List[int]:
    ids = set()
    for v in (text, text.lower(), text.upper()):
        for p in ("", " "):
            t = tokenizer.encode(p + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(int(t[0]))
    return sorted(ids)


def _build_loader(
    *,
    data_root: str,
    tokenizer,
    dataset_name: str,
    task_family: str,
    qa_type: str,
    population_output_mode: str,
    population_group_size: int,
    max_samples: int,
    batch_size: int,
    train: bool,
    seed: int,
):
    if task_family == "single_image":
        ds = CIFARSingleImageQADataset(
            root=data_root,
            tokenizer=tokenizer,
            train=train,
            dataset_name=dataset_name,
            mode="vision",
            qa_type=qa_type,
            max_samples=max_samples,
            seed=seed,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=train, collate_fn=collate_single_image)

    ds = CIFARPopulationDataset(
        root=data_root,
        tokenizer=tokenizer,
        train=train,
        dataset_name=dataset_name,
        group_size=population_group_size,
        output_mode=population_output_mode,
        max_groups=max_samples,
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, collate_fn=collate_population)


def _build_model_and_tokenizer(
    *,
    model_name: str,
    server_models_path: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    dev: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=server_models_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = DomainQwenForCausalLM.from_pretrained_qwen(
        model_name,
        cache_dir=server_models_path,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        trust_remote_code=True,
        tokenizer=tokenizer,
    )
    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.base_model = get_peft_model(model.base_model, cfg)
    model.base_model.print_trainable_parameters()

    model = model.to(dev)
    if dev.type == "cuda":
        model = model.to(torch.bfloat16)
    model.train()
    return model, tokenizer


def _train(
    *,
    model,
    loader,
    dev: torch.device,
    train_steps: int,
    lr: float,
):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
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


def _eval(
    *,
    model,
    loader,
    tokenizer,
    task_family: str,
    qa_type: str,
    population_output_mode: str,
    dev: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    pop_true: List[float] = []
    pop_pred: List[float] = []

    yes_ids = set(_single_token_ids(tokenizer, "Yes"))
    no_ids = set(_single_token_ids(tokenizer, "No"))

    with torch.no_grad():
        for batch in loader:
            tokens = batch["prompt_tokens"].to(dev)
            mask = batch["prompt_mask"].to(dev)
            out = model(tokens, attention_mask=mask, use_cache=False)
            pred_ids = out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            gt_ids = batch["target_token_id"].cpu().tolist()

            if task_family == "single_image" and qa_type in {"yesno", "yesno_set2"}:
                gt_yesno = [1 if str(a).lower().startswith("yes") else 0 for a in batch["answer_text"]]
                pred_yesno = []
                for p in pred_ids:
                    if int(p) in yes_ids:
                        pred_yesno.append(1)
                    elif int(p) in no_ids:
                        pred_yesno.append(0)
                    else:
                        pred_yesno.append(-1)
                for p, g in zip(pred_yesno, gt_yesno):
                    correct += int(p == g)
                    total += 1
                    y_pred.append(int(max(0, p)))
                    y_true.append(int(g))
                continue

            if task_family == "single_image":
                for p, g in zip(pred_ids, gt_ids):
                    correct += int(int(p) == int(g))
                    total += 1
                    y_pred.append(int(p))
                    y_true.append(int(g))
                continue

            # population
            pred_texts = [tokenizer.decode([int(pid)], skip_special_tokens=True).strip() for pid in pred_ids]
            pred_bins = [parse_population_answer(x, mode=population_output_mode) for x in pred_texts]
            gt_bins = [parse_population_answer(x, mode=population_output_mode) for x in batch["answer_text"]]
            for pb, gb in zip(pred_bins, gt_bins):
                if pb is None:
                    pb = 0
                if gb is None:
                    gb = 0
                correct += int(int(pb) == int(gb))
                total += 1
                pop_pred.append(population_bin_to_fraction_midpoint(int(pb), mode=population_output_mode))
                pop_true.append(population_bin_to_fraction_midpoint(int(gb), mode=population_output_mode))

    out = {"accuracy": float(correct / max(1, total))}
    if task_family == "population":
        out["mae"] = float(mean_absolute_error(pop_true, pop_pred))
        out["metric_name"] = "mae"
        out["metric"] = out["mae"]
    else:
        out["f1"] = float(macro_f1_from_ints(y_true, y_pred, num_classes=max(2, len(set(y_true))))) if y_true else 0.0
        out["metric_name"] = "accuracy"
        out["metric"] = out["accuracy"]
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run LoRA baselines on Llamdex task protocol.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_steps", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--population_group_size", type=int, default=8)
    p.add_argument("--population_output_mode", type=str, default="integer")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/lora_task_baselines.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/lora_task_baselines.json")
    p.add_argument("--datasets", type=str, nargs="+", default=["dtd", "oxford_pet", "cifar10"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dev = _device(args.device)

    specs = [
        RunSpec(dataset_name="dtd", task_family="single_image", qa_type="label_code"),
        RunSpec(dataset_name="oxford_pet", task_family="single_image", qa_type="label_code"),
        RunSpec(dataset_name="cifar10", task_family="single_image", qa_type="label_code"),
        RunSpec(dataset_name="dtd", task_family="single_image", qa_type="yesno"),
        RunSpec(dataset_name="oxford_pet", task_family="population", qa_type="label_code"),
    ]
    specs = [s for s in specs if s.dataset_name in set(args.datasets)]

    rows: List[Dict] = []
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    for spec in specs:
        print(f"[lora] start dataset={spec.dataset_name} task_family={spec.task_family} qa_type={spec.qa_type}")
        model, tokenizer = _build_model_and_tokenizer(
            model_name=args.model_name,
            server_models_path=args.server_models_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            dev=dev,
        )
        train_loader = _build_loader(
            data_root=args.data_root,
            tokenizer=tokenizer,
            dataset_name=spec.dataset_name,
            task_family=spec.task_family,
            qa_type=spec.qa_type,
            population_output_mode=args.population_output_mode,
            population_group_size=args.population_group_size,
            max_samples=args.max_train_samples,
            batch_size=args.batch_size,
            train=True,
            seed=args.seed,
        )
        eval_loader = _build_loader(
            data_root=args.data_root,
            tokenizer=tokenizer,
            dataset_name=spec.dataset_name,
            task_family=spec.task_family,
            qa_type=spec.qa_type,
            population_output_mode=args.population_output_mode,
            population_group_size=args.population_group_size,
            max_samples=args.max_eval_samples,
            batch_size=args.batch_size,
            train=False,
            seed=args.seed + 1,
        )
        _train(model=model, loader=train_loader, dev=dev, train_steps=args.train_steps, lr=args.lr)
        m = _eval(
            model=model,
            loader=eval_loader,
            tokenizer=tokenizer,
            task_family=spec.task_family,
            qa_type=spec.qa_type,
            population_output_mode=args.population_output_mode,
            dev=dev,
        )

        if spec.task_family == "population":
            task_name = "population"
        elif spec.qa_type in {"yesno", "yesno_set2"}:
            task_name = "strict_yesno"
        else:
            task_name = "finegrained"

        row = {
            "benchmark_family": "architecture",
            "benchmark_source": "lora_finetune",
            "baseline_name": "lora_qwen35_9b",
            "task": task_name,
            "dataset": spec.dataset_name,
            "backbone": args.model_name,
            "fusion_policy": "none",
            "bundle_id": "",
            "layer_idx": "",
            "k": "",
            "evidence_dim": "",
            "metric_name": m["metric_name"],
            "metric": float(m["metric"]),
            "metric_value": float(m["metric"]),
            "status": "ok",
            "seed": args.seed,
            "seed_count": 1,
            "error": "",
            "notes": "LoRA-only baseline trained on prompt->answer next-token objective",
            "accuracy": float(m.get("accuracy", 0.0)),
            "f1": float(m.get("f1", 0.0)),
            "mae": float(m.get("mae", 0.0)),
        }
        rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # placeholder grounded row until dedicated generation LoRA eval path is added
    rows.append(
        {
            "benchmark_family": "architecture",
            "benchmark_source": "lora_finetune",
            "baseline_name": "lora_qwen35_9b",
            "task": "grounded_generation",
            "dataset": "dtd",
            "backbone": args.model_name,
            "fusion_policy": "none",
            "bundle_id": "",
            "layer_idx": "",
            "k": "",
            "evidence_dim": "",
            "metric_name": "rationale_faithful_rate",
            "metric": 0.0,
            "metric_value": 0.0,
            "status": "available_not_run",
            "seed": args.seed,
            "seed_count": 0,
            "error": "",
            "notes": "Grounded-generation LoRA eval path not yet wired in this runner.",
            "accuracy": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }
    )

    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[lora] wrote {len(rows)} rows")
    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
