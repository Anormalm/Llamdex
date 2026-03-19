from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, build_expert_encoder
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.tasks.parsing import (
    parse_population_answer,
    population_bin_to_fraction_midpoint,
)
from src.multimodal.utils.repro import set_seed


@dataclass
class AblationConfig:
    server_models_path: str = "/disk1/lfhu/hf_cache"
    model_name: str = "Qwen/Qwen3.5-9B"
    dataset: str = "dtd"
    data_root: str = "./data"
    task: str = "single"  # single|population
    qa_type: str = "label_code"
    population_output_mode: str = "integer"
    population_group_size: int = 8
    expert_type: str = "clip"  # clip|siglip|groundingdino_sam2|xgboost|ft_transformer|tabpfn
    expert_model_id: Optional[str] = None
    expert_model_path: Optional[str] = None
    expert_output_dim: int = 512
    use_runtime_detector: bool = False
    backbone: str = "domain_qwen"
    k: int = 4
    layer_idx: int = 0
    evidence_dim: int = 256
    alpha: float = 1.0
    batch_size: int = 8
    max_train_samples: Optional[int] = 1024
    max_eval_samples: Optional[int] = 256
    train_steps: int = 100
    learning_rate: float = 2e-4
    seed: int = 42
    device: str = "cuda"
    out_csv: str = "runs/multimodal_replaceability_results.csv"
    out_json: str = "runs/multimodal_replaceability_results.json"


def _device(device: str):
    return torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")


def _build_model_tokenizer(cfg: AblationConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.server_models_path,
        torch_dtype=torch.bfloat16,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        cfg.model_name,
        cache_dir=cfg.server_models_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        tokenizer=tokenizer,
    )
    model.num_tokens = int(cfg.k)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def _build_dataloaders(cfg: AblationConfig, tokenizer):
    if cfg.task == "single":
        train_ds = CIFARSingleImageQADataset(
            root=cfg.data_root,
            tokenizer=tokenizer,
            train=True,
            dataset_name=cfg.dataset,
            mode="vision",
            qa_type=cfg.qa_type,
            seed=cfg.seed,
            max_samples=cfg.max_train_samples,
        )
        eval_ds = CIFARSingleImageQADataset(
            root=cfg.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=cfg.dataset,
            mode="vision",
            qa_type=cfg.qa_type,
            seed=cfg.seed,
            max_samples=cfg.max_eval_samples,
        )
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_single_image),
            DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_single_image),
        )

    train_ds = CIFARPopulationDataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=True,
        dataset_name=cfg.dataset,
        group_size=cfg.population_group_size,
        output_mode=cfg.population_output_mode,
        seed=cfg.seed,
        max_groups=cfg.max_train_samples,
    )
    eval_ds = CIFARPopulationDataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=False,
        dataset_name=cfg.dataset,
        group_size=cfg.population_group_size,
        output_mode=cfg.population_output_mode,
        seed=cfg.seed,
        max_groups=cfg.max_eval_samples,
    )
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_population),
        DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_population),
    )


def _expert_input_from_batch(cfg: AblationConfig, batch: Dict, dev: torch.device):
    if cfg.expert_type in {"xgboost", "ft_transformer", "tabpfn", "tabular"}:
        if "tabular" in batch:
            return {"tabular": batch["tabular"].to(dev)}
        # Fallback adapter: derive lightweight tabular features from image statistics.
        imgs = batch["images"].to(dev).float()
        if imgs.ndim == 5:
            imgs = imgs.mean(dim=1)
        mean_rgb = imgs.mean(dim=(2, 3))
        std_rgb = imgs.std(dim=(2, 3))
        feat = torch.cat([mean_rgb, std_rgb], dim=1)
        return {"tabular": feat}
    if cfg.task == "single":
        return {"images": batch["images"].to(dev)}
    return {"images": batch["images"].to(dev)}


def _primary_metric_name(cfg: AblationConfig) -> str:
    return "accuracy" if cfg.task == "single" else "mae"


def _evaluate(cfg: AblationConfig, model, loader, tokenizer, dev) -> Dict[str, float]:
    from src.multimodal.tasks.parsing import parse_population_answer

    model.eval()
    correct = 0
    total = 0
    maes = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["prompt_tokens"].to(dev)
            mask = batch["prompt_mask"].to(dev)
            outputs = model(tokens, attention_mask=mask, expert_inputs=(_expert_input_from_batch(cfg, batch, dev),), use_cache=False)
            pred_ids = outputs.logits[:, -1, :].argmax(dim=-1)
            if cfg.task == "single":
                targets = batch["target_token_id"].to(dev)
                correct += (pred_ids == targets).sum().item()
                total += targets.numel()
            else:
                pred_texts = [tokenizer.decode([int(pid)]).strip() for pid in pred_ids.cpu().tolist()]
                pred_bins = [parse_population_answer(x, mode=cfg.population_output_mode) for x in pred_texts]
                gt_bins = [parse_population_answer(x, mode=cfg.population_output_mode) for x in batch["answer_text"]]
                for p, g in zip(pred_bins, gt_bins):
                    if p is not None and g is not None:
                        pf = population_bin_to_fraction_midpoint(p, mode=cfg.population_output_mode)
                        gf = population_bin_to_fraction_midpoint(g, mode=cfg.population_output_mode)
                        maes.append(abs(pf - gf))
                        correct += int(int(p) == int(g))
                    total += 1
    if cfg.task == "single":
        return {"accuracy": correct / max(1, total)}
    return {"bin_accuracy": correct / max(1, total), "mae": float(np.mean(maes) if maes else 0.0)}


def run_single_experiment(cfg: AblationConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    dev = _device(cfg.device)
    model, tokenizer = _build_model_tokenizer(cfg)
    train_loader, eval_loader = _build_dataloaders(cfg, tokenizer)

    enc = build_expert_encoder(
        ExpertEncoderSpec(
            expert_type=cfg.expert_type,
            model_id=cfg.expert_model_id,
            model_path=cfg.expert_model_path,
            output_dim=cfg.expert_output_dim,
            cache_dir=cfg.server_models_path,
            use_runtime_detector=cfg.use_runtime_detector,
        )
    )
    builder = EncoderEvidenceBuilder(expert_encoder=enc, evidence_dim=cfg.evidence_dim, output_dim=enc.output_dim)
    projector = EvidenceProjector(
        evidence_dim=cfg.evidence_dim,
        hidden_size=model.config.hidden_size,
        num_tokens=cfg.k,
        alpha=cfg.alpha,
    )
    semantic = SemanticEvidenceDomainExpert(builder, projector)
    model.model.layers[cfg.layer_idx].add_expert_(semantic, map_to_expert_emb=None)

    model = model.to(dev).to(torch.bfloat16 if dev.type == "cuda" else torch.float32)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    model.train()
    if cfg.train_steps > 0:
        while step < cfg.train_steps:
            for batch in train_loader:
                tokens = batch["prompt_tokens"].to(dev)
                mask = batch["prompt_mask"].to(dev)
                targets = batch["target_token_id"].to(dev)
                outputs = model(
                    tokens,
                    attention_mask=mask,
                    expert_inputs=(_expert_input_from_batch(cfg, batch, dev),),
                    use_cache=False,
                )
                logits = outputs.logits[:, -1, :]
                loss = loss_fn(logits, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step >= cfg.train_steps:
                    break

    metrics = _evaluate(cfg, model, eval_loader, tokenizer, dev)
    primary_name = _primary_metric_name(cfg)
    metric_value = float(metrics[primary_name])
    row = {
        "expert_type": cfg.expert_type,
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "k": int(cfg.k),
        "layer_idx": int(cfg.layer_idx),
        "evidence_dim": int(cfg.evidence_dim),
        "metric": metric_value,
        "metric_name": primary_name,
    }
    row.update(metrics)
    return row


def _write_rows(rows: List[Dict], out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_ablation_grid(
    base_cfg: AblationConfig,
    k_grid: List[int],
    layer_grid: List[int],
    evidence_dim_grid: List[int],
) -> List[Dict]:
    rows = []
    for k in k_grid:
        for layer in layer_grid:
            for ed in evidence_dim_grid:
                cfg = AblationConfig(**asdict(base_cfg))
                cfg.k = int(k)
                cfg.layer_idx = int(layer)
                cfg.evidence_dim = int(ed)
                row = run_single_experiment(cfg)
                rows.append(row)
                print(row)
    _write_rows(rows, base_cfg.out_csv, base_cfg.out_json)
    return rows
