from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, TabularExpertEncoder, build_expert_encoder
from src.multimodal.framework.runner import AblationConfig, run_single_experiment
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.utils.repro import set_seed


@dataclass
class BackboneSpec:
    name: str
    model_name: str
    server_models_path: str = "/disk1/lfhu/hf_cache"
    enabled: bool = True


@dataclass
class BackboneScalingConfig:
    # Fixed task and expert (for "fixed expert, vary backbone")
    dataset: str = "dtd"
    data_root: str = "./data"
    task: str = "single"  # single|population
    qa_type: str = "label_code"
    population_output_mode: str = "integer"
    population_group_size: int = 8
    expert_type: str = "clip"
    expert_model_id: Optional[str] = None
    expert_model_path: Optional[str] = None
    expert_output_dim: int = 512
    use_runtime_detector: bool = False
    backbone: str = "domain_qwen"
    # Base operating point
    base_k: int = 4
    base_layer_idx: int = 0
    base_evidence_dim: int = 256
    alpha: float = 1.0
    batch_size: int = 8
    max_train_samples: int = 512
    max_eval_samples: int = 256
    train_steps: int = 50
    learning_rate: float = 2e-4
    # Sensitivity sweeps
    k_grid: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    layer_grid: List[int] = field(default_factory=lambda: [0])
    evidence_dim_grid: List[int] = field(default_factory=lambda: [128, 256, 512])
    # Pure-language regression
    language_prompts: List[str] = field(
        default_factory=lambda: [
            "Summarize why deterministic seeds matter in model evaluation.",
            "Explain overfitting in one short paragraph.",
            "Write one sentence about why calibration matters.",
            "List three risks of noisy labels.",
        ]
    )
    language_max_new_tokens: int = 32
    language_degradation_threshold: float = 0.25
    language_tabular_dim: int = 16
    # Backbone list
    backbones: List[BackboneSpec] = field(default_factory=list)
    # Output
    out_dir: str = "runs/backbone_scaling_suite"
    seed: int = 42
    device: str = "cuda"


def _device(device: str) -> torch.device:
    return torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")


def _ensure_default_backbones(cfg: BackboneScalingConfig) -> None:
    if cfg.backbones:
        return
    cfg.backbones = [
        BackboneSpec(name="qwen3.5-9b", model_name="Qwen/Qwen3.5-9B"),
        BackboneSpec(name="llama3-8b", model_name="meta-llama/Meta-Llama-3-8B-Instruct"),
        BackboneSpec(name="qwen2.5-7b", model_name="Qwen/Qwen2.5-7B-Instruct"),
        BackboneSpec(name="qwen2.5-14b", model_name="Qwen/Qwen2.5-14B-Instruct"),
        BackboneSpec(name="llama3-70b-sanity", model_name="meta-llama/Meta-Llama-3-70B-Instruct", enabled=False),
    ]


def _token_jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def _greedy_generate(model, input_ids, attention_mask, max_new_tokens: int, expert_inputs=None):
    seq = input_ids
    mask = attention_mask
    for _ in range(max_new_tokens):
        out = model(seq, attention_mask=mask, expert_inputs=expert_inputs, use_cache=False)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
    return seq


def _build_base_model(backbone: BackboneSpec, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        backbone.model_name,
        cache_dir=backbone.server_models_path,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        backbone.model_name,
        cache_dir=backbone.server_models_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        tokenizer=tokenizer,
    )
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.eval()
    return model, tokenizer


def _attach_injection_for_language(
    model,
    evidence_dim: int,
    num_tokens: int,
    layer_idx: int,
    alpha: float,
    tabular_dim: int,
):
    tab = TabularExpertEncoder(model_kind="tabular", model_path=None, output_dim=tabular_dim)
    builder = EncoderEvidenceBuilder(expert_encoder=tab, evidence_dim=evidence_dim, output_dim=tab.output_dim)
    projector = EvidenceProjector(
        evidence_dim=evidence_dim,
        hidden_size=model.config.hidden_size,
        num_tokens=num_tokens,
        alpha=alpha,
    )
    expert = SemanticEvidenceDomainExpert(builder, projector)
    model.num_tokens = num_tokens
    model.model.layers[layer_idx].add_expert_(expert, map_to_expert_emb=None)


def _language_regression(backbone: BackboneSpec, cfg: BackboneScalingConfig) -> Dict:
    dev = _device(cfg.device)
    baseline_model, tokenizer = _build_base_model(backbone, dev)
    injected_model, _ = _build_base_model(backbone, dev)
    _attach_injection_for_language(
        injected_model,
        evidence_dim=cfg.base_evidence_dim,
        num_tokens=cfg.base_k,
        layer_idx=cfg.base_layer_idx,
        alpha=cfg.alpha,
        tabular_dim=cfg.language_tabular_dim,
    )
    injected_model = injected_model.to(dev).to(torch.bfloat16 if dev.type == "cuda" else torch.float32)
    injected_model.eval()

    sim_scores = []
    for p in cfg.language_prompts:
        msgs = [{"role": "user", "content": p}]
        inp = tokenizer.apply_chat_template(msgs, return_tensors="pt").to(dev)
        mask = torch.ones_like(inp, device=dev)

        with torch.no_grad():
            b_out = _greedy_generate(
                baseline_model,
                inp,
                mask,
                max_new_tokens=cfg.language_max_new_tokens,
                expert_inputs=None,
            )
            # Fixed neutral tabular evidence for pure-language regression.
            tab = torch.zeros((1, cfg.language_tabular_dim), device=dev, dtype=torch.float32)
            i_out = _greedy_generate(
                injected_model,
                inp,
                mask,
                max_new_tokens=cfg.language_max_new_tokens,
                expert_inputs=({"tabular": tab},),
            )
        b_new = b_out[0, inp.shape[1] :].tolist()
        i_new = i_out[0, inp.shape[1] :].tolist()
        sim_scores.append(_token_jaccard(b_new, i_new))

    avg_sim = float(sum(sim_scores) / max(1, len(sim_scores)))
    degradation = float(1.0 - avg_sim)
    return {
        "language_similarity": avg_sim,
        "language_degradation": degradation,
        "language_pass": float(degradation < cfg.language_degradation_threshold),
    }


def _run_sweep_axis(
    backbone: BackboneSpec,
    cfg: BackboneScalingConfig,
    axis: str,
    values: List[int],
) -> List[Dict]:
    rows = []
    for v in values:
        k = cfg.base_k
        layer_idx = cfg.base_layer_idx
        evidence_dim = cfg.base_evidence_dim
        if axis == "k":
            k = int(v)
        elif axis == "layer_idx":
            layer_idx = int(v)
        elif axis == "evidence_dim":
            evidence_dim = int(v)
        else:
            raise ValueError(axis)

        run_cfg = AblationConfig(
            server_models_path=backbone.server_models_path,
            model_name=backbone.model_name,
            dataset=cfg.dataset,
            data_root=cfg.data_root,
            task=cfg.task,
            qa_type=cfg.qa_type,
            population_output_mode=cfg.population_output_mode,
            population_group_size=cfg.population_group_size,
            expert_type=cfg.expert_type,
            expert_model_id=cfg.expert_model_id,
            expert_model_path=cfg.expert_model_path,
            expert_output_dim=cfg.expert_output_dim,
            use_runtime_detector=cfg.use_runtime_detector,
            backbone=cfg.backbone,
            k=k,
            layer_idx=layer_idx,
            evidence_dim=evidence_dim,
            alpha=cfg.alpha,
            batch_size=cfg.batch_size,
            max_train_samples=cfg.max_train_samples,
            max_eval_samples=cfg.max_eval_samples,
            train_steps=cfg.train_steps,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            device=cfg.device,
            out_csv=os.path.join(cfg.out_dir, "_tmp.csv"),
            out_json=os.path.join(cfg.out_dir, "_tmp.json"),
        )
        r = run_single_experiment(run_cfg)
        metric_name = r.get("metric_name", "accuracy")
        row = {
            "suite": "backbone_scaling",
            "axis": axis,
            "axis_value": int(v),
            "backbone_name": backbone.name,
            "backbone_model": backbone.model_name,
            "expert_type": cfg.expert_type,
            "dataset": cfg.dataset,
            "task": cfg.task,
            "k": k,
            "layer_idx": layer_idx,
            "evidence_dim": evidence_dim,
            "metric_name": metric_name,
            "metric": float(r.get("metric", 0.0)),
        }
        row.update({k2: v2 for k2, v2 in r.items() if k2 not in row})
        rows.append(row)
        print(row)
    return rows


def _save_csv_json(rows: List[Dict], out_csv: str, out_json: str):
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


def _write_stat_table(rows: List[Dict], path: str):
    grouped: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("axis") == "language_regression":
            continue
        if "metric" not in r:
            continue
        grouped.setdefault(r["backbone_name"], []).append(float(r["metric"]))

    lines = ["| backbone | mean_metric | std_metric | n |", "|---|---:|---:|---:|"]
    for b, vals in grouped.items():
        m = statistics.mean(vals) if vals else 0.0
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        lines.append(f"| {b} | {m:.4f} | {s:.4f} | {len(vals)} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _save_curves(rows: List[Dict], out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    axes = ["k", "layer_idx", "evidence_dim"]
    for axis in axes:
        subset = [r for r in rows if r.get("axis") == axis]
        if not subset:
            continue
        bnames = sorted(set(r["backbone_name"] for r in subset))
        plt.figure(figsize=(7, 4))
        for b in bnames:
            vals = sorted([(int(r["axis_value"]), float(r["metric"])) for r in subset if r["backbone_name"] == b], key=lambda x: x[0])
            if not vals:
                continue
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            plt.plot(xs, ys, marker="o", label=b)
        plt.xlabel(axis)
        plt.ylabel("metric")
        plt.title(f"Stability curve: {axis}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"curve_{axis}.png"), dpi=150)
        plt.close()


def run_backbone_scaling_suite(cfg: BackboneScalingConfig) -> Dict[str, str]:
    set_seed(cfg.seed)
    _ensure_default_backbones(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)

    all_rows: List[Dict] = []
    for b in cfg.backbones:
        if not b.enabled:
            continue
        # Sweep each sensitivity axis around fixed operating point.
        all_rows.extend(_run_sweep_axis(b, cfg, "k", cfg.k_grid))
        all_rows.extend(_run_sweep_axis(b, cfg, "layer_idx", cfg.layer_grid))
        all_rows.extend(_run_sweep_axis(b, cfg, "evidence_dim", cfg.evidence_dim_grid))

        # Pure-language regression: baseline LLM vs injected LLM.
        lr = _language_regression(b, cfg)
        all_rows.append(
            {
                "suite": "backbone_scaling",
                "axis": "language_regression",
                "axis_value": -1,
                "backbone_name": b.name,
                "backbone_model": b.model_name,
                "expert_type": cfg.expert_type,
                "dataset": "language_internal",
                "task": "regression",
                "k": cfg.base_k,
                "layer_idx": cfg.base_layer_idx,
                "evidence_dim": cfg.base_evidence_dim,
                "metric_name": "language_degradation",
                "metric": float(lr["language_degradation"]),
                **lr,
            }
        )

    out_csv = os.path.join(cfg.out_dir, "scaling_results.csv")
    out_json = os.path.join(cfg.out_dir, "scaling_results.json")
    stat_md = os.path.join(cfg.out_dir, "statistical_comparison.md")
    _save_csv_json(all_rows, out_csv, out_json)
    _write_stat_table(all_rows, stat_md)
    _save_curves(all_rows, cfg.out_dir)
    return {
        "csv": out_csv,
        "json": out_json,
        "stat_table": stat_md,
        "curves_dir": cfg.out_dir,
    }


def load_backbone_scaling_config(path: str) -> BackboneScalingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bks = [BackboneSpec(**b) for b in raw.pop("backbones", [])]
    cfg = BackboneScalingConfig(**raw)
    if bks:
        cfg.backbones = bks
    return cfg
