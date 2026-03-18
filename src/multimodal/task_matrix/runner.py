from __future__ import annotations

import csv
import json
import os
import re
import traceback
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, build_expert_encoder
from src.multimodal.injection import (
    EvidenceProjector,
    FusionContext,
    SemanticEvidenceDomainExpert,
    build_fusion_policy,
)
from src.multimodal.server import BundleCompatibilitySpec, load_expert_bundle, save_expert_bundle
from src.multimodal.task_matrix.datasets import (
    FineGrainedPetDataset,
    GroundedGenerationDataset,
    PopulationBagDataset,
    StrictYesNoPetDataset,
    VQASubsetDataset,
    VQASubsetSpec,
    collate_task_batch,
)
from src.multimodal.task_matrix.metrics import (
    expected_calibration_error_from_bins,
    keyword_consistency_score,
    macro_f1_from_ints,
    mean_absolute_error,
)
from src.multimodal.tasks.parsing import parse_population_answer, population_bin_to_fraction_midpoint
from src.multimodal.utils.repro import set_seed


BASELINE_MODES = {"llm_only", "text_prompt_only", "overwrite", "router_parallel"}


@dataclass
class TaskSpec:
    name: str  # vqa|finegrained|population|grounded_generation
    train_steps: int = 100
    max_train_samples: int = 512
    max_eval_samples: int = 256
    batch_size: int = 8
    constrained_decoding: bool = True
    vqa_hf_dataset_name: str = "Graphcore/gqa-lxmert"
    vqa_split: str = "validation[:1024]"
    vqa_image_field: str = "image"
    vqa_question_field: str = "question"
    vqa_answer_field: str = "answer"
    population_group_size: int = 8
    gen_max_new_tokens: int = 24
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_repetition_penalty: float = 1.1
    gen_do_sample: bool = False


@dataclass
class TaskMatrixConfig:
    server_models_path: str = "/disk1/lfhu/hf_cache"
    model_name: str = "Qwen/Qwen3.5-9B"
    data_root: str = "./data"
    seed: int = 42
    device: str = "cuda"
    learning_rate: float = 2e-4
    evidence_dim: int = 256
    num_tokens: int = 4
    layer_idx: int = 0
    alpha: float = 1.0
    expert_type: str = "clip"
    expert_model_id: Optional[str] = None
    expert_model_path: Optional[str] = None
    expert_output_dim: int = 512
    use_runtime_detector: bool = False
    fusion_policy: str = "pre_attn_overwrite"  # pre_attn_overwrite|post_attn_router_parallel
    baseline_modes: List[str] = field(default_factory=lambda: ["overwrite"])
    save_bundle_dir: Optional[str] = None
    load_bundle_dir: Optional[str] = None
    label_schema: Optional[Dict[str, object]] = None
    normalization_stats: Optional[Dict[str, object]] = None
    tasks: List[TaskSpec] = field(default_factory=lambda: [TaskSpec(name="finegrained")])
    out_csv: str = "runs/task_matrix_results.csv"
    out_json: str = "runs/task_matrix_results.json"


def _device(device: str) -> torch.device:
    return torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")


def _normalize_baseline_modes(modes: List[str]) -> List[str]:
    aliases = {
        "injection": "overwrite",
        "injection_overwrite": "overwrite",
        "pre_attn_overwrite": "overwrite",
        "post_attn_router_parallel": "router_parallel",
    }
    out = []
    for raw in modes or ["overwrite"]:
        mode = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if mode == "expert_only":
            raise ValueError("expert_only baseline has been removed; use llm_only, text_prompt_only, overwrite, or router_parallel.")
        if mode not in BASELINE_MODES:
            raise ValueError(f"Unsupported baseline_mode: {raw}")
        if mode not in out:
            out.append(mode)
    return out


def _fusion_policy_for_mode(mode: str) -> str:
    if mode == "overwrite":
        return "pre_attn_overwrite"
    if mode == "router_parallel":
        return "post_attn_router_parallel"
    return "pre_attn_overwrite"


def _build_model_tokenizer(cfg: TaskMatrixConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.server_models_path,
        torch_dtype=torch.bfloat16,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        cfg.model_name,
        cache_dir=cfg.server_models_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        tokenizer=tokenizer,
    )
    model.num_tokens = cfg.num_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


@dataclass
class RuntimeConnector:
    semantic_expert: SemanticEvidenceDomainExpert
    fusion_policy: nn.Module
    inject_via_layer: bool


def _attach_connector(cfg: TaskMatrixConfig, model, baseline_mode: str) -> Optional[RuntimeConnector]:
    if baseline_mode in {"llm_only", "text_prompt_only"}:
        return None

    fusion_policy_name = _fusion_policy_for_mode(baseline_mode)
    if cfg.load_bundle_dir:
        loaded = load_expert_bundle(
            cfg.load_bundle_dir,
            compatibility=BundleCompatibilitySpec(
                hidden_size=int(model.config.hidden_size),
                model_name=cfg.model_name,
                layer_idx=cfg.layer_idx,
                fusion_policy=fusion_policy_name,
            ),
        )
        expert = loaded["semantic_expert"]
        fusion_policy = loaded["fusion_policy"]
        manifest = loaded.get("manifest")
        policy_name = manifest.fusion_policy if manifest is not None else getattr(fusion_policy, "policy_name", fusion_policy_name)
        inject_via_layer = False
        if policy_name == "pre_attn_overwrite":
            layer = model.model.layers[cfg.layer_idx]
            if hasattr(layer, "add_expert_"):
                layer.add_expert_(expert, map_to_expert_emb=None)
                inject_via_layer = True
            else:
                if not hasattr(model, "_external_experts"):
                    model._external_experts = nn.ModuleList()
                model._external_experts.append(expert)
        return RuntimeConnector(semantic_expert=expert, fusion_policy=fusion_policy, inject_via_layer=inject_via_layer)

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
        num_tokens=cfg.num_tokens,
        alpha=cfg.alpha,
    )
    expert = SemanticEvidenceDomainExpert(builder, projector)
    fusion_policy = build_fusion_policy(fusion_policy_name, hidden_size=model.config.hidden_size)
    inject_via_layer = False
    if fusion_policy_name == "pre_attn_overwrite":
        layer = model.model.layers[cfg.layer_idx]
        if hasattr(layer, "add_expert_"):
            layer.add_expert_(expert, map_to_expert_emb=None)
            inject_via_layer = True
        else:
            if not hasattr(model, "_external_experts"):
                model._external_experts = nn.ModuleList()
            model._external_experts.append(expert)
    return RuntimeConnector(semantic_expert=expert, fusion_policy=fusion_policy, inject_via_layer=inject_via_layer)


def _build_task_loaders(task: TaskSpec, tokenizer, data_root: str, seed: int):
    if task.name == "vqa":
        spec = VQASubsetSpec(
            hf_dataset_name=task.vqa_hf_dataset_name,
            split=task.vqa_split,
            image_field=task.vqa_image_field,
            question_field=task.vqa_question_field,
            answer_field=task.vqa_answer_field,
        )
        train_ds = VQASubsetDataset(tokenizer=tokenizer, spec=spec, max_samples=task.max_train_samples, seed=seed)
        eval_ds = VQASubsetDataset(tokenizer=tokenizer, spec=spec, max_samples=task.max_eval_samples, seed=seed + 1)
    elif task.name == "finegrained":
        train_ds = FineGrainedPetDataset(root=data_root, tokenizer=tokenizer, train=True, max_samples=task.max_train_samples)
        eval_ds = FineGrainedPetDataset(root=data_root, tokenizer=tokenizer, train=False, max_samples=task.max_eval_samples)
    elif task.name == "population":
        train_ds = PopulationBagDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=True,
            group_size=task.population_group_size,
            max_groups=task.max_train_samples,
            seed=seed,
        )
        eval_ds = PopulationBagDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=False,
            group_size=task.population_group_size,
            max_groups=task.max_eval_samples,
            seed=seed + 1,
        )
    elif task.name == "grounded_generation":
        train_ds = GroundedGenerationDataset(root=data_root, tokenizer=tokenizer, train=True, max_samples=task.max_train_samples)
        eval_ds = GroundedGenerationDataset(root=data_root, tokenizer=tokenizer, train=False, max_samples=task.max_eval_samples)
    elif task.name == "strict_yesno":
        train_ds = StrictYesNoPetDataset(
            root=data_root, tokenizer=tokenizer, train=True, max_samples=task.max_train_samples, seed=seed
        )
        eval_ds = StrictYesNoPetDataset(
            root=data_root, tokenizer=tokenizer, train=False, max_samples=task.max_eval_samples, seed=seed + 1
        )
    else:
        raise ValueError(f"Unsupported task name: {task.name}")

    train_loader = DataLoader(train_ds, batch_size=task.batch_size, shuffle=True, collate_fn=collate_task_batch)
    eval_loader = DataLoader(eval_ds, batch_size=task.batch_size, shuffle=False, collate_fn=collate_task_batch)
    return train_loader, eval_loader


def _constrained_argmax(logits: torch.Tensor, allowed_token_ids: List[int]) -> torch.Tensor:
    allowed = torch.tensor(sorted(set(allowed_token_ids)), dtype=torch.long, device=logits.device)
    local = logits.index_select(dim=-1, index=allowed).argmax(dim=-1)
    return allowed[local]


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _sample_next_token(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> torch.Tensor:
    step_logits = logits.clone()
    if repetition_penalty > 1.0 and generated_ids:
        for tid in set(generated_ids):
            step_logits[..., tid] = step_logits[..., tid] / repetition_penalty
    if temperature > 0:
        step_logits = step_logits / temperature
    step_logits = _top_p_filter(step_logits, top_p=top_p)
    if do_sample:
        probs = F.softmax(step_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
    else:
        nxt = step_logits.argmax(dim=-1, keepdim=True)
    return nxt.long()


def _expert_input_from_batch(batch: Dict, device: torch.device, expert_type: str):
    if expert_type in {"xgboost", "ft_transformer", "tabpfn", "tabular"}:
        imgs = batch["images"].to(device).float()
        if imgs.ndim == 5:
            imgs = imgs.mean(dim=1)
        feat = torch.cat([imgs.mean(dim=(2, 3)), imgs.std(dim=(2, 3))], dim=1)
        return {"tabular": feat}
    return {"images": batch["images"].to(device)}


def _unique_params(*param_groups):
    seen = set()
    out = []
    for grp in param_groups:
        for p in grp:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
    return out


def _apply_fusion_policy(
    *,
    model,
    runtime: RuntimeConnector,
    outputs,
    logits: torch.Tensor,
    expert_input,
    task_name: str,
    allowed_token_ids: Optional[List[int]],
):
    z = runtime.semantic_expert.evidence_builder(expert_input)
    z_tokens = runtime.semantic_expert.projector(z)
    z_ctx = z_tokens.mean(dim=1)
    ctx = FusionContext(task_name=task_name, allowed_token_ids=allowed_token_ids)
    return runtime.fusion_policy.forward_logits(model=model, outputs=outputs, logits=logits, z_ctx=z_ctx, ctx=ctx)


def _forward_logits_for_batch(
    model,
    runtime: Optional[RuntimeConnector],
    batch: Dict,
    cfg: TaskMatrixConfig,
    device: torch.device,
    baseline_mode: str,
) -> torch.Tensor:
    tokens = batch["prompt_tokens"].to(device)
    mask = batch["prompt_mask"].to(device)
    expert_input = None if baseline_mode in {"llm_only", "text_prompt_only"} else _expert_input_from_batch(batch, device, cfg.expert_type)
    model_out = model(
        tokens,
        attention_mask=mask,
        expert_inputs=(expert_input,) if runtime is not None and runtime.inject_via_layer else None,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    logits = model_out.logits[:, -1, :]
    if baseline_mode == "router_parallel" and runtime is not None:
        logits = _apply_fusion_policy(
            model=model,
            runtime=runtime,
            outputs=model_out,
            logits=logits,
            expert_input=expert_input,
            task_name=batch.get("task_name", ""),
            allowed_token_ids=batch.get("allowed_token_ids"),
        )
    return logits


def _loss_on_allowed_ids(logits: torch.Tensor, targets: torch.Tensor, allowed_token_ids: Optional[List[int]]):
    if not allowed_token_ids:
        return nn.CrossEntropyLoss()(logits, targets)
    allowed = torch.tensor(sorted(set(int(x) for x in allowed_token_ids)), device=logits.device, dtype=torch.long)
    subset = logits.index_select(dim=-1, index=allowed)
    pos = torch.searchsorted(allowed, targets)
    in_range = pos < allowed.numel()
    safe = torch.where(in_range, pos, torch.zeros_like(pos))
    ok = in_range & (allowed[safe] == targets)
    if not bool(ok.all()):
        return nn.CrossEntropyLoss()(logits, targets)
    return nn.CrossEntropyLoss()(subset, pos)


def _train_connector(
    model,
    runtime: Optional[RuntimeConnector],
    train_loader,
    task: TaskSpec,
    cfg: TaskMatrixConfig,
    device: torch.device,
    baseline_mode: str,
):
    if runtime is None:
        return
    model.train()
    trainable = _unique_params(
        model.parameters(),
        runtime.semantic_expert.parameters(),
        runtime.fusion_policy.parameters(),
    )
    if len(trainable) == 0:
        return
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    step = 0
    while step < task.train_steps:
        for batch in train_loader:
            targets = batch["target_token_id"].to(device)
            logits = _forward_logits_for_batch(model, runtime, batch, cfg, device, baseline_mode)
            loss = _loss_on_allowed_ids(logits, targets, batch.get("allowed_token_ids"))
            if not loss.requires_grad:
                step += 1
                if step >= task.train_steps:
                    break
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= task.train_steps:
                break


def _generate_rationale(
    model,
    runtime: RuntimeConnector,
    tokenizer,
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    expert_input,
    task_name: str,
    allowed_token_ids: Optional[List[int]],
    cfg: TaskMatrixConfig,
    max_new_tokens: int = 24,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
):
    seq = prompt_tokens.clone()
    mask = prompt_mask.clone()
    generated_ids: List[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        out = model(
            seq,
            attention_mask=mask,
            expert_inputs=(expert_input,) if runtime.inject_via_layer else None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        step_logits = out.logits[:, -1, :]
        if getattr(runtime.fusion_policy, "policy_name", "") == "post_attn_router_parallel":
            step_logits = _apply_fusion_policy(
                model=model,
                runtime=runtime,
                outputs=out,
                logits=step_logits,
                expert_input=expert_input,
                task_name=task_name,
                allowed_token_ids=allowed_token_ids,
            )
        nxt = _sample_next_token(
            step_logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        generated_ids.append(int(nxt.item()))
        seq = torch.cat([seq, nxt], dim=1)
        mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break
    text = tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)
    return text


def _eval_task(
    model,
    runtime: Optional[RuntimeConnector],
    tokenizer,
    eval_loader,
    task: TaskSpec,
    cfg: TaskMatrixConfig,
    device: torch.device,
    baseline_mode: str,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    pop_true = []
    pop_pred = []
    rationale_scores = []
    fmt_ok = 0
    rationale_has_content = 0

    with torch.no_grad():
        for batch in eval_loader:
            tokens = batch["prompt_tokens"].to(device)
            mask = batch["prompt_mask"].to(device)
            logits = _forward_logits_for_batch(model, runtime, batch, cfg, device, baseline_mode)
            if task.constrained_decoding and batch.get("allowed_token_ids"):
                pred_ids = _constrained_argmax(logits, batch["allowed_token_ids"]).cpu()
            else:
                pred_ids = logits.argmax(dim=-1).cpu()

            targets = batch["target_token_id"].cpu()
            correct += (pred_ids == targets).sum().item()
            total += targets.numel()

            if task.name in {"vqa", "finegrained", "grounded_generation", "strict_yesno"}:
                if "answer_idx" in batch:
                    y_true.extend(batch["answer_idx"].cpu().tolist())
                    allowed = batch["allowed_token_ids"]
                    lookup = {tid: i for i, tid in enumerate(allowed)}
                    y_pred.extend([lookup.get(int(x), 0) for x in pred_ids.tolist()])
                elif "label_idx" in batch:
                    y_true.extend(batch["label_idx"].cpu().tolist())
                    allowed = batch["allowed_token_ids"]
                    lookup = {tid: i for i, tid in enumerate(allowed)}
                    y_pred.extend([lookup.get(int(x), 0) for x in pred_ids.tolist()])

            if task.name == "population":
                pred_texts = [tokenizer.decode([int(x)]).strip() for x in pred_ids.tolist()]
                bins = [parse_population_answer(t, mode="integer") for t in pred_texts]
                pred_f = [population_bin_to_fraction_midpoint(b if b is not None else 0, mode="integer") for b in bins]
                true_f = batch["fraction"].cpu().tolist()
                pop_pred.extend(pred_f)
                pop_true.extend(true_f)

            if task.name == "grounded_generation":
                for i in range(tokens.size(0)):
                    if runtime is None:
                        continue
                    exp_in = _expert_input_from_batch(
                        {"images": batch["images"][i : i + 1]}, device, cfg.expert_type
                    )
                    gen = _generate_rationale(
                        model,
                        runtime,
                        tokenizer,
                        tokens[i : i + 1],
                        mask[i : i + 1],
                        exp_in,
                        task_name="grounded_generation",
                        allowed_token_ids=batch.get("allowed_token_ids"),
                        cfg=cfg,
                        max_new_tokens=int(task.gen_max_new_tokens),
                        temperature=float(task.gen_temperature),
                        top_p=float(task.gen_top_p),
                        repetition_penalty=float(task.gen_repetition_penalty),
                        do_sample=bool(task.gen_do_sample),
                    )
                    kws = batch.get("rationale_keywords", [[]])[i]
                    m_ans = re.search(r"(?im)\banswer\s*:\s*([^\n\.]+)", gen)
                    m_rat = re.search(r"(?im)\brationale\s*:\s*([^\n]+)", gen)
                    if m_ans is not None and m_rat is not None:
                        fmt_ok += 1
                    rationale_text = m_rat.group(1).strip() if m_rat is not None else gen
                    if len(rationale_text) >= 8:
                        rationale_has_content += 1
                    rationale_scores.append(keyword_consistency_score(rationale_text, kws))

    acc = correct / max(1, total)
    result: Dict[str, float] = {"accuracy": float(acc)}

    if task.name in {"vqa", "finegrained", "grounded_generation", "strict_yesno"} and y_true:
        n_cls = max(1, len(set(y_true)))
        result["f1"] = macro_f1_from_ints(y_true, y_pred, n_cls)
    if task.name == "population":
        result["mae"] = mean_absolute_error(pop_true, pop_pred)
        result["calibration_error"] = expected_calibration_error_from_bins(
            torch.tensor(pop_true),
            torch.tensor(pop_pred),
            n_bins=10,
        )
    if task.name == "grounded_generation":
        result["rationale_consistency"] = float(sum(rationale_scores) / max(1, len(rationale_scores)))
        result["format_compliance"] = float(fmt_ok / max(1, len(rationale_scores)))
        result["rationale_nonempty_rate"] = float(rationale_has_content / max(1, len(rationale_scores)))
    return result


def _save_rows(rows: List[Dict], out_csv: str, out_json: str):
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


def run_task_matrix(cfg: TaskMatrixConfig) -> List[Dict]:
    set_seed(cfg.seed)
    device = _device(cfg.device)
    rows: List[Dict] = []
    baseline_modes = _normalize_baseline_modes(cfg.baseline_modes)

    for i, task in enumerate(cfg.tasks):
        for baseline_mode in baseline_modes:
            try:
                model, tokenizer = _build_model_tokenizer(cfg)
                runtime = _attach_connector(cfg, model, baseline_mode)
                model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)
                if runtime is not None:
                    runtime_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                    runtime.semantic_expert = runtime.semantic_expert.to(device=device, dtype=runtime_dtype)
                    runtime.fusion_policy = runtime.fusion_policy.to(device=device, dtype=runtime_dtype)
                train_loader, eval_loader = _build_task_loaders(task, tokenizer, cfg.data_root, cfg.seed + i)
                _train_connector(model, runtime, train_loader, task, cfg, device, baseline_mode)
                metrics = _eval_task(model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode)

                if cfg.save_bundle_dir and runtime is not None:
                    bundle_dir = os.path.join(cfg.save_bundle_dir, f"{task.name}_{baseline_mode}")
                    enc_spec = {
                        "expert_type": cfg.expert_type,
                        "model_id": cfg.expert_model_id,
                        "model_path": cfg.expert_model_path,
                        "output_dim": cfg.expert_output_dim,
                        "cache_dir": cfg.server_models_path,
                        "use_runtime_detector": cfg.use_runtime_detector,
                    }
                    save_expert_bundle(
                        bundle_dir,
                        encoder_spec=enc_spec,
                        semantic_expert=runtime.semantic_expert,
                        fusion_policy=runtime.fusion_policy,
                        normalization_stats=cfg.normalization_stats,
                        label_schema=cfg.label_schema,
                        metadata={
                            "task": task.name,
                            "model_name": cfg.model_name,
                            "fusion_policy": _fusion_policy_for_mode(baseline_mode),
                            "layer_idx": cfg.layer_idx,
                            "feature_format": "tabular" if cfg.expert_type in {"xgboost", "ft_transformer", "tabpfn", "tabular"} else "image_tensor",
                        },
                    )

                row = {
                    "task": task.name,
                    "baseline_mode": baseline_mode,
                    "expert_type": cfg.expert_type,
                    "backbone": cfg.model_name,
                    "fusion_policy": _fusion_policy_for_mode(baseline_mode),
                    "k": cfg.num_tokens if runtime is not None else 0,
                    "layer_idx": cfg.layer_idx,
                    "evidence_dim": cfg.evidence_dim if runtime is not None else 0,
                    "metric": metrics.get("accuracy", metrics.get("mae", 0.0)),
                    "status": "ok",
                }
                if baseline_mode == "text_prompt_only":
                    row["baseline_note"] = "Prompt-only baseline over the standardized task text; no uploaded expert or injection path."
                row.update(metrics)
            except Exception as exc:
                row = {
                    "task": task.name,
                    "baseline_mode": baseline_mode,
                    "expert_type": cfg.expert_type,
                    "backbone": cfg.model_name,
                    "fusion_policy": _fusion_policy_for_mode(baseline_mode),
                    "k": cfg.num_tokens if baseline_mode in {"overwrite", "router_parallel"} else 0,
                    "layer_idx": cfg.layer_idx,
                    "evidence_dim": cfg.evidence_dim if baseline_mode in {"overwrite", "router_parallel"} else 0,
                    "metric": 0.0,
                    "status": "error",
                    "error": str(exc) or repr(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            rows.append(row)
            print(row)

    _save_rows(rows, cfg.out_csv, cfg.out_json)
    return rows


def load_task_matrix_config(path: str) -> TaskMatrixConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks = [TaskSpec(**t) for t in raw.pop("tasks", [])]
    cfg = TaskMatrixConfig(**raw)
    cfg.tasks = tasks if tasks else cfg.tasks
    return cfg
