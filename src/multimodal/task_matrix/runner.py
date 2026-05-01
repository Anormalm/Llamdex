from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, build_expert_encoder
from src.multimodal.injection import (
    EvidencePreRouter,
    EvidenceProjector,
    FusionContext,
    SemanticEvidenceDomainExpert,
    build_fusion_policy,
    task_name_to_id,
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


BASELINE_MODES = {"llm_only", "text_prompt_only", "overwrite", "router_parallel", "expert_direct"}


@dataclass
class TaskSpec:
    name: str  # vqa|finegrained|population|grounded_generation
    image_dataset_name: str = "oxford_pet"
    train_steps: int = 100
    max_train_samples: int = 512
    max_eval_samples: int = 256
    batch_size: int = 8
    constrained_decoding: bool = True
    vqa_hf_dataset_name: str = "Graphcore/gqa-lxmert"
    vqa_eval_hf_dataset_name: str = ""
    vqa_split: str = "validation[:1024]"
    vqa_image_field: str = "image"
    vqa_second_image_field: str = ""
    vqa_question_field: str = "question"
    vqa_answer_field: str = "answer"
    vqa_context_field: str = ""
    population_group_size: int = 8
    gen_max_new_tokens: int = 24
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_repetition_penalty: float = 1.1
    gen_do_sample: bool = False
    min_rationale_chars: int = 8
    min_rationale_keyword_score: float = 0.5
    strict_answer_code: bool = True
    prompt_template_style: str = "legacy"  # legacy|compact


@dataclass
class TaskMatrixConfig:
    server_models_path: str = "/disk1/lfhu/hf_cache"
    model_name: str = "Qwen/Qwen3.5-9B"
    data_root: str = "/disk1/lfhu/data"
    seed: int = 42
    device: str = "cuda"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    lr_schedule: str = "cosine"  # constant|cosine
    lr_warmup_steps: int = 0
    evidence_dim: int = 256
    num_tokens: int = 4
    layer_idx: int = 0
    router_layer_indices: Optional[List[int]] = None
    alpha: float = 1.0
    expert_type: str = "siglip"
    expert_model_id: Optional[str] = None
    expert_model_path: Optional[str] = None
    expert_output_dim: int = 512
    use_runtime_detector: bool = False
    local_files_only: bool = False
    fusion_policy: str = "pre_ffn_router_parallel"  # pre_attn_overwrite|post_attn_router_parallel|post_attn_router_layers|pre_ffn_router_parallel
    baseline_modes: List[str] = field(default_factory=lambda: ["router_parallel"])
    save_bundle_dir: Optional[str] = None
    load_bundle_dir: Optional[str] = None
    upload_scope: str = "expert_only"
    # Privacy contract: when loading client-uploaded bundles on server,
    # keep expert weights frozen and train only connector/injection modules.
    freeze_loaded_semantic_expert: bool = True
    freeze_loaded_fusion_policy: bool = False
    label_schema: Optional[Dict[str, object]] = None
    normalization_stats: Optional[Dict[str, object]] = None
    repeat_seeds: List[int] = field(default_factory=list)
    aggregate_seed_metrics: bool = True
    use_pre_router: bool = True
    pre_router_mode: str = "global_feature"
    pre_router_hidden_dim: int = 128
    task_conditioning: bool = True
    pre_router_task_embedding_dim: int = 16
    pre_router_max_task_ids: int = 32
    joint_score_latency_weight: float = 0.03
    joint_score_format_weight: float = 0.25
    joint_score_instability_weight: float = 0.5
    tasks: List[TaskSpec] = field(default_factory=lambda: [TaskSpec(name="finegrained")])
    out_csv: str = "/disk1/lfhu/runs/task_matrix_results.csv"
    out_json: str = "/disk1/lfhu/runs/task_matrix_results.json"


def _device(device: str) -> torch.device:
    return torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")


def _should_use_local_files_only(cfg: TaskMatrixConfig) -> bool:
    if bool(getattr(cfg, "local_files_only", False)):
        return True
    return any(str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes"} for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _set_module_trainable(module: Optional[nn.Module], trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(trainable)
    module.train(bool(trainable))


@contextmanager
def _suppress_hf_auto_conversion_noise(enabled: bool):
    if not enabled:
        yield
        return

    def _quiet_auto_conversion(*args, **kwargs):
        kwargs["ignore_errors_during_conversion"] = True
        return None, None, None

    import transformers.modeling_utils as modeling_utils
    import transformers.safetensors_conversion as safetensors_conversion

    with patch.object(modeling_utils, "auto_conversion", _quiet_auto_conversion), patch.object(
        safetensors_conversion, "auto_conversion", _quiet_auto_conversion
    ):
        yield


def _build_lr_scheduler(opt: torch.optim.Optimizer, cfg: TaskMatrixConfig, total_steps: int):
    schedule = str(getattr(cfg, "lr_schedule", "cosine")).strip().lower()
    warmup_steps = max(0, int(getattr(cfg, "lr_warmup_steps", 0)))
    total_steps = max(1, int(total_steps))
    if schedule in {"", "constant", "none"} and warmup_steps <= 0:
        return None

    def _lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if schedule in {"", "constant", "none"}:
            return 1.0
        if schedule == "cosine":
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)


def _normalize_baseline_modes(modes: List[str]) -> List[str]:
    aliases = {
        "injection": "overwrite",
        "injection_overwrite": "overwrite",
        "pre_attn_overwrite": "overwrite",
        "post_attn_router_parallel": "router_parallel",
        "post_attn_router_layers": "router_parallel",
        "post_attn_router_all_layers": "router_parallel",
        "classifier_direct": "expert_direct",
        "vision_direct": "expert_direct",
    }
    out = []
    for raw in modes or ["router_parallel"]:
        mode = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if mode == "expert_only":
            mode = "expert_direct"
        if mode not in BASELINE_MODES:
            raise ValueError(f"Unsupported baseline_mode: {raw}")
        if mode not in out:
            out.append(mode)
    return out


def _fusion_policy_for_mode(cfg: TaskMatrixConfig, mode: str) -> str:
    if mode == "expert_direct":
        return "expert_direct"
    if mode == "overwrite":
        return "pre_attn_overwrite"
    if mode == "router_parallel":
        preferred = str(getattr(cfg, "fusion_policy", "post_attn_router_parallel")).strip().lower()
        if preferred == "hybrid_task_routed":
            return "pre_ffn_router_parallel"
        if preferred in {"post_attn_router_parallel", "post_attn_router_layers", "post_attn_router_all_layers", "pre_ffn_router_parallel"}:
            return preferred
        return "post_attn_router_parallel"
    return "post_attn_router_parallel"


def _resolve_cfg_for_task(cfg: TaskMatrixConfig, task: TaskSpec, baseline_mode: str) -> TaskMatrixConfig:
    if baseline_mode != "router_parallel":
        return cfg
    preferred = str(getattr(cfg, "fusion_policy", "post_attn_router_parallel")).strip().lower()
    if preferred != "hybrid_task_routed":
        return cfg
    task_name = str(getattr(task, "name", "")).strip().lower()
    routed = "post_attn_router_layers" if task_name == "grounded_generation" else "pre_ffn_router_parallel"
    return replace(cfg, fusion_policy=routed)


def _build_model_tokenizer(cfg: TaskMatrixConfig):
    local_files_only = _should_use_local_files_only(cfg)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    with _suppress_hf_auto_conversion_noise(local_files_only):
        model_config = AutoConfig.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.server_models_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.server_models_path,
            torch_dtype=torch.bfloat16,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        model = DomainQwenForCausalLM.from_pretrained_qwen(
            cfg.model_name,
            config=model_config,
            generation_config=GenerationConfig.from_model_config(model_config),
            cache_dir=cfg.server_models_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            tokenizer=tokenizer,
            local_files_only=local_files_only,
        )
    model.num_tokens = cfg.num_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def _build_tokenizer(cfg: TaskMatrixConfig):
    local_files_only = _should_use_local_files_only(cfg)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    with _suppress_hf_auto_conversion_noise(local_files_only):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.server_models_path,
            torch_dtype=torch.bfloat16,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer


def _attach_direct_expert(cfg: TaskMatrixConfig) -> RuntimeConnector:
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
    return RuntimeConnector(semantic_expert=enc, fusion_policy=nn.Identity(), inject_via_layer=False)


@dataclass
class RuntimeConnector:
    semantic_expert: SemanticEvidenceDomainExpert
    fusion_policy: nn.Module
    inject_via_layer: bool


def _router_layer_indices(cfg: TaskMatrixConfig, policy_name: str):
    indices = getattr(cfg, "router_layer_indices", None)
    if indices:
        return [int(i) for i in indices]
    if policy_name in {"post_attn_router_layers", "post_attn_router_all_layers"}:
        return None
    return int(cfg.layer_idx)


def _attach_connector(cfg: TaskMatrixConfig, model, baseline_mode: str) -> Optional[RuntimeConnector]:
    if baseline_mode in {"llm_only", "text_prompt_only"}:
        return None

    fusion_policy_name = _fusion_policy_for_mode(cfg, baseline_mode)
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
        if bool(getattr(cfg, "use_pre_router", False)) and getattr(expert, "pre_router", None) is None:
            expert.pre_router = EvidencePreRouter(
                evidence_dim=int(cfg.evidence_dim),
                mode=str(getattr(cfg, "pre_router_mode", "global_feature")),
                hidden_dim=int(getattr(cfg, "pre_router_hidden_dim", 128)),
                task_conditioning=bool(getattr(cfg, "task_conditioning", False)),
                task_embedding_dim=int(getattr(cfg, "pre_router_task_embedding_dim", 16)),
                max_task_ids=int(getattr(cfg, "pre_router_max_task_ids", 32)),
            )
        fusion_policy = loaded["fusion_policy"]
        manifest = loaded.get("manifest")
        policy_name = manifest.fusion_policy if manifest is not None else getattr(fusion_policy, "policy_name", fusion_policy_name)
        inject_via_layer = False
        if policy_name == "pre_attn_overwrite":
            if hasattr(model, "add_expert_"):
                model.add_expert_(expert, layer_id=cfg.layer_idx, map_to_expert_emb=None)
                inject_via_layer = True
            else:
                layer = model.model.layers[cfg.layer_idx]
                if hasattr(layer, "add_expert_"):
                    layer.add_expert_(expert, map_to_expert_emb=None)
                    inject_via_layer = True
                else:
                    if not hasattr(model, "_external_experts"):
                        model._external_experts = nn.ModuleList()
                    model._external_experts.append(expert)
        elif policy_name in {"pre_ffn_router_parallel", "post_attn_router_layers", "post_attn_router_all_layers"} and hasattr(
            fusion_policy, "register_to_model"
        ):
            fusion_policy.register_to_model(model, _router_layer_indices(cfg, policy_name))

        if bool(getattr(cfg, "freeze_loaded_semantic_expert", True)):
            _set_module_trainable(expert, False)
        if bool(getattr(cfg, "freeze_loaded_fusion_policy", False)):
            _set_module_trainable(fusion_policy, False)

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
    pre_router = None
    if bool(getattr(cfg, "use_pre_router", False)):
        pre_router = EvidencePreRouter(
            evidence_dim=int(cfg.evidence_dim),
            mode=str(getattr(cfg, "pre_router_mode", "global_feature")),
            hidden_dim=int(getattr(cfg, "pre_router_hidden_dim", 128)),
            task_conditioning=bool(getattr(cfg, "task_conditioning", False)),
            task_embedding_dim=int(getattr(cfg, "pre_router_task_embedding_dim", 16)),
            max_task_ids=int(getattr(cfg, "pre_router_max_task_ids", 32)),
        )
    expert = SemanticEvidenceDomainExpert(builder, projector, pre_router=pre_router)
    fusion_policy = build_fusion_policy(fusion_policy_name, hidden_size=model.config.hidden_size)
    inject_via_layer = False
    if fusion_policy_name == "pre_attn_overwrite":
        if hasattr(model, "add_expert_"):
            model.add_expert_(expert, layer_id=cfg.layer_idx, map_to_expert_emb=None)
            inject_via_layer = True
        else:
            layer = model.model.layers[cfg.layer_idx]
            if hasattr(layer, "add_expert_"):
                layer.add_expert_(expert, map_to_expert_emb=None)
                inject_via_layer = True
            else:
                if not hasattr(model, "_external_experts"):
                    model._external_experts = nn.ModuleList()
                model._external_experts.append(expert)
    elif fusion_policy_name in {"pre_ffn_router_parallel", "post_attn_router_layers", "post_attn_router_all_layers"} and hasattr(
        fusion_policy, "register_to_model"
    ):
        fusion_policy.register_to_model(model, _router_layer_indices(cfg, fusion_policy_name))
    return RuntimeConnector(semantic_expert=expert, fusion_policy=fusion_policy, inject_via_layer=inject_via_layer)


def _build_task_loaders(task: TaskSpec, tokenizer, data_root: str, seed: int):
    if task.name == "vqa":
        spec = VQASubsetSpec(
            hf_dataset_name=task.vqa_hf_dataset_name,
            split=task.vqa_split,
            image_field=task.vqa_image_field,
            second_image_field=task.vqa_second_image_field,
            question_field=task.vqa_question_field,
            answer_field=task.vqa_answer_field,
            context_field=task.vqa_context_field,
        )
        eval_spec = replace(spec, hf_dataset_name=task.vqa_eval_hf_dataset_name or task.vqa_hf_dataset_name)
        train_ds = VQASubsetDataset(tokenizer=tokenizer, spec=spec, max_samples=task.max_train_samples, seed=seed)
        eval_ds = VQASubsetDataset(tokenizer=tokenizer, spec=eval_spec, max_samples=task.max_eval_samples, seed=seed + 1)
    elif task.name == "finegrained":
        train_ds = FineGrainedPetDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=True,
            max_samples=task.max_train_samples,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
        )
        eval_ds = FineGrainedPetDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=False,
            max_samples=task.max_eval_samples,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
        )
    elif task.name == "population":
        train_ds = PopulationBagDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=True,
            dataset_name=task.image_dataset_name,
            group_size=task.population_group_size,
            max_groups=task.max_train_samples,
            seed=seed,
            prompt_template_style=task.prompt_template_style,
        )
        eval_ds = PopulationBagDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=task.image_dataset_name,
            group_size=task.population_group_size,
            max_groups=task.max_eval_samples,
            seed=seed + 1,
            prompt_template_style=task.prompt_template_style,
        )
    elif task.name == "grounded_generation":
        train_ds = GroundedGenerationDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=True,
            max_samples=task.max_train_samples,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
        )
        eval_ds = GroundedGenerationDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=False,
            max_samples=task.max_eval_samples,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
        )
    elif task.name == "strict_yesno":
        train_ds = StrictYesNoPetDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=True,
            max_samples=task.max_train_samples,
            seed=seed,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
        )
        eval_ds = StrictYesNoPetDataset(
            root=data_root,
            tokenizer=tokenizer,
            train=False,
            max_samples=task.max_eval_samples,
            seed=seed + 1,
            dataset_name=task.image_dataset_name,
            prompt_template_style=task.prompt_template_style,
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
        if "tabular" in batch:
            return {"tabular": batch["tabular"].to(device).float()}
        imgs = batch["images"].to(device).float()
        if imgs.ndim == 5:
            imgs = imgs.mean(dim=1)
        feat = torch.cat([imgs.mean(dim=(2, 3)), imgs.std(dim=(2, 3))], dim=1)
        return {"tabular": feat}
    return {"images": batch["images"].to(device)}


def _task_ids_for_batch(cfg: TaskMatrixConfig, batch: Dict, device: torch.device) -> Optional[torch.Tensor]:
    if not bool(getattr(cfg, "task_conditioning", False)):
        return None
    bsz = int(batch["prompt_tokens"].size(0))
    raw_task_name = batch.get("task_name", "")
    if isinstance(raw_task_name, (list, tuple)):
        task_name = str(raw_task_name[0] if len(raw_task_name) > 0 else "").strip().lower()
    else:
        task_name = str(raw_task_name).strip().lower()
    task_name = task_name or "single_label"
    task_id = task_name_to_id(task_name, max_task_ids=int(getattr(cfg, "pre_router_max_task_ids", 32)))
    return torch.full((bsz,), int(task_id), dtype=torch.long, device=device)


def _set_expert_task_context(runtime: Optional[RuntimeConnector], task_ids: Optional[torch.Tensor]) -> None:
    if runtime is None:
        return
    expert = runtime.semantic_expert
    if getattr(expert, "pre_router", None) is None:
        return
    if task_ids is None:
        expert.clear_task_context()
    else:
        expert.set_task_context(task_id=task_ids)


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
    task_ids: Optional[torch.Tensor] = None,
):
    policy_name = getattr(runtime.fusion_policy, "policy_name", "")
    z_ctx = None
    if policy_name != "pre_ffn_router_parallel":
        z = runtime.semantic_expert.evidence_builder(expert_input)
        z_tokens = runtime.semantic_expert.project_from_evidence(z, task_id=task_ids)
        z_ctx = z_tokens.mean(dim=1)
    ctx = FusionContext(task_name=task_name, allowed_token_ids=allowed_token_ids)
    return runtime.fusion_policy.forward_logits(model=model, outputs=outputs, logits=logits, z_ctx=z_ctx, ctx=ctx)


def _set_router_context_if_needed(runtime: Optional[RuntimeConnector], expert_input, task_ids: Optional[torch.Tensor] = None) -> bool:
    if runtime is None or expert_input is None or not hasattr(runtime.fusion_policy, "set_context"):
        return False
    policy_name = getattr(runtime.fusion_policy, "policy_name", "")
    if policy_name not in {"pre_ffn_router_parallel", "post_attn_router_layers", "post_attn_router_all_layers"}:
        return False
    z = runtime.semantic_expert.evidence_builder(expert_input)
    z_tokens = runtime.semantic_expert.project_from_evidence(z, task_id=task_ids)
    runtime.fusion_policy.set_context(z_tokens.mean(dim=1))
    return True


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
    policy_name = getattr(runtime.fusion_policy, "policy_name", "") if runtime is not None else ""
    task_ids = _task_ids_for_batch(cfg, batch, device) if runtime is not None else None
    _set_expert_task_context(runtime, task_ids)
    context_was_set = _set_router_context_if_needed(runtime, expert_input, task_ids=task_ids)
    try:
        model_out = model(
            tokens,
            attention_mask=mask,
            expert_inputs=(expert_input,) if runtime is not None and runtime.inject_via_layer else None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    finally:
        if context_was_set and runtime is not None and hasattr(runtime.fusion_policy, "clear_context"):
            runtime.fusion_policy.clear_context()
        _set_expert_task_context(runtime, None)
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
            task_ids=task_ids,
        )
    return logits


def _expert_direct_pred_ids(
    runtime: RuntimeConnector,
    batch: Dict,
    cfg: TaskMatrixConfig,
    device: torch.device,
) -> torch.Tensor:
    expert = runtime.semantic_expert
    task_name = str(batch.get("task_name", ""))
    allowed = list(batch.get("allowed_token_ids") or [])
    if not allowed:
        raise ValueError("expert_direct requires allowed_token_ids in the batch.")

    if task_name == "population":
        images = batch["images"].to(device)
        target_cls = batch["target_class"].to(device)
        if images.ndim != 5:
            raise ValueError("expert_direct population expects image bags with shape [B, N, C, H, W].")
        bsz, group = images.shape[:2]
        logits = expert.encode(images.reshape(bsz * group, *images.shape[2:]))
        pred_cls = logits.argmax(dim=-1).reshape(bsz, group)
        frac = (pred_cls == target_cls.unsqueeze(1)).float().mean(dim=1)
        bins = torch.round(frac * 10.0).long().clamp_(0, 10).cpu().tolist()
        return torch.tensor([int(allowed[int(b)]) for b in bins], dtype=torch.long)

    logits = expert.encode(_expert_input_from_batch(batch, device, cfg.expert_type))
    pred_cls = logits.argmax(dim=-1).detach().cpu()

    if task_name == "strict_yesno":
        target_cls = batch["target_class"].cpu()
        yes_id, no_id = int(allowed[0]), int(allowed[1])
        return torch.tensor([yes_id if int(p) == int(t) else no_id for p, t in zip(pred_cls.tolist(), target_cls.tolist())], dtype=torch.long)

    return torch.tensor([int(allowed[min(max(int(p), 0), len(allowed) - 1)]) for p in pred_cls.tolist()], dtype=torch.long)


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
    if runtime is None or baseline_mode == "expert_direct":
        return
    model.train()
    runtime.semantic_expert.train(any(p.requires_grad for p in runtime.semantic_expert.parameters()))
    runtime.fusion_policy.train(any(p.requires_grad for p in runtime.fusion_policy.parameters()))
    trainable = _unique_params(
        model.parameters(),
        runtime.semantic_expert.parameters(),
        runtime.fusion_policy.parameters(),
    )
    if len(trainable) == 0:
        return
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    scheduler = _build_lr_scheduler(opt, cfg, total_steps=task.train_steps)
    grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0))
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
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
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
    prefix_text: str = "",
):
    seq = prompt_tokens.clone()
    mask = prompt_mask.clone()
    generated_ids: List[int] = []
    prefix_text = str(prefix_text or "")
    if prefix_text:
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        if prefix_ids:
            prefix = torch.tensor(prefix_ids, dtype=seq.dtype, device=seq.device).unsqueeze(0)
            seq = torch.cat([seq, prefix], dim=1)
            mask = torch.cat([mask, torch.ones_like(prefix)], dim=1)
    eos_id = tokenizer.eos_token_id
    policy_name = getattr(runtime.fusion_policy, "policy_name", "")
    task_ids = None
    if bool(getattr(cfg, "task_conditioning", False)):
        task_id = task_name_to_id(task_name, max_task_ids=int(getattr(cfg, "pre_router_max_task_ids", 32)))
        task_ids = torch.full((seq.size(0),), int(task_id), dtype=torch.long, device=seq.device)
    _set_expert_task_context(runtime, task_ids)
    context_was_set = _set_router_context_if_needed(runtime, expert_input, task_ids=task_ids)
    try:
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
            if policy_name in {"post_attn_router_parallel", "post_attn_router_layers", "post_attn_router_all_layers"}:
                step_logits = _apply_fusion_policy(
                    model=model,
                    runtime=runtime,
                    outputs=out,
                    logits=step_logits,
                    expert_input=expert_input,
                    task_name=task_name,
                    allowed_token_ids=allowed_token_ids,
                    task_ids=task_ids,
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
    finally:
        if context_was_set and hasattr(runtime.fusion_policy, "clear_context"):
            runtime.fusion_policy.clear_context()
        _set_expert_task_context(runtime, None)
    # Decode generated continuation only; decoding the full sequence includes the
    # prompt template (which itself contains "Answer:/Rationale:" placeholders).
    try:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception:
        text = tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)
    return f"{prefix_text}{text}".strip()


def _extract_answer_rationale(text: str):
    raw = str(text)
    answer_text = ""
    rationale_text = ""
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                answer_text = str(payload.get("answer", "")).strip()
                rationale_text = str(payload.get("rationale", "")).strip()
        except Exception:
            pass
    if not answer_text:
        ans_matches = re.findall(r"(?im)\banswer\s*:\s*([^\n\.]+)", raw)
        answer_text = ans_matches[-1].strip() if ans_matches else ""
    if not rationale_text:
        rat_matches = re.findall(r"(?ims)\brationale\s*:\s*(.+?)(?=\banswer\s*:|$)", raw)
        rationale_text = rat_matches[-1].strip() if rat_matches else ""
    return answer_text, rationale_text


def _canonicalize_grounded_output(raw_text: str, predicted_code: str) -> str:
    answer_text, rationale_text = _extract_answer_rationale(raw_text)
    final_answer = _normalize_code_text(answer_text) or _normalize_code_text(predicted_code)
    final_rationale = rationale_text.strip()
    if not final_rationale:
        final_rationale = str(raw_text).strip()
    final_rationale = re.sub(r"(?im)^\s*answer\s*:\s*", "", final_rationale).strip()
    final_rationale = re.sub(r"(?im)^\s*rationale\s*:\s*", "", final_rationale).strip()
    return f"Answer: {final_answer}. Rationale: {final_rationale}".strip()


def _predicted_label_text(batch: Dict, pred_token_id: int) -> str:
    allowed = list(batch.get("allowed_token_ids") or [])
    code_labels = list(batch.get("code_labels") or [])
    lookup = {int(token_id): idx for idx, token_id in enumerate(allowed)}
    idx = lookup.get(int(pred_token_id))
    if idx is None or idx >= len(code_labels):
        return ""
    return str(code_labels[idx]).replace("_", " ").strip()


def _normalize_code_text(s: str) -> str:
    m = re.search(r"[A-Za-z0-9!@#$%^&*()\[\]{}<>?/|]", str(s))
    return m.group(0) if m else ""


def _metric_name_for_task(task_name: str) -> str:
    if task_name == "population":
        return "mae"
    return "accuracy"


def _result_dataset_name(task: TaskSpec) -> str:
    if task.name == "vqa":
        return str(task.vqa_hf_dataset_name)
    return str(task.image_dataset_name)


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
    if model is not None:
        model.eval()
    if runtime is not None:
        runtime.semantic_expert.eval()
        runtime.fusion_policy.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    pop_true = []
    pop_pred = []
    rationale_scores = []
    fmt_ok = 0
    rationale_has_content = 0
    rationale_faithful = 0
    answer_code_valid = 0
    answer_code_correct = 0
    rationale_hallucinated = 0
    gate_values: List[float] = []
    gate_correct: List[int] = []
    eval_wall_seconds = 0.0

    with torch.no_grad():
        for batch in eval_loader:
            t0 = time.perf_counter()
            tokens = batch["prompt_tokens"].to(device)
            mask = batch["prompt_mask"].to(device)
            if baseline_mode == "expert_direct":
                if runtime is None:
                    raise ValueError("expert_direct requires a runtime expert.")
                pred_ids = _expert_direct_pred_ids(runtime, batch, cfg, device)
            else:
                logits = _forward_logits_for_batch(model, runtime, batch, cfg, device, baseline_mode)
                if task.constrained_decoding and batch.get("allowed_token_ids"):
                    pred_ids = _constrained_argmax(logits, batch["allowed_token_ids"]).cpu()
                else:
                    pred_ids = logits.argmax(dim=-1).cpu()

            targets = batch["target_token_id"].cpu()
            correct += (pred_ids == targets).sum().item()
            total += targets.numel()
            semantic_expert = getattr(runtime, "semantic_expert", None) if runtime is not None else None
            if semantic_expert is not None and getattr(semantic_expert, "pre_router_gate", None) is not None:
                vals = semantic_expert.pre_router_gate.detach().float().view(-1).cpu().tolist()
                if vals:
                    gate_values.extend(vals)
                    gate_correct.extend((pred_ids == targets).int().tolist())

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
                    pred_code = _normalize_code_text(tokenizer.decode([int(pred_ids[i].item())]))
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
                        allowed_token_ids=None,  # allow free-text rationale generation
                        cfg=cfg,
                        max_new_tokens=int(task.gen_max_new_tokens),
                        temperature=float(task.gen_temperature),
                        top_p=float(task.gen_top_p),
                        repetition_penalty=float(task.gen_repetition_penalty),
                        do_sample=bool(task.gen_do_sample),
                        prefix_text=(
                            f"Answer: {pred_code}. Rationale: {_predicted_label_text(batch, int(pred_ids[i].item()))} ".strip()
                            if pred_code
                            else ""
                        ),
                    )
                    kws = batch.get("rationale_keywords", [[]])[i]
                    true_code = _normalize_code_text(tokenizer.decode([int(targets[i].item())]))
                    gen = _canonicalize_grounded_output(gen, pred_code)
                    answer_text, rationale_text = _extract_answer_rationale(gen)
                    pred_code = _normalize_code_text(answer_text)
                    if pred_code:
                        answer_code_valid += 1
                    if pred_code and pred_code == true_code:
                        answer_code_correct += 1
                    if answer_text and rationale_text:
                        fmt_ok += 1
                    if not rationale_text:
                        rationale_text = gen
                    if len(rationale_text) >= int(task.min_rationale_chars):
                        rationale_has_content += 1
                    kw_score = keyword_consistency_score(rationale_text, kws)
                    rationale_scores.append(kw_score)
                    if kw_score <= 0.0:
                        rationale_hallucinated += 1
                    if (
                        kw_score >= float(task.min_rationale_keyword_score)
                        and pred_code == true_code
                        and len(rationale_text) >= int(task.min_rationale_chars)
                    ):
                        rationale_faithful += 1
            eval_wall_seconds += max(0.0, time.perf_counter() - t0)

    acc = correct / max(1, total)
    result: Dict[str, float] = {"accuracy": float(acc)}
    result["latency_ms_per_sample"] = float(1000.0 * eval_wall_seconds / max(1, total))

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
        result["answer_code_valid_rate"] = float(answer_code_valid / max(1, len(rationale_scores)))
        result["answer_code_accuracy"] = float(answer_code_correct / max(1, len(rationale_scores)))
        result["rationale_hallucination_rate"] = float(rationale_hallucinated / max(1, len(rationale_scores)))
        result["rationale_faithful_rate"] = float(rationale_faithful / max(1, len(rationale_scores)))
    if gate_values:
        g = torch.tensor(gate_values, dtype=torch.float32)
        result["pre_router_gate_mean"] = float(g.mean().item())
        result["pre_router_gate_std"] = float(g.std(unbiased=False).item())
    else:
        result["pre_router_gate_mean"] = 1.0
        result["pre_router_gate_std"] = 0.0
    if gate_values and gate_correct and len(gate_values) == len(gate_correct):
        g = torch.tensor(gate_values, dtype=torch.float32)
        c = torch.tensor(gate_correct, dtype=torch.float32)
        if float(g.std(unbiased=False).item()) > 0.0 and float(c.std(unbiased=False).item()) > 0.0:
            result["pre_router_gate_accuracy_corr"] = float(torch.corrcoef(torch.stack([g, c]))[0, 1].item())
        else:
            result["pre_router_gate_accuracy_corr"] = 0.0
    else:
        result["pre_router_gate_accuracy_corr"] = 0.0
    return result


def _metric_fields_for_aggregate(rows: List[Dict]) -> List[str]:
    out = []
    for row in rows:
        for k, v in row.items():
            if k in {"seed", "status", "error", "traceback"}:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append(k)
    return sorted(set(out))


def _append_seed_aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    passthrough = []
    for row in rows:
        if row.get("seed") is None or row.get("seed") == "aggregate":
            passthrough.append(row)
            continue
        key = (row.get("task"), row.get("dataset"), row.get("baseline_mode"), row.get("backbone"), row.get("expert_type"))
        grouped.setdefault(key, []).append(row)

    agg_rows: List[Dict] = []
    for _, group in grouped.items():
        if len(group) <= 1:
            continue
        first = dict(group[0])
        first["seed"] = "aggregate"
        first["seed_count"] = len(group)
        first["status"] = "ok" if all(g.get("status") == "ok" for g in group) else "mixed"
        metric_name = str(first.get("metric_name") or _metric_name_for_task(str(first.get("task", ""))))
        for m in _metric_fields_for_aggregate(group):
            vals = [float(g[m]) for g in group if isinstance(g.get(m), (int, float))]
            if not vals:
                continue
            mean = float(sum(vals) / len(vals))
            var = float(sum((x - mean) ** 2 for x in vals) / len(vals))
            first[f"{m}_mean"] = mean
            first[f"{m}_std"] = var ** 0.5
        if f"{metric_name}_mean" in first:
            first[metric_name] = float(first[f"{metric_name}_mean"])
        if "metric_mean" in first:
            first["metric"] = float(first["metric_mean"])
        agg_rows.append(first)
    return rows + agg_rows


def _save_rows(rows: List[Dict], out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
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


def _as_float(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _row_value_or_mean(row: Dict, key: str, default: float) -> float:
    for k in (f"{key}_mean", key):
        v = _as_float(row.get(k))
        if v is not None:
            return v
    return float(default)


def _quality_score_from_row(row: Dict) -> float:
    task_name = str(row.get("task", "")).strip().lower()
    if task_name == "population":
        mae = _row_value_or_mean(row, "mae", default=1.0)
        return float(max(0.0, min(1.0, 1.0 - mae)))
    acc = _row_value_or_mean(row, "accuracy", default=_row_value_or_mean(row, "metric", default=0.0))
    return float(max(0.0, min(1.0, acc)))


def _instability_from_row(row: Dict) -> float:
    metric_name = str(row.get("metric_name") or _metric_name_for_task(str(row.get("task", ""))))
    for k in ("metric_std", f"{metric_name}_std"):
        v = _as_float(row.get(k))
        if v is not None:
            return float(max(0.0, v))
    return 0.0


def _annotate_joint_scores(rows: List[Dict], cfg: TaskMatrixConfig) -> List[Dict]:
    latency_w = float(getattr(cfg, "joint_score_latency_weight", 0.03))
    format_w = float(getattr(cfg, "joint_score_format_weight", 0.25))
    instability_w = float(getattr(cfg, "joint_score_instability_weight", 0.5))
    for row in rows:
        quality = _quality_score_from_row(row)
        fmt = _row_value_or_mean(row, "format_compliance", default=1.0)
        format_fail = float(max(0.0, min(1.0, 1.0 - fmt)))
        latency_ms = _row_value_or_mean(row, "latency_ms_per_sample", default=0.0)
        latency_s = float(max(0.0, latency_ms) / 1000.0)
        instability = _instability_from_row(row)
        joint = quality - (latency_w * latency_s) - (format_w * format_fail) - (instability_w * instability)
        row["quality_score"] = float(quality)
        row["format_fail_rate"] = float(format_fail)
        row["latency_s_per_sample"] = float(latency_s)
        row["instability_penalty"] = float(instability)
        row["joint_score"] = float(joint)
    return rows


def run_task_matrix(cfg: TaskMatrixConfig) -> List[Dict]:
    set_seed(cfg.seed)
    device = _device(cfg.device)
    rows: List[Dict] = []
    baseline_modes = _normalize_baseline_modes(cfg.baseline_modes)
    run_seeds = [cfg.seed] + [int(s) for s in (cfg.repeat_seeds or []) if int(s) != int(cfg.seed)]

    for seed_idx, run_seed in enumerate(run_seeds):
        set_seed(run_seed)
        for i, task in enumerate(cfg.tasks):
            for baseline_mode in baseline_modes:
                cfg_run = _resolve_cfg_for_task(cfg, task, baseline_mode)
                try:
                    if baseline_mode == "expert_direct":
                        model = None
                        tokenizer = _build_tokenizer(cfg_run)
                        runtime = _attach_direct_expert(cfg_run)
                    else:
                        model, tokenizer = _build_model_tokenizer(cfg_run)
                        runtime = _attach_connector(cfg_run, model, baseline_mode)
                        model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                        if device.type == "cuda":
                            model = model.to(dtype=model_dtype).to(device=device)
                        else:
                            model = model.to(device=device, dtype=model_dtype)
                    if runtime is not None:
                        runtime_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                        runtime.semantic_expert = runtime.semantic_expert.to(device=device, dtype=runtime_dtype)
                        runtime.fusion_policy = runtime.fusion_policy.to(device=device, dtype=runtime_dtype)
                    train_loader, eval_loader = _build_task_loaders(task, tokenizer, cfg_run.data_root, run_seed + i)
                    _train_connector(model, runtime, train_loader, task, cfg_run, device, baseline_mode)
                    metrics = _eval_task(model, runtime, tokenizer, eval_loader, task, cfg_run, device, baseline_mode)

                    if cfg_run.save_bundle_dir and runtime is not None:
                        bundle_dir = os.path.join(cfg_run.save_bundle_dir, f"{task.name}_{baseline_mode}")
                        enc_spec = {
                            "expert_type": cfg_run.expert_type,
                            "model_id": cfg_run.expert_model_id,
                            "model_path": cfg_run.expert_model_path,
                            "output_dim": cfg_run.expert_output_dim,
                            "cache_dir": cfg_run.server_models_path,
                            "use_runtime_detector": cfg_run.use_runtime_detector,
                        }
                        save_expert_bundle(
                            bundle_dir,
                            encoder_spec=enc_spec,
                            semantic_expert=runtime.semantic_expert,
                            fusion_policy=runtime.fusion_policy,
                            normalization_stats=cfg_run.normalization_stats,
                            label_schema=cfg_run.label_schema,
                            upload_scope=str(getattr(cfg_run, "upload_scope", "expert_only")),
                            metadata={
                                "task": task.name,
                                "model_name": cfg_run.model_name,
                                "fusion_policy": _fusion_policy_for_mode(cfg_run, baseline_mode),
                                "layer_idx": cfg_run.layer_idx,
                                "feature_format": "tabular"
                                if cfg_run.expert_type in {"xgboost", "ft_transformer", "tabpfn", "tabular"}
                                else "image_tensor",
                            },
                        )

                    row = {
                        "task": task.name,
                        "dataset": _result_dataset_name(task),
                        "baseline_mode": baseline_mode,
                        "expert_type": cfg_run.expert_type,
                        "backbone": cfg_run.model_name,
                        "fusion_policy": _fusion_policy_for_mode(cfg_run, baseline_mode),
                        "k": cfg_run.num_tokens if runtime is not None else 0,
                        "layer_idx": cfg_run.layer_idx,
                        "evidence_dim": cfg_run.evidence_dim if runtime is not None else 0,
                        "use_pre_router": bool(getattr(cfg_run, "use_pre_router", False)),
                        "pre_router_mode": str(getattr(cfg_run, "pre_router_mode", "global_feature")),
                        "task_conditioning": bool(getattr(cfg_run, "task_conditioning", False)),
                        "metric_name": _metric_name_for_task(task.name),
                        "metric": metrics.get("mae", metrics.get("accuracy", 0.0)),
                        "seed": int(run_seed),
                        "seed_index": int(seed_idx),
                        "status": "ok",
                    }
                    if baseline_mode == "text_prompt_only":
                        row["baseline_note"] = "Prompt-only baseline over the standardized task text; no uploaded expert or injection path."
                    row.update(metrics)
                except Exception as exc:
                    row = {
                        "task": task.name,
                        "dataset": _result_dataset_name(task),
                        "baseline_mode": baseline_mode,
                        "expert_type": cfg_run.expert_type,
                        "backbone": cfg_run.model_name,
                        "fusion_policy": _fusion_policy_for_mode(cfg_run, baseline_mode),
                        "k": cfg_run.num_tokens if baseline_mode in {"overwrite", "router_parallel"} else 0,
                        "layer_idx": cfg_run.layer_idx,
                        "evidence_dim": cfg_run.evidence_dim if baseline_mode in {"overwrite", "router_parallel"} else 0,
                        "use_pre_router": bool(getattr(cfg_run, "use_pre_router", False)),
                        "pre_router_mode": str(getattr(cfg_run, "pre_router_mode", "global_feature")),
                        "task_conditioning": bool(getattr(cfg_run, "task_conditioning", False)),
                        "pre_router_gate_mean": 1.0,
                        "pre_router_gate_std": 0.0,
                        "pre_router_gate_accuracy_corr": 0.0,
                        "metric_name": _metric_name_for_task(task.name),
                        "metric": 0.0,
                        "seed": int(run_seed),
                        "seed_index": int(seed_idx),
                        "status": "error",
                        "error": str(exc) or repr(exc),
                        "traceback": traceback.format_exc(limit=5),
                    }
                rows.append(row)
                print(row)

    if cfg.aggregate_seed_metrics:
        rows = _append_seed_aggregate_rows(rows)
    rows = _annotate_joint_scores(rows, cfg)

    _save_rows(rows, cfg.out_csv, cfg.out_json)
    return rows


def load_task_matrix_config(path: str) -> TaskMatrixConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks = [TaskSpec(**t) for t in raw.pop("tasks", [])]
    cfg = TaskMatrixConfig(**raw)
    cfg.tasks = tasks if tasks else cfg.tasks
    return cfg
