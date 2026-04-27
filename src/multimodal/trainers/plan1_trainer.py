from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

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
from src.multimodal.data.hospital_text_qa import HospitalTextQADataset, collate_hospital_text
from src.multimodal.evidence import (
    DiffusionFutureEvidenceBuilder,
    PopulationStatsEvidenceBuilder,
    TextEvidenceBuilder,
    VisionEvidenceBuilder,
)
from src.multimodal.experts import build_vision_expert
from src.multimodal.injection import (
    EvidencePreRouter,
    EvidenceProjector,
    SemanticEvidenceDomainExpert,
    task_name_to_id,
)
from src.multimodal.tasks.prompts import class_names_for_dataset
from src.multimodal.utils.backbone import resolve_backbone
from src.multimodal.utils.logging_utils import MetricLogger
from src.multimodal.utils.repro import save_run_config, save_versions, set_seed
from transformers import AutoTokenizer as EncoderTokenizer


@dataclass
class TrainPlan1Args:
    server_models_path: str = "/disk1/lfhu/hf_cache"
    model_name: str = "Qwen/Qwen3.5-9B"
    run_dir: str = "runs/plan1_default"
    dataset_name: str = "dtd"
    hospital_train_file: Optional[str] = None
    hospital_eval_file: Optional[str] = None
    data_root: str = "./data"
    task_family: str = "single_image"
    evidence_source: str = "text"  # vision|text|diffusion_future
    description_file: Optional[str] = None
    qa_type: str = "label"  # label|yesno|index
    population_output_mode: str = "integer"  # integer|letter
    population_group_size: int = 16
    population_sigma: float = 0.0
    num_tokens: int = 4
    layer_to_add: int = 0
    evidence_dim: int = 256
    alpha: float = 1.0
    expert_kind: str = "classifier"  # classifier|embedding
    expert_output_mode: str = "logits"  # logits|embedding
    expert_checkpoint: Optional[str] = None
    expert_init_weights: str = "imagenet"
    text_encoder_model_id: str = "distilroberta-base"
    batch_size: int = 8
    num_epochs: int = 1
    learning_rate: float = 1e-4
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 512
    eval_every_steps: int = 100
    answer_only_loss: bool = True
    use_adapters: bool = True
    adapter_bottleneck: Optional[int] = 64
    adapter_dropout: float = 0.0
    adapter_activation: str = "gelu"
    tune_layernorm: bool = False
    inject_location: str = "layer_input"
    fusion_policy: str = "post_attn_router_parallel"
    use_pre_router: bool = False
    pre_router_mode: str = "global_feature"  # global|feature|global_feature
    pre_router_hidden_dim: int = 128
    task_conditioning: bool = False
    pre_router_task_embedding_dim: int = 16
    pre_router_max_task_ids: int = 32
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_cpu_offload: bool = False
    device_map: Optional[str] = None
    seed: int = 42
    device: str = "cuda"


def _build_model_and_tokenizer(args: TrainPlan1Args):
    resolved_model_name, reason = resolve_backbone(args.model_name, args.server_models_path)
    if resolved_model_name != args.model_name:
        print(f"[plan1] backbone: {args.model_name} -> {resolved_model_name} ({reason})")
    args.model_name = resolved_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.server_models_path,
        torch_dtype=torch.bfloat16,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        args.model_name,
        cache_dir=args.server_models_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_cpu_offload=args.bnb_4bit_cpu_offload,
        device_map=args.device_map,
        tokenizer=tokenizer,
    )
    model.num_tokens = args.num_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for p in model.parameters():
        p.requires_grad = False
    model.configure_injection_(layer_id=args.layer_to_add, inject_location=args.inject_location)
    model.configure_adapters_(
        use_adapters=args.use_adapters,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        adapter_activation=args.adapter_activation,
    )
    model.set_layernorm_tuning_(requires_grad=args.tune_layernorm and args.use_adapters)
    return model, tokenizer


def _num_classes_for_dataset(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "dtd":
        return 47
    if dataset_name == "oxford_pet":
        return 37
    if dataset_name == "hospital_text":
        # only used by vision/population paths; text path infers classes from file dataset
        return 2
    raise ValueError(dataset_name)


def _attach_semantic_expert(args: TrainPlan1Args, model, tokenizer):
    hidden_size = model.config.hidden_size
    num_classes = _num_classes_for_dataset(args.dataset_name)

    if args.task_family == "population":
        evidence_builder = PopulationStatsEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            num_classes=num_classes,
            sigma=args.population_sigma,
        )
        vision_expert = build_vision_expert(
            expert_kind="classifier",
            checkpoint_path=args.expert_checkpoint,
            num_classes=num_classes,
            init_weights=args.expert_init_weights,
        )
    elif args.evidence_source == "vision":
        vision_expert = build_vision_expert(
            expert_kind=args.expert_kind,
            checkpoint_path=args.expert_checkpoint,
            num_classes=num_classes,
            init_weights=args.expert_init_weights,
        )
        evidence_builder = VisionEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            vision_expert=vision_expert,
            expert_output_mode=args.expert_output_mode,
            num_classes=num_classes,
            embedding_dim=512,
            use_embedding_adapter=False,
        )
    elif args.evidence_source == "text":
        vision_expert = None
        evidence_builder = TextEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            text_encoder_model_id=args.text_encoder_model_id,
            cache_dir=args.server_models_path,
        )
    elif args.evidence_source == "diffusion_future":
        vision_expert = None
        evidence_builder = DiffusionFutureEvidenceBuilder(evidence_dim=args.evidence_dim)
    else:
        raise ValueError(args.evidence_source)

    projector = EvidenceProjector(
        evidence_dim=args.evidence_dim,
        hidden_size=hidden_size,
        num_tokens=args.num_tokens,
        alpha=args.alpha,
    )
    pre_router = None
    if bool(getattr(args, "use_pre_router", False)):
        pre_router = EvidencePreRouter(
            evidence_dim=args.evidence_dim,
            mode=str(getattr(args, "pre_router_mode", "global_feature")),
            hidden_dim=int(getattr(args, "pre_router_hidden_dim", 128)),
            task_conditioning=bool(getattr(args, "task_conditioning", False)),
            task_embedding_dim=int(getattr(args, "pre_router_task_embedding_dim", 16)),
            max_task_ids=int(getattr(args, "pre_router_max_task_ids", 32)),
        )
    semantic_expert = SemanticEvidenceDomainExpert(
        evidence_builder=evidence_builder,
        projector=projector,
        pre_router=pre_router,
    )
    if hasattr(model, "add_expert_"):
        model.add_expert_(semantic_expert, layer_id=args.layer_to_add, map_to_expert_emb=None)
    else:
        layer = model.model.layers[args.layer_to_add]
        if hasattr(layer, "add_expert_"):
            layer.add_expert_(semantic_expert, map_to_expert_emb=None)
        else:
            if not hasattr(model, "_external_experts"):
                model._external_experts = nn.ModuleList()
            model._external_experts.append(semantic_expert)
    model.configure_injection_(layer_id=args.layer_to_add, inject_location=args.inject_location)
    return model, semantic_expert, vision_expert


def _build_dataloaders(args: TrainPlan1Args, tokenizer):
    if args.dataset_name == "hospital_text":
        if not args.hospital_train_file or not args.hospital_eval_file:
            raise ValueError("hospital_text requires --hospital_train_file and --hospital_eval_file")
        train_ds = HospitalTextQADataset(
            file_path=args.hospital_train_file,
            tokenizer=tokenizer,
            qa_type=args.qa_type,
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
        eval_ds = HospitalTextQADataset(
            file_path=args.hospital_eval_file,
            tokenizer=tokenizer,
            qa_type=args.qa_type,
            seed=args.seed,
            max_samples=args.max_eval_samples,
        )
        return (
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hospital_text),
            DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hospital_text),
        )

    if args.task_family == "single_image":
        train_ds = CIFARSingleImageQADataset(
            root=args.data_root,
            tokenizer=tokenizer,
            train=True,
            dataset_name=args.dataset_name,
            mode=args.evidence_source if args.evidence_source in {"vision", "text"} else "vision",
            qa_type=args.qa_type,
            seed=args.seed,
            max_samples=args.max_train_samples,
            description_file=args.description_file,
        )
        eval_ds = CIFARSingleImageQADataset(
            root=args.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=args.dataset_name,
            mode=args.evidence_source if args.evidence_source in {"vision", "text"} else "vision",
            qa_type=args.qa_type,
            seed=args.seed,
            max_samples=args.max_eval_samples,
            description_file=args.description_file,
        )
        return (
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_single_image),
            DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_image),
        )

    train_ds = CIFARPopulationDataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=True,
        dataset_name=args.dataset_name,
        group_size=args.population_group_size,
        output_mode=args.population_output_mode,
        seed=args.seed,
        max_groups=args.max_train_samples,
    )
    eval_ds = CIFARPopulationDataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=False,
        dataset_name=args.dataset_name,
        group_size=args.population_group_size,
        output_mode=args.population_output_mode,
        seed=args.seed,
        max_groups=args.max_eval_samples,
    )
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_population),
        DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_population),
    )


def _build_expert_inputs(args: TrainPlan1Args, batch: Dict, device, text_tokenizer, semantic_expert, vision_expert):
    if args.task_family == "population":
        images = batch["images"].to(device)  # (B,N,C,H,W)
        b, n = images.shape[:2]
        flat = images.view(b * n, *images.shape[2:])
        with torch.no_grad():
            logits = vision_expert(flat).logits.float()
            probs = torch.softmax(logits, dim=-1).view(b, n, -1)
            mean_prob = probs.mean(dim=1)
        return ({"mean_prob": mean_prob, "n_images": torch.full((b,), n, device=device)},)

    if args.evidence_source == "vision":
        return (batch["images"].to(device),)

    if args.evidence_source == "text":
        encoded = text_tokenizer(
            batch["description"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        return (
            {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
            },
        )

    raise ValueError(args.evidence_source)


def _task_key_from_batch_args(args: TrainPlan1Args) -> str:
    if args.task_family == "population":
        return "population"
    if args.qa_type == "label_code":
        return "single_label_code"
    if args.qa_type == "yesno_set2":
        return "single_yesno_set2"
    if args.qa_type == "yesno":
        return "single_yesno"
    return "single_label"


def _task_ids_for_batch(args: TrainPlan1Args, batch: Dict, device: torch.device) -> Optional[torch.Tensor]:
    if not bool(getattr(args, "task_conditioning", False)):
        return None
    bsz = int(batch["prompt_tokens"].size(0))
    task_id = task_name_to_id(_task_key_from_batch_args(args), max_task_ids=int(getattr(args, "pre_router_max_task_ids", 32)))
    return torch.full((bsz,), int(task_id), dtype=torch.long, device=device)


def _set_expert_task_context(
    semantic_expert: Optional[SemanticEvidenceDomainExpert],
    *,
    task_id: Optional[torch.Tensor],
) -> None:
    if semantic_expert is None:
        return
    if getattr(semantic_expert, "pre_router", None) is None:
        return
    if task_id is None:
        semantic_expert.clear_task_context()
        return
    semantic_expert.set_task_context(task_id=task_id)


def _collect_trainable_named_parameters(model: nn.Module, semantic_expert: SemanticEvidenceDomainExpert):
    seen = set()
    rows = []
    for prefix, module in (("model", model), ("semantic_expert", semantic_expert)):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            rows.append((f"{prefix}.{name}", int(param.numel())))
    return rows


def _print_trainable_parameter_report(model: nn.Module, semantic_expert: SemanticEvidenceDomainExpert):
    rows = _collect_trainable_named_parameters(model, semantic_expert)
    total = int(sum(n for _, n in rows))
    print(f"[plan1] trainable_parameter_count={total}")
    for name, n in rows:
        print(f"[plan1] trainable {name} params={n}")


def _single_token_ids(tokenizer, text: str):
    ids = set()
    variants = [text, text.lower(), text.upper()]
    for v in variants:
        for prefix in (" ", ""):
            toks = tokenizer.encode(prefix + v, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return ids


def _allowed_answer_token_ids(args: TrainPlan1Args, tokenizer, dataset):
    ids = set()
    qa_mode = getattr(dataset, "effective_qa_type", args.qa_type)
    class_names = getattr(dataset, "class_names", class_names_for_dataset(args.dataset_name))
    code_to_token_id = getattr(dataset, "code_to_token_id", None)

    if args.task_family == "single_image":
        if qa_mode in {"yesno", "yesno_set2"}:
            ids.update(_single_token_ids(tokenizer, "Yes"))
            ids.update(_single_token_ids(tokenizer, "No"))
        elif qa_mode == "label_code" and code_to_token_id is not None:
            ids.update(int(tid) for tid in code_to_token_id.values())
        elif qa_mode == "index":
            for i in range(len(class_names)):
                ids.update(_single_token_ids(tokenizer, str(i)))
        elif qa_mode == "label" and len(class_names) <= 26:
            for i in range(len(class_names)):
                ids.update(_single_token_ids(tokenizer, chr(ord("A") + i)))
    else:
        if args.population_output_mode == "integer":
            for i in range(11):
                ids.update(_single_token_ids(tokenizer, str(i)))
        else:
            for i in range(10):
                ids.update(_single_token_ids(tokenizer, chr(ord("A") + i)))
    return sorted(ids)


def _subset_ce_loss(logits: torch.Tensor, targets: torch.Tensor, allowed_ids_device: Optional[torch.Tensor], fallback_ce):
    if allowed_ids_device is None or allowed_ids_device.numel() == 0:
        return fallback_ce(logits, targets)
    subset_logits = logits.index_select(dim=-1, index=allowed_ids_device)
    pos = torch.searchsorted(allowed_ids_device, targets)
    in_range = pos < allowed_ids_device.numel()
    safe_pos = torch.where(in_range, pos, torch.zeros_like(pos))
    valid = in_range & (allowed_ids_device[safe_pos] == targets)
    if not bool(valid.all()):
        return fallback_ce(logits, targets)
    return fallback_ce(subset_logits, pos)


@torch.no_grad()
def _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert):
    model.eval()
    correct = 0
    total = 0
    gate_vals = []
    for batch in eval_loader:
        tokens = batch["prompt_tokens"].to(device)
        mask = batch["prompt_mask"].to(device)
        targets = batch["target_token_id"].to(device)
        expert_inputs = _build_expert_inputs(args, batch, device, text_tokenizer, semantic_expert, vision_expert)
        task_ids = _task_ids_for_batch(args, batch, device)
        _set_expert_task_context(semantic_expert, task_id=task_ids)
        outputs = model(tokens, attention_mask=mask, expert_inputs=expert_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        if getattr(semantic_expert, "pre_router_gate", None) is not None:
            gate_vals.extend(semantic_expert.pre_router_gate.detach().float().view(-1).cpu().tolist())
        semantic_expert.clear_task_context()
    out = {"accuracy": correct / max(1, total)}
    if gate_vals:
        g = torch.tensor(gate_vals, dtype=torch.float32)
        out["pre_router_gate_mean"] = float(g.mean().item())
        out["pre_router_gate_std"] = float(g.std(unbiased=False).item())
    else:
        out["pre_router_gate_mean"] = 1.0
        out["pre_router_gate_std"] = 0.0
    return out


def train_plan1(args: TrainPlan1Args):
    set_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    save_run_config(asdict(args), args.run_dir)
    save_versions(args.run_dir)
    logger = MetricLogger(args.run_dir)

    model, tokenizer = _build_model_and_tokenizer(args)
    model, semantic_expert, vision_expert = _attach_semantic_expert(args, model, tokenizer)
    train_loader, eval_loader = _build_dataloaders(args, tokenizer)

    text_tokenizer = None
    if args.evidence_source == "text":
        text_tokenizer = EncoderTokenizer.from_pretrained(args.text_encoder_model_id, cache_dir=args.server_models_path)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if not model.is_quantized_4bit:
        model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)
    if vision_expert is not None:
        vision_expert = vision_expert.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    _print_trainable_parameter_report(model, semantic_expert)
    loss_fn = nn.CrossEntropyLoss()
    allowed_ids = _allowed_answer_token_ids(args, tokenizer, train_loader.dataset)
    allowed_ids_device = (
        torch.tensor(allowed_ids, dtype=torch.long, device=device) if args.answer_only_loss and len(allowed_ids) > 0 else None
    )

    global_step = 0
    best_eval = -1.0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            tokens = batch["prompt_tokens"].to(device)
            mask = batch["prompt_mask"].to(device)
            targets = batch["target_token_id"].to(device)

            expert_inputs = _build_expert_inputs(args, batch, device, text_tokenizer, semantic_expert, vision_expert)
            task_ids = _task_ids_for_batch(args, batch, device)
            _set_expert_task_context(semantic_expert, task_id=task_ids)
            outputs = model(tokens, attention_mask=mask, expert_inputs=expert_inputs, use_cache=False)
            logits = outputs.logits[:, -1, :]
            loss = _subset_ce_loss(logits, targets, allowed_ids_device, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            batch_acc = (pred == targets).float().mean().item()
            row = {
                "split": "train",
                "epoch": epoch,
                "step": global_step,
                "loss": float(loss.item()),
                "acc": float(batch_acc),
                "pre_router_gate_mean": float(semantic_expert.pre_router_gate_stats()["mean"]),
                "pre_router_gate_std": float(semantic_expert.pre_router_gate_stats()["std"]),
            }
            logger.log(row)
            global_step += 1
            semantic_expert.clear_task_context()

            if global_step % args.eval_every_steps == 0:
                eval_metrics = _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert)
                eval_acc = float(eval_metrics.get("accuracy", 0.0))
                logger.log({"split": "eval", "epoch": epoch, "step": global_step, "acc": eval_acc, **eval_metrics})
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    torch.save(
                        {
                            "evidence_builder": semantic_expert.evidence_builder.state_dict(),
                            "projector": semantic_expert.projector.state_dict(),
                            "pre_router": semantic_expert.pre_router.state_dict() if semantic_expert.pre_router is not None else None,
                            "adapters": model.adapter_state_dict(),
                            "layernorm": {
                                str(i): {
                                    "input_layernorm": layer.input_layernorm.state_dict(),
                                    "post_attention_layernorm": layer.post_attention_layernorm.state_dict(),
                                }
                                for i, layer in enumerate(model.model.layers)
                                if hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm")
                            },
                            "args": asdict(args),
                        },
                        os.path.join(args.run_dir, "best_connectors.pt"),
                    )

    final_eval_metrics = _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert)
    final_eval = float(final_eval_metrics.get("accuracy", 0.0))
    logger.log({"split": "eval", "epoch": args.num_epochs, "step": global_step, "acc": final_eval, **final_eval_metrics})
    torch.save(
        {
            "evidence_builder": semantic_expert.evidence_builder.state_dict(),
            "projector": semantic_expert.projector.state_dict(),
            "pre_router": semantic_expert.pre_router.state_dict() if semantic_expert.pre_router is not None else None,
            "adapters": model.adapter_state_dict(),
            "layernorm": {
                str(i): {
                    "input_layernorm": layer.input_layernorm.state_dict(),
                    "post_attention_layernorm": layer.post_attention_layernorm.state_dict(),
                }
                for i, layer in enumerate(model.model.layers)
                if hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm")
            },
            "args": asdict(args),
        },
        os.path.join(args.run_dir, "last_connectors.pt"),
    )
    return {
        "final_eval_acc": final_eval,
        "best_eval_acc": best_eval,
        "pre_router_gate_mean": float(final_eval_metrics.get("pre_router_gate_mean", 1.0)),
        "pre_router_gate_std": float(final_eval_metrics.get("pre_router_gate_std", 0.0)),
        "use_pre_router": bool(getattr(args, "use_pre_router", False)),
        "pre_router_mode": str(getattr(args, "pre_router_mode", "global_feature")),
        "task_conditioning": bool(getattr(args, "task_conditioning", False)),
    }
