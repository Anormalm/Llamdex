from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoTokenizer as EncoderTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.data.hospital_text_qa import HospitalTextQADataset, collate_hospital_text
from src.multimodal.evidence import PopulationStatsEvidenceBuilder, TextEvidenceBuilder, VisionEvidenceBuilder
from src.multimodal.experts import build_vision_expert
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.tasks.parsing import (
    parse_label_answer,
    parse_population_answer,
    population_fraction_to_bin,
    population_bin_to_fraction_midpoint,
)
from src.multimodal.tasks.prompts import class_names_for_dataset
from src.multimodal.utils.backbone import resolve_backbone


def _chat_template_to_ids(tokenizer, messages) -> torch.Tensor:
    try:
        out = tokenizer.apply_chat_template(messages, return_tensors="pt")
    except Exception:
        plain = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        out = tokenizer(plain, return_tensors="pt", add_special_tokens=True)["input_ids"]
    if isinstance(out, torch.Tensor):
        return out.squeeze(0).long()
    if hasattr(out, "input_ids"):
        ids = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        ids = out["input_ids"]
    else:
        raise TypeError("Unsupported apply_chat_template return type for tokenizer.")
    if isinstance(ids, list):
        ids = torch.tensor(ids, dtype=torch.long)
    if isinstance(ids, torch.Tensor) and ids.dim() == 2:
        return ids.squeeze(0).long()
    return ids.long()


@dataclass
class EvalPlan1Args:
    mistral_models_path: str = "model/llm"
    model_name: str = "Qwen/Qwen3.5-9B"
    connectors_path: Optional[str] = None
    dataset_name: str = "dtd"
    hospital_eval_file: Optional[str] = None
    data_root: str = "./data"
    task_family: str = "single_image"
    evidence_source: str = "text"
    description_file: Optional[str] = None
    qa_type: str = "label"
    population_output_mode: str = "integer"
    population_group_size: int = 16
    population_sigma: float = 0.0
    num_tokens: int = 4
    layer_to_add: int = 0
    evidence_dim: int = 256
    alpha: float = 1.0
    expert_kind: str = "classifier"
    expert_output_mode: str = "logits"
    expert_checkpoint: Optional[str] = None
    expert_init_weights: str = "imagenet"
    text_encoder_model_id: str = "distilroberta-base"
    batch_size: int = 8
    max_eval_samples: Optional[int] = 512
    baseline: str = "injection"  # injection|vision_only|llm_only|text_prompt
    device: str = "cuda"
    constrain_llm_only_outputs: bool = True
    use_adapters: bool = True
    adapter_bottleneck: Optional[int] = 64
    adapter_dropout: float = 0.0
    adapter_activation: str = "gelu"
    tune_layernorm: bool = False
    inject_location: str = "post_attn"
    enforce_checkpoint_compat: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_cpu_offload: bool = False
    device_map: Optional[str] = None


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
        return 2
    raise ValueError(dataset_name)


def _single_token_ids(tokenizer, text: str):
    ids = set()
    variants = [text, text.lower(), text.upper()]
    for v in variants:
        for prefix in (" ", ""):
            toks = tokenizer.encode(prefix + v, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return ids


def _allowed_answer_token_ids(args: EvalPlan1Args, tokenizer, class_names):
    ids = set()
    if args.task_family == "single_image":
        if args.qa_type in {"yesno", "yesno_set2"}:
            ids.update(_single_token_ids(tokenizer, "Yes"))
            ids.update(_single_token_ids(tokenizer, "No"))
        elif args.qa_type == "label_code":
            # Build a stable single-token code set (same candidate order as dataset codebook).
            candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()[]{}<>?/|")
            for c in candidates:
                if len(ids) >= len(class_names):
                    break
                one = _single_token_ids(tokenizer, c)
                if one:
                    ids.update(one)
        elif args.qa_type == "index":
            for i in range(len(class_names)):
                ids.update(_single_token_ids(tokenizer, str(i)))
        else:
            if len(class_names) <= 26:
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


def _yes_no_token_sets(tokenizer):
    yes_ids = set(_single_token_ids(tokenizer, "Yes"))
    no_ids = set(_single_token_ids(tokenizer, "No"))
    return yes_ids, no_ids


def _build_model(args: EvalPlan1Args):
    resolved_model_name, reason = resolve_backbone(args.model_name, args.mistral_models_path)
    if resolved_model_name != args.model_name:
        print(f"[plan1] backbone: {args.model_name} -> {resolved_model_name} ({reason})")
    args.model_name = resolved_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.mistral_models_path,
        torch_dtype=torch.bfloat16,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        args.model_name,
        cache_dir=args.mistral_models_path,
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
    model.set_layernorm_tuning_(requires_grad=False)
    return model, tokenizer


def _build_eval_dataset(args: EvalPlan1Args, tokenizer):
    if args.dataset_name == "hospital_text":
        if not args.hospital_eval_file:
            raise ValueError("hospital_text requires --hospital_eval_file")
        ds = HospitalTextQADataset(
            file_path=args.hospital_eval_file,
            tokenizer=tokenizer,
            qa_type=args.qa_type,
            max_samples=args.max_eval_samples,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hospital_text)

    if args.task_family == "single_image":
        ds = CIFARSingleImageQADataset(
            root=args.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=args.dataset_name,
            mode=args.evidence_source if args.evidence_source in {"vision", "text"} else "vision",
            qa_type=args.qa_type,
            max_samples=args.max_eval_samples,
            description_file=args.description_file,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_image)

    ds = CIFARPopulationDataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=False,
        dataset_name=args.dataset_name,
        group_size=args.population_group_size,
        output_mode=args.population_output_mode,
        max_groups=args.max_eval_samples,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_population)


def _build_connectors(args: EvalPlan1Args, model):
    hidden = model.config.hidden_size
    num_classes = _num_classes_for_dataset(args.dataset_name)
    vision_expert = build_vision_expert(
        expert_kind="classifier" if args.task_family == "population" else args.expert_kind,
        checkpoint_path=args.expert_checkpoint,
        num_classes=num_classes,
        init_weights=args.expert_init_weights,
    )
    if args.task_family == "population":
        builder = PopulationStatsEvidenceBuilder(args.evidence_dim, num_classes=num_classes, sigma=args.population_sigma)
    elif args.evidence_source == "vision":
        builder = VisionEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            vision_expert=vision_expert,
            expert_output_mode=args.expert_output_mode,
            num_classes=num_classes,
            embedding_dim=512,
            use_embedding_adapter=False,
        )
    elif args.evidence_source == "text":
        builder = TextEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            text_encoder_model_id=args.text_encoder_model_id,
            cache_dir=args.mistral_models_path,
        )
    else:
        raise ValueError(args.evidence_source)
    projector = EvidenceProjector(args.evidence_dim, hidden, args.num_tokens, alpha=args.alpha)
    expert = SemanticEvidenceDomainExpert(builder, projector)
    model.model.layers[args.layer_to_add].add_expert_(expert, map_to_expert_emb=None)

    def _safe_load(module: torch.nn.Module, state_dict: Dict):
        cur = module.state_dict()
        keep = {}
        for k, v in state_dict.items():
            if k in cur and hasattr(v, "shape") and hasattr(cur[k], "shape") and tuple(v.shape) == tuple(cur[k].shape):
                keep[k] = v
        module.load_state_dict(keep, strict=False)

    if args.connectors_path:
        state = torch.load(args.connectors_path, map_location="cpu")
        if args.enforce_checkpoint_compat and "args" in state and isinstance(state["args"], dict):
            ck = state["args"]
            checks = {
                "dataset_name": args.dataset_name,
                "task_family": args.task_family,
                "qa_type": args.qa_type,
                "evidence_source": args.evidence_source,
                "evidence_dim": int(args.evidence_dim),
                "num_tokens": int(args.num_tokens),
                "layer_to_add": int(args.layer_to_add),
                "expert_kind": args.expert_kind,
                "expert_output_mode": args.expert_output_mode,
                "use_adapters": bool(args.use_adapters),
                "inject_location": args.inject_location,
            }
            mismatches = []
            for k, v in checks.items():
                if k in ck and ck[k] != v:
                    mismatches.append(f"{k}: ckpt={ck[k]} vs eval={v}")
            if mismatches:
                raise ValueError(
                    "Connector checkpoint is incompatible with current eval args. "
                    "This often causes collapsed output. "
                    "Mismatches: " + "; ".join(mismatches)
                )
        if "evidence_builder" in state:
            _safe_load(expert.evidence_builder, state["evidence_builder"])
        if "projector" in state:
            _safe_load(expert.projector, state["projector"])
        if "adapters" in state and args.use_adapters:
            model.load_adapter_state_dict(state["adapters"], strict=False)
        if "layernorm" in state and args.tune_layernorm:
            for i, layer in enumerate(model.model.layers):
                key = str(i)
                if key not in state["layernorm"]:
                    continue
                ln_state = state["layernorm"][key]
                if "input_layernorm" in ln_state:
                    layer.input_layernorm.load_state_dict(ln_state["input_layernorm"], strict=False)
                if "post_attention_layernorm" in ln_state:
                    layer.post_attention_layernorm.load_state_dict(ln_state["post_attention_layernorm"], strict=False)
    return expert, vision_expert


def _expert_inputs_for_batch(args, batch, device, text_tokenizer, vision_expert):
    if args.task_family == "population":
        images = batch["images"].to(device)
        b, n = images.shape[:2]
        flat = images.view(b * n, *images.shape[2:])
        with torch.no_grad():
            logits = vision_expert(flat).logits.float()
            probs = torch.softmax(logits, dim=-1).view(b, n, -1)
            mean_prob = probs.mean(dim=1)
        return {"mean_prob": mean_prob, "n_images": torch.full((b,), n, device=device)}
    if args.evidence_source == "vision":
        return batch["images"].to(device)
    if args.evidence_source == "text":
        enc = text_tokenizer(batch["description"], return_tensors="pt", padding=True, truncation=True, max_length=64)
        return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}
    raise ValueError(args.evidence_source)


def evaluate_plan1(args: EvalPlan1Args) -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenizer = _build_model(args)
    eval_loader = _build_eval_dataset(args, tokenizer)
    expert = None
    vision_expert = None
    if args.baseline == "injection":
        expert, vision_expert = _build_connectors(args, model)
    elif args.baseline in {"vision_only", "text_prompt"}:
        num_classes = _num_classes_for_dataset(args.dataset_name)
        vision_expert = build_vision_expert(
            expert_kind="classifier",
            checkpoint_path=args.expert_checkpoint,
            num_classes=num_classes,
            init_weights=args.expert_init_weights,
        )
        model.num_tokens = 0
    else:
        model.num_tokens = 0
    text_tokenizer = None
    if args.evidence_source == "text":
        text_tokenizer = EncoderTokenizer.from_pretrained(args.text_encoder_model_id, cache_dir=args.mistral_models_path)

    eval_ds = eval_loader.dataset
    class_names = getattr(eval_ds, "class_names", class_names_for_dataset(args.dataset_name))
    qa_mode = getattr(eval_ds, "effective_qa_type", args.qa_type)
    allowed_token_ids = _allowed_answer_token_ids(args, tokenizer, class_names)
    yes_token_ids, no_token_ids = _yes_no_token_sets(tokenizer) if qa_mode in {"yesno", "yesno_set2"} else (set(), set())
    if qa_mode == "label_code" and getattr(eval_ds, "code_to_token_id", None) is not None:
        allowed_token_ids = sorted(int(t) for t in eval_ds.code_to_token_id.values())
    if not model.is_quantized_4bit:
        model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)
    if vision_expert is not None:
        vision_expert = vision_expert.to(device)
    model.eval()

    total = 0
    correct = 0
    maes = []

    with torch.no_grad():
        for batch in eval_loader:
            tokens = batch["prompt_tokens"].to(device)
            mask = batch["prompt_mask"].to(device)

            if args.baseline == "vision_only":
                if args.task_family == "single_image":
                    logits = vision_expert(batch["images"].to(device)).logits.float()
                    pred_cls = logits.argmax(dim=-1).cpu().tolist()
                    if args.qa_type in {"yesno", "yesno_set2"}:
                        # Question format from CIFARSingleImageQADataset: "Is this a {class}? ..."
                        qtxts = batch["question_text"]
                        asked_name_sets = []
                        for q in qtxts:
                            ql = q.lower()
                            if "either " in ql and " or " in ql:
                                inside = ql.split("either ", 1)[1].split("?", 1)[0]
                                left, right = inside.split(" or ", 1)
                                asked_name_sets.append({left.strip(), right.strip()})
                            elif "is this a " in ql:
                                asked = ql.split("is this a ", 1)[1].split("?", 1)[0].strip()
                                asked_name_sets.append({asked})
                            else:
                                asked_name_sets.append(set())

                        cls_name_to_idx = {name.lower(): i for i, name in enumerate(class_names)}
                        pred_yesno = []
                        for p, asked_set in zip(pred_cls, asked_name_sets):
                            asked_indices = {cls_name_to_idx[a] for a in asked_set if a in cls_name_to_idx}
                            pred_yesno.append(1 if p in asked_indices else 0)
                        gt_yesno = [1 if a.lower().startswith("yes") else 0 for a in batch["answer_text"]]
                        correct += sum(int(a == b) for a, b in zip(pred_yesno, gt_yesno))
                        total += len(gt_yesno)
                    else:
                        gts = batch["labels"].cpu().tolist()
                        correct += sum(int(a == b) for a, b in zip(pred_cls, gts))
                        total += len(gts)
                else:
                    images = batch["images"].to(device)
                    b, n = images.shape[:2]
                    flat = images.view(b * n, *images.shape[2:])
                    logits = vision_expert(flat).logits.float().view(b, n, -1)
                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                    target_class = batch["target_class"].to(device)
                    frac = probs[torch.arange(b, device=device), target_class].cpu().numpy()
                    gt = batch["fraction"].cpu().numpy()
                    maes.extend(np.abs(frac - gt).tolist())
                    pred_bin = [population_fraction_to_bin(float(x), mode=args.population_output_mode) for x in frac]
                    gt_bin = [population_fraction_to_bin(float(x), mode=args.population_output_mode) for x in gt]
                    correct += sum(int(a == b) for a, b in zip(pred_bin, gt_bin))
                    total += b
                continue

            if args.baseline == "llm_only":
                outputs = model(tokens, attention_mask=mask, expert_inputs=None, use_cache=False)
            elif args.baseline == "text_prompt":
                prompt_tokens = []
                for i in range(tokens.size(0)):
                    if args.task_family == "single_image":
                        image = batch["images"][i : i + 1].to(device)
                        logits = vision_expert(image).logits.float()
                        pred_idx = int(logits.argmax(dim=-1).item())
                        hint = f"Expert says: {class_names[pred_idx]}."
                    else:
                        images = batch["images"][i].to(device)
                        logits = vision_expert(images).logits.float()
                        probs = torch.softmax(logits, dim=-1).mean(dim=0)
                        cls = int(batch["target_class"][i].item())
                        hint = f"Expert says P(class={class_names[cls]})={float(probs[cls]):.3f}."

                    question = batch["question_text"][i] if args.task_family == "single_image" else "Use the expert summary."
                    messages = [{"role": "system", "content": "Follow output format strictly."}, {"role": "user", "content": f"{question}\n{hint}"}]
                    t = _chat_template_to_ids(tokenizer, messages)
                    prompt_tokens.append(t)

                max_len = max(x.size(0) for x in prompt_tokens)
                padded = torch.full((len(prompt_tokens), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                pmask = torch.zeros((len(prompt_tokens), max_len), dtype=torch.long, device=device)
                for i, t in enumerate(prompt_tokens):
                    padded[i, max_len - t.size(0) :] = t.to(device)
                    pmask[i, max_len - t.size(0) :] = 1
                outputs = model(padded, attention_mask=pmask, expert_inputs=None, use_cache=False)
            else:
                expert_inputs = _expert_inputs_for_batch(args, batch, device, text_tokenizer, vision_expert)
                outputs = model(tokens, attention_mask=mask, expert_inputs=(expert_inputs,), use_cache=False)

            next_logits = outputs.logits[:, -1, :]
            if args.baseline in {"llm_only", "text_prompt", "injection"} and args.constrain_llm_only_outputs and allowed_token_ids:
                allowed = torch.tensor(allowed_token_ids, device=next_logits.device, dtype=torch.long)
                pred_local = next_logits.index_select(dim=-1, index=allowed).argmax(dim=-1)
                pred_ids = allowed[pred_local].cpu().tolist()
            else:
                pred_ids = next_logits.argmax(dim=-1).cpu().tolist()
            pred_texts = [tokenizer.decode([pid]).strip() for pid in pred_ids]

            if args.task_family == "single_image":
                if qa_mode in {"yesno", "yesno_set2"}:
                    gt = [1 if a.lower().startswith("yes") else 0 for a in batch["answer_text"]]
                    for p, g in zip(pred_ids, gt):
                        if int(p) in yes_token_ids:
                            pred_bin = 1
                        elif int(p) in no_token_ids:
                            pred_bin = 0
                        else:
                            pred_bin = -1
                        if pred_bin == int(g):
                            correct += 1
                        total += 1
                    continue
                elif qa_mode == "label_code":
                    gt_ids = batch["target_token_id"].cpu().tolist()
                    for p, g in zip(pred_ids, gt_ids):
                        if int(p) == int(g):
                            correct += 1
                        total += 1
                    continue
                else:
                    parsed_pred = [parse_label_answer(x, class_names) for x in pred_texts]
                    if qa_mode == "index":
                        gt = [int(x) for x in batch["answer_text"]]
                    else:
                        gt = [parse_label_answer(x, class_names) for x in batch["answer_text"]]
                for p, g in zip(parsed_pred, gt):
                    if p is not None and int(p) == int(g):
                        correct += 1
                    total += 1
            else:
                pred_bin = [parse_population_answer(x, mode=args.population_output_mode) for x in pred_texts]
                gt_bin = [parse_population_answer(x, mode=args.population_output_mode) for x in batch["answer_text"]]
                for p, g in zip(pred_bin, gt_bin):
                    if p is not None and g is not None:
                        pred_frac = population_bin_to_fraction_midpoint(p, mode=args.population_output_mode)
                        gt_frac = population_bin_to_fraction_midpoint(g, mode=args.population_output_mode)
                        maes.append(abs(pred_frac - gt_frac))
                        correct += int(int(p) == int(g))
                    total += 1

    if args.task_family == "single_image":
        return {"accuracy": correct / max(1, total)}
    return {"bin_accuracy": correct / max(1, total), "mae": float(np.mean(maes) if maes else 0.0)}
