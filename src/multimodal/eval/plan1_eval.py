from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoTokenizer as EncoderTokenizer

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.evidence import PopulationStatsEvidenceBuilder, TextEvidenceBuilder, VisionEvidenceBuilder
from src.multimodal.experts import build_vision_expert
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.tasks.parsing import (
    parse_label_answer,
    parse_population_answer,
    parse_yes_no_answer,
    population_bin_to_fraction_midpoint,
)
from src.multimodal.tasks.prompts import class_names_for_dataset


@dataclass
class EvalPlan1Args:
    mistral_models_path: str = "model/llm"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    connectors_path: Optional[str] = None
    dataset_name: str = "cifar10"
    data_root: str = "./data/cifar10"
    task_family: str = "single_image"
    evidence_source: str = "vision"
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


def _build_model(args: EvalPlan1Args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.mistral_models_path,
        torch_dtype=torch.bfloat16,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        args.model_name,
        cache_dir=args.mistral_models_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        tokenizer=tokenizer,
    )
    model.num_tokens = args.num_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def _build_eval_dataset(args: EvalPlan1Args, tokenizer):
    if args.task_family == "single_image":
        ds = CIFARSingleImageQADataset(
            root=args.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=args.dataset_name,
            mode=args.evidence_source if args.evidence_source in {"vision", "text"} else "vision",
            qa_type=args.qa_type,
            max_samples=args.max_eval_samples,
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
    num_classes = 10 if args.dataset_name == "cifar10" else 100
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
    if args.connectors_path:
        state = torch.load(args.connectors_path, map_location="cpu")
        expert.evidence_builder.load_state_dict(state["evidence_builder"], strict=False)
        expert.projector.load_state_dict(state["projector"], strict=False)
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
        num_classes = 10 if args.dataset_name == "cifar10" else 100
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

    class_names = class_names_for_dataset(args.dataset_name)
    model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)
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
                    preds = logits.argmax(dim=-1).cpu().tolist()
                    gts = batch["labels"].cpu().tolist()
                    correct += sum(int(a == b) for a, b in zip(preds, gts))
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
                    t = tokenizer.apply_chat_template(messages, return_tensors="pt").squeeze(0)
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

            pred_ids = outputs.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            pred_texts = [tokenizer.decode([pid]).strip() for pid in pred_ids]

            if args.task_family == "single_image":
                if args.qa_type == "yesno":
                    parsed_pred = [parse_yes_no_answer(x) for x in pred_texts]
                    gt = [1 if a.lower().startswith("yes") else 0 for a in batch["answer_text"]]
                else:
                    parsed_pred = [parse_label_answer(x, class_names) for x in pred_texts]
                    if args.qa_type == "index":
                        gt = [int(x) for x in batch["answer_text"]]
                    else:
                        gt = [class_names.index(x) for x in batch["answer_text"]]
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
