from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.evidence import (
    DiffusionFutureEvidenceBuilder,
    PopulationStatsEvidenceBuilder,
    TextEvidenceBuilder,
    VisionEvidenceBuilder,
)
from src.multimodal.experts import build_vision_expert
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.utils.logging_utils import MetricLogger
from src.multimodal.utils.repro import save_run_config, save_versions, set_seed
from transformers import AutoTokenizer as EncoderTokenizer


@dataclass
class TrainPlan1Args:
    mistral_models_path: str = "model/llm"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    run_dir: str = "runs/plan1_default"
    dataset_name: str = "cifar10"
    data_root: str = "./data/cifar10"
    task_family: str = "single_image"
    evidence_source: str = "vision"  # vision|text|diffusion_future
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
    seed: int = 42
    device: str = "cuda"


def _build_model_and_tokenizer(args: TrainPlan1Args):
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


def _attach_semantic_expert(args: TrainPlan1Args, model, tokenizer):
    hidden_size = model.config.hidden_size
    num_classes = 10 if args.dataset_name == "cifar10" else 100

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
            cache_dir=args.mistral_models_path,
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
    semantic_expert = SemanticEvidenceDomainExpert(evidence_builder=evidence_builder, projector=projector)
    model.model.layers[args.layer_to_add].add_expert_(semantic_expert, map_to_expert_emb=None)
    return model, semantic_expert, vision_expert


def _build_dataloaders(args: TrainPlan1Args, tokenizer):
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


@torch.no_grad()
def _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert):
    model.eval()
    correct = 0
    total = 0
    for batch in eval_loader:
        tokens = batch["prompt_tokens"].to(device)
        mask = batch["prompt_mask"].to(device)
        targets = batch["target_token_id"].to(device)
        expert_inputs = _build_expert_inputs(args, batch, device, text_tokenizer, semantic_expert, vision_expert)
        outputs = model(tokens, attention_mask=mask, expert_inputs=expert_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return correct / max(1, total)


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
        text_tokenizer = EncoderTokenizer.from_pretrained(args.text_encoder_model_id, cache_dir=args.mistral_models_path)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device).to(torch.bfloat16 if device.type == "cuda" else torch.float32)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    best_eval = -1.0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            tokens = batch["prompt_tokens"].to(device)
            mask = batch["prompt_mask"].to(device)
            targets = batch["target_token_id"].to(device)

            expert_inputs = _build_expert_inputs(args, batch, device, text_tokenizer, semantic_expert, vision_expert)
            outputs = model(tokens, attention_mask=mask, expert_inputs=expert_inputs, use_cache=False)
            logits = outputs.logits[:, -1, :]
            loss = loss_fn(logits, targets)

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
            }
            logger.log(row)
            global_step += 1

            if global_step % args.eval_every_steps == 0:
                eval_acc = _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert)
                logger.log({"split": "eval", "epoch": epoch, "step": global_step, "acc": float(eval_acc)})
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    torch.save(
                        {
                            "evidence_builder": semantic_expert.evidence_builder.state_dict(),
                            "projector": semantic_expert.projector.state_dict(),
                            "args": asdict(args),
                        },
                        os.path.join(args.run_dir, "best_connectors.pt"),
                    )

    final_eval = _evaluate(model, args, eval_loader, device, text_tokenizer, semantic_expert, vision_expert)
    logger.log({"split": "eval", "epoch": args.num_epochs, "step": global_step, "acc": float(final_eval)})
    torch.save(
        {
            "evidence_builder": semantic_expert.evidence_builder.state_dict(),
            "projector": semantic_expert.projector.state_dict(),
            "args": asdict(args),
        },
        os.path.join(args.run_dir, "last_connectors.pt"),
    )
    return {"final_eval_acc": final_eval, "best_eval_acc": best_eval}
