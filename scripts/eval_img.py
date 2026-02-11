"""
Evaluation script for Llamdex-IMG with baselines.

Baselines:
1. Vision-only: Expert argmax (logits) for accuracy
2. LLM-only: Disable injection (alpha=0 or bypass)
3. Prompt baseline: Run vision-only first, then insert prediction into prompt

Usage:
    python scripts/eval_img.py --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.vision.vision_domain_expert import VisionDomainExpert
from src.vision.cifar_dataset import CIFAR10Dataset, get_cifar10_collate_fn


def evaluate_img(
    mistral_models_path: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    model_state_dict: str = None,
    num_tokens: int = 10,
    layer_to_add: int = 0,
    vision_model: str = "resnet18",
    num_classes: int = 10,
    vision_output: str = "logits",
    ffn_hidden_size: int = 2048,
    alpha: float = 1.0,
    dropout: float = 0.0,
    batch_size: int = 32,
    data_root: str = "./data/cifar10",
    baseline: str = None,  # 'vision_only', 'llm_only', 'prompt'
    seed: int = 42,
):
    """Evaluate Llamdex-IMG on CIFAR-10."""
    
    accelerator = Accelerator(mixed_precision="bf16")
    
    if accelerator.is_local_main_process:
        print("=" * 80)
        print("Llamdex-IMG Evaluation Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Vision Model: {vision_model}")
        print(f"  Vision Output: {vision_output}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Num Tokens: {num_tokens}")
        print(f"  Layer: {layer_to_add}")
        print(f"  Alpha: {alpha}")
        print(f"  Baseline: {baseline if baseline else 'Llamdex-IMG'}")
        print("=" * 80)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=mistral_models_path,
        torch_dtype=torch.bfloat16
    )
    
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        model_name,
        cache_dir=mistral_models_path,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        llamdex_padding='gaussian',
    )
    
    model.num_tokens = num_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token
    
    embed_size = model.config.hidden_size
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Create vision expert
    vision_expert = VisionDomainExpert(
        embed_size=embed_size,
        num_tokens=num_tokens,
        vision_model=vision_model,
        num_classes=num_classes,
        vision_output=vision_output,
        ffn_hidden_size=ffn_hidden_size,
        alpha=alpha if baseline != 'llm_only' else 0.0,  # Disable injection for LLM-only baseline
        dropout=dropout,
    )
    
    # Add expert to specified layer
    base_model = model.model
    base_model.layers[layer_to_add].add_expert_(vision_expert, map_to_expert_emb=None)
    
    # Load model state dict if provided
    if model_state_dict is not None and baseline != 'vision_only' and baseline != 'prompt':
        if accelerator.is_local_main_process:
            print(f"Loading model state dict from {model_state_dict}")
        state = torch.load(model_state_dict, map_location='cpu')
        model.load_state_dict(state, strict=False)
    
    model = model.to(torch.bfloat16)
    
    # Load test dataset
    test_dataset = CIFAR10Dataset(
        root=data_root,
        train=False,
        tokenizer=tokenizer,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_cifar10_collate_fn(tokenizer.pad_token_id),
    )
    
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Get digit token IDs
    digit_token_ids = test_dataset.digit_token_ids
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, disable=not accelerator.is_main_process):
            images = batch["images"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            if baseline == 'vision_only':
                # Baseline 1: Vision-only classifier
                vision_expert.vision_expert.eval()
                with torch.no_grad():
                    vision_output = vision_expert.vision_expert(images)
                    if vision_output.dim() == 2 and vision_output.size(1) == num_classes:
                        # Logits mode
                        preds = vision_output.argmax(dim=-1)
                    else:
                        # Embedding mode - need to use classifier
                        # For embedding mode, we need to get logits from the classifier
                        if hasattr(vision_expert.vision_expert, 'classifier') and vision_expert.vision_expert.classifier is not None:
                            logits = vision_expert.vision_expert.classifier(vision_output)
                            preds = logits.argmax(dim=-1)
                        else:
                            # Fallback: random predictions (should not happen with proper setup)
                            preds = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
                            print("Warning: Vision-only baseline with embedding mode: classifier not available")
            
            elif baseline == 'prompt':
                # Baseline 3: Prompt baseline
                # Run vision expert first
                vision_expert.vision_expert.eval()
                with torch.no_grad():
                    vision_output = vision_expert.vision_expert(images)
                    if vision_output.dim() == 2 and vision_output.size(1) == num_classes:
                        vision_preds = vision_output.argmax(dim=-1).cpu().numpy()
                    else:
                        # Embedding mode - use classifier if available
                        if hasattr(vision_expert.vision_expert, 'classifier') and vision_expert.vision_expert.classifier is not None:
                            logits = vision_expert.vision_expert.classifier(vision_output)
                            vision_preds = logits.argmax(dim=-1).cpu().numpy()
                        else:
                            vision_preds = np.zeros(images.size(0), dtype=np.int64)
                
                # Create prompts with vision predictions
                preds = []
                for i, vision_pred in enumerate(vision_preds):
                    prompt_text = f"The classifier predicts: {vision_pred}."
                    messages = [
                        {"role": "system", "content": "You are a classifier. Answer with only one token: a digit 0-9."},
                        {"role": "user", "content": prompt_text}
                    ]
                    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
                    attn_mask = (tokens != tokenizer.pad_token_id).long()
                    
                    # Run LLM without injection
                    outputs = model(tokens, attention_mask=attn_mask, use_cache=False)
                    logits = outputs.logits[:, -1, :]
                    digit_logits = logits[:, digit_token_ids]
                    pred = digit_logits.argmax(dim=-1).item()
                    preds.append(pred)
                
                preds = torch.tensor(preds, dtype=torch.long, device=model.device)
            
            else:
                # Llamdex-IMG or LLM-only baseline
                tokens = batch["tokens"].to(model.device)
                attn_mask = batch["attention_mask"].to(model.device)
                
                if baseline == 'llm_only':
                    # Baseline 2: LLM-only (no injection, alpha=0 already set)
                    expert_inputs = None
                else:
                    # Llamdex-IMG: use injection
                    expert_inputs = (images,)
                
                outputs = model(
                    tokens,
                    expert_inputs=expert_inputs,
                    attention_mask=attn_mask,
                    use_cache=False,
                )
                
                logits = outputs.logits
                last_logits = logits[:, -1, :]
                
                # Get predictions: argmax over digit token IDs
                digit_logits = last_logits[:, digit_token_ids]
                preds = digit_logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    if accelerator.is_local_main_process:
        print(f"\n{'=' * 80}")
        print(f"Results ({baseline if baseline else 'Llamdex-IMG'}):")
        print(f"  Accuracy: {accuracy:.4f} ({np.sum(np.array(all_preds) == np.array(all_labels))}/{len(all_labels)})")
        print(f"{'=' * 80}")
    
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Llamdex-IMG on CIFAR-10")
    
    parser.add_argument("--mistral_models_path", type=str, default="model/llm",
                       help="Path to Mistral model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Mistral model name")
    parser.add_argument("--model_state_dict", type=str, default=None,
                       help="Path to model state dict (for Llamdex-IMG)")
    parser.add_argument("--num_tokens", type=int, default=10,
                       help="Number of tokens to inject")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer to inject expert (0-indexed)")
    parser.add_argument("--vision_model", type=str, default="resnet18",
                       help="Vision model name")
    parser.add_argument("--num_classes", type=int, default=10,
                       help="Number of classes")
    parser.add_argument("--vision_output", type=str, default="logits", choices=["logits", "embedding"],
                       help="Vision expert output mode")
    parser.add_argument("--ffn_hidden_size", type=int, default=2048,
                       help="FFN hidden size (-1 for Linear)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Alpha scaling factor")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--data_root", type=str, default="./data/cifar10",
                       help="Root directory for CIFAR-10 data")
    parser.add_argument("--baseline", type=str, default=None,
                       choices=[None, 'vision_only', 'llm_only', 'prompt'],
                       help="Baseline to evaluate (None for Llamdex-IMG)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    evaluate_img(
        mistral_models_path=args.mistral_models_path,
        model_name=args.model_name,
        model_state_dict=args.model_state_dict,
        num_tokens=args.num_tokens,
        layer_to_add=args.layer,
        vision_model=args.vision_model,
        num_classes=args.num_classes,
        vision_output=args.vision_output,
        ffn_hidden_size=args.ffn_hidden_size,
        alpha=args.alpha,
        dropout=args.dropout,
        batch_size=args.batch_size,
        data_root=args.data_root,
        baseline=args.baseline,
        seed=args.seed,
    )
