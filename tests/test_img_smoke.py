"""
Smoke test for Llamdex-IMG: run a few training steps without crashing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.vision.vision_domain_expert import VisionDomainExpert
from src.vision.cifar_dataset import CIFAR10Dataset, get_cifar10_collate_fn


def test_smoke():
    """Run a smoke test: 10 training steps on a tiny subset."""
    print("=" * 80)
    print("Llamdex-IMG Smoke Test")
    print("=" * 80)
    
    # Use CPU for smoke test (faster, no GPU required)
    accelerator = Accelerator(cpu=True)
    
    mistral_models_path = "model/llm"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=mistral_models_path,
        )
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Skipping smoke test (requires model download)")
        return
    
    # Load model
    try:
        model = DomainMistralForCausalLM.from_pretrained_mistral(
            model_name,
            cache_dir=mistral_models_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            tokenizer=tokenizer,
            llamdex_padding='gaussian',
        )
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Skipping smoke test (requires model download)")
        return
    
    model.num_tokens = 4  # Small number for smoke test
    embed_size = model.config.hidden_size
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Create vision expert
    vision_expert = VisionDomainExpert(
        embed_size=embed_size,
        num_tokens=4,
        vision_model="resnet18",
        num_classes=10,
        vision_output="logits",
        ffn_hidden_size=512,  # Smaller for smoke test
        alpha=1.0,
    )
    
    # Add expert to layer 0
    base_model = model.model
    base_model.layers[0].add_expert_(vision_expert, map_to_expert_emb=None)
    
    # Setup decoder as trainable
    model.setup_decoder(0, 0, True)
    
    # Load tiny subset of CIFAR-10
    try:
        dataset = CIFAR10Dataset(
            root="./data/cifar10",
            train=True,
            tokenizer=tokenizer,
        )
        
        # Create a tiny subset (first 20 samples)
        from torch.utils.data import Subset
        subset = Subset(dataset, list(range(min(20, len(dataset)))))
        
        dataloader = DataLoader(
            subset,
            batch_size=2,  # Small batch
            shuffle=False,
            collate_fn=get_cifar10_collate_fn(tokenizer.pad_token_id),
        )
    except Exception as e:
        print(f"Warning: Could not load CIFAR-10 dataset: {e}")
        print("Skipping smoke test (requires CIFAR-10 download)")
        return
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Get digit token IDs
    digit_token_ids = dataset.digit_token_ids
    
    print("\nRunning 10 training steps...")
    model.train()
    
    step = 0
    for batch in dataloader:
        if step >= 10:
            break
        
        optimizer.zero_grad()
        
        images = batch["images"].to(model.device)
        tokens = batch["tokens"].to(model.device)
        attn_mask = batch["attention_mask"].to(model.device)
        target_token_ids = batch["target_token_ids"].to(model.device)
        
        # Forward pass
        outputs = model(
            tokens,
            expert_inputs=(images,),
            attention_mask=attn_mask,
            use_cache=False,
        )
        
        logits = outputs.logits
        last_logits = logits[:, -1, :]
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(last_logits, target_token_ids)
        
        # Backward
        accelerator.backward(loss)
        optimizer.step()
        
        # Get predictions
        digit_logits = last_logits[:, digit_token_ids]
        preds = digit_logits.argmax(dim=-1)
        labels = batch["labels"].to(model.device)
        correct = (preds == labels).sum().item()
        acc = correct / labels.size(0)
        
        print(f"  Step {step + 1}/10: Loss={loss.item():.4f}, Acc={acc:.4f}")
        
        step += 1
    
    # Evaluation
    print("\nRunning evaluation...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(model.device)
            tokens = batch["tokens"].to(model.device)
            attn_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            outputs = model(
                tokens,
                expert_inputs=(images,),
                attention_mask=attn_mask,
                use_cache=False,
            )
            
            logits = outputs.logits
            last_logits = logits[:, -1, :]
            digit_logits = last_logits[:, digit_token_ids]
            preds = digit_logits.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nEvaluation Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\n" + "=" * 80)
    print("Smoke test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    test_smoke()
