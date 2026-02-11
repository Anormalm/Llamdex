"""
Training script for Llamdex-IMG (image classification extension).

Usage:
    python scripts/train_img.py --mistral_models_path model/llm --model_name mistralai/Mistral-7B-Instruct-v0.3
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.vision.vision_domain_expert import VisionDomainExpert
from src.vision.cifar_dataset import CIFAR10Dataset, get_cifar10_collate_fn


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_img(
    mistral_models_path: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    save_dir: str = "model/llm/llamdex_img_cifar10",
    num_tokens: int = 10,
    layer_to_add: int = 0,
    vision_model: str = "resnet18",
    num_classes: int = 10,
    vision_output: str = "logits",  # 'logits' or 'embedding'
    ffn_hidden_size: int = 2048,
    alpha: float = 1.0,
    dropout: float = 0.0,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    save_period: int = 1000,
    data_root: str = "./data/cifar10",
    seed: int = 42,
):
    """Train Llamdex-IMG on CIFAR-10."""
    
    set_seed(seed)
    
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")
    
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=f"runs/llamdex_img_cifar10")
        print("=" * 80)
        print("Llamdex-IMG Training Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Vision Model: {vision_model}")
        print(f"  Vision Output: {vision_output}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Num Tokens: {num_tokens}")
        print(f"  Layer: {layer_to_add}")
        print(f"  Alpha: {alpha}")
        print(f"  FFN Hidden Size: {ffn_hidden_size}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Num Epochs: {num_epochs}")
        print(f"  Device: {accelerator.device}")
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
        alpha=alpha,
        dropout=dropout,
    )
    
    # Add expert to specified layer
    base_model = model.model
    base_model.layers[layer_to_add].add_expert_(vision_expert, map_to_expert_emb=None)
    
    # Setup decoder as trainable
    model.setup_decoder(layer_to_add, 0, True)
    
    # Setup encoder as frozen (vision expert is always frozen)
    model.setup_encoder(layer_to_add, 0, False)
    
    model = model.to(torch.bfloat16)
    
    # Print trainable parameters
    if accelerator.is_local_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load datasets
    train_dataset = CIFAR10Dataset(
        root=data_root,
        train=True,
        tokenizer=tokenizer,
    )
    
    test_dataset = CIFAR10Dataset(
        root=data_root,
        train=False,
        tokenizer=tokenizer,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // gradient_accumulation_steps,
        shuffle=True,
        collate_fn=get_cifar10_collate_fn(tokenizer.pad_token_id),
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size // gradient_accumulation_steps,
        shuffle=False,
        collate_fn=get_cifar10_collate_fn(tokenizer.pad_token_id),
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_dataloader),
    )
    
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get digit token IDs for evaluation
    digit_token_ids = train_dataset.digit_token_ids
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for index, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    
                    images = batch["images"].to(model.device)  # (B, 3, 32, 32)
                    tokens = batch["tokens"].to(model.device)
                    attn_mask = batch["attention_mask"].to(model.device)
                    target_token_ids = batch["target_token_ids"].to(model.device)  # (B,)
                    
                    # Forward pass: images -> vision expert -> decoder -> injection
                    outputs = model(
                        tokens,
                        expert_inputs=(images,),  # Pass images as expert_inputs
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
                    
                    logits = outputs.logits  # (B, seq_len, vocab_size)
                    
                    # Get logits at the last position (first generation token)
                    last_logits = logits[:, -1, :]  # (B, vocab_size)
                    
                    # Compute loss on target token IDs
                    loss = nn.CrossEntropyLoss()(last_logits, target_token_ids)
                    
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    
                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_epoch_{epoch + 1}_step_{index + 1}.pt")
                    
                    if accelerator.is_local_main_process:
                        writer.add_scalar('Loss/train', loss.item(), global_step)
            
            global_step += 1
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, disable=not accelerator.is_main_process):
                with accelerator.autocast():
                    images = batch["images"].to(model.device)
                    tokens = batch["tokens"].to(model.device)
                    attn_mask = batch["attention_mask"].to(model.device)
                    target_token_ids = batch["target_token_ids"].to(model.device)
                    labels = batch["labels"].to(model.device)
                    
                    outputs = model(
                        tokens,
                        expert_inputs=(images,),
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
                    
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    
                    loss = nn.CrossEntropyLoss()(last_logits, target_token_ids)
                    test_loss += loss.item()
                    
                    # Get predictions: argmax over digit token IDs
                    digit_logits = last_logits[:, digit_token_ids]  # (B, 10)
                    preds = digit_logits.argmax(dim=-1)  # (B,)
                    
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
        
        test_acc = correct / total if total > 0 else 0.0
        avg_train_loss = train_loss / len(train_dataloader)
        avg_test_loss = test_loss / len(test_dataloader)
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Test Loss: {avg_test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f} ({correct}/{total})")
            writer.add_scalar('Loss/test', avg_test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            
            # Save checkpoint
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_epoch_{epoch + 1}.pt")
    
    if accelerator.is_local_main_process:
        # Save final model
        state = accelerator.get_state_dict(model)
        accelerator.save(state, f"{save_dir}/model_final.pt")
        writer.close()
        print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Llamdex-IMG on CIFAR-10")
    
    parser.add_argument("--mistral_models_path", type=str, default="model/llm",
                       help="Path to Mistral model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Mistral model name")
    parser.add_argument("--save_dir", type=str, default="model/llm/llamdex_img_cifar10",
                       help="Directory to save checkpoints")
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--save_period", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--data_root", type=str, default="./data/cifar10",
                       help="Root directory for CIFAR-10 data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    train_img(
        mistral_models_path=args.mistral_models_path,
        model_name=args.model_name,
        save_dir=args.save_dir,
        num_tokens=args.num_tokens,
        layer_to_add=args.layer,
        vision_model=args.vision_model,
        num_classes=args.num_classes,
        vision_output=args.vision_output,
        ffn_hidden_size=args.ffn_hidden_size,
        alpha=args.alpha,
        dropout=args.dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_period=args.save_period,
        data_root=args.data_root,
        seed=args.seed,
    )
