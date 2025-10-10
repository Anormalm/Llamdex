import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import random
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import sys
from torch.nn import KLDivLoss
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data using SeqPATE')

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
    parser.add_argument('--epsilon', type=float, nargs='+', required=True, help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=str, default='auto', help='Privacy parameter delta. Use "auto" for 1/n, or specify a float value')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to generate')
    parser.add_argument('--n_teachers', type=int, default=10, help='Number of teacher models')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--n_instances', type=int, default=None, help='Number of instances to use')
    parser.add_argument("--top_k", type=int, default=200, help="Top-k filtering parameter")
    parser.add_argument('--lambda_teacher', type=float, default=20.0, help='Weight for teacher supervision loss')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def get_delta(args, n):
    if args.delta == 'auto':
        return 1/n
    return float(args.delta)

def format_number(value):
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip('0').rstrip('.')
    return str(value)

def create_instance_prompt(row, target_column):
    # Construct a prompt from the row excluding the target column
    prompt = ""
    for col in row.index:
        value = format_number(row[col])
        if col != target_column:
            prompt += f"{col}: {value}\n"
    prompt += f"{target_column}:"
    return prompt

class TabularDataset(Dataset):
    def __init__(self, data, tokenizer, config, target_column):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.target_column = target_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = create_instance_prompt(row, self.target_column)
        # The answer is the label after the prompt
        answer = format_number(row[self.target_column])
        prompt_with_answer = prompt + f" {answer}"

        # Tokenize the prompt with the answer
        inputs = self.tokenizer(
            prompt_with_answer,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=160,
            add_special_tokens=True,
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Create labels: ignore all tokens except the answer token
        labels = torch.full_like(input_ids, -100)
        # Find the position of the answer token
        answer_position = len(input_ids) - 1
        labels[answer_position] = input_ids[answer_position]
        # print(self.tokenizer.decode(input_ids[answer_position]))
        # print("Number of pad tokens:", torch.sum(input_ids == self.tokenizer.pad_token_id))
        # print(input_ids, answer_token)
        # print([self.tokenizer.decode([x]) for x in input_ids])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def aggregate_teacher_distributions(teacher_outputs, sigma):
    mean_distribution = np.mean(teacher_outputs, axis=1)
    noisy_distribution = mean_distribution + np.random.normal(0, sigma, size=mean_distribution.shape)
    noisy_distribution = np.clip(noisy_distribution, 0, None)
    noisy_distribution /= np.sum(noisy_distribution, axis=-1, keepdims=True)
    return noisy_distribution


def train_teacher_models(args, config, tokenizer, train_df, target_column, pseudo_data):
    """Train teacher models one at a time, save predictions and LoRA parameters."""
    num_classes = len(train_df[target_column].unique())
    subset_size = len(train_df) // args.n_teachers
    teacher_outputs_list = []

    for i in range(args.n_teachers):
        print(f"Training teacher model {i+1}/{args.n_teachers}")
        # Get the subset for this teacher
        if i < args.n_teachers - 1:
            subset = train_df.iloc[i*subset_size:(i+1)*subset_size]
        else:
            subset = train_df.iloc[i*subset_size:]

        # Create dataset and dataloader
        dataset = TabularDataset(subset, tokenizer, config, target_column)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize base model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.mistral_models_path
        )

        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer._pad_token is None:
            tokenizer._pad_token = tokenizer.unk_token

        # Set up LoRA configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(torch.bfloat16)

        os.makedirs(f"model/llm/teacher_models_seed{args.seed}/teacher_{i}", exist_ok=True)
        # Setup training arguments, no need to save the model
        # Disable wandb
        training_args = TrainingArguments(
            output_dir=f"model/llm/teacher_models_seed{args.seed}/teacher_{i}",
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logging_steps=100,
            optim="adamw_torch",
            save_strategy="no",
            save_total_limit=0,
            bf16=True,
            report_to=[],
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # Train the model
        trainer.train()

        # Save only the LoRA parameters
        peft_save_path = f"model/llm/teacher_models_seed{args.seed}/teacher_{i}_lora"
        os.makedirs(peft_save_path, exist_ok=True)
        model.save_pretrained(peft_save_path)
        tokenizer.save_pretrained(peft_save_path)

        # Evaluate on pseudo data and save outputs
        pseudo_dataset = TabularDataset(pseudo_data, tokenizer, config, target_column)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=False)

        model.eval()
        teacher_outputs = []

        with torch.no_grad():
            for batch in tqdm(pseudo_loader, desc=f"Teacher {i+1} inference"):
                batch = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

                # The answer token is at the last position
                answer_position = batch['input_ids'].size(1) - 1
                teacher_logits = logits[:, answer_position, :]  # Shape: [batch_size, vocab_size]
                probs = torch.softmax(teacher_logits, dim=-1)
                teacher_outputs.append(probs.cpu().numpy())

        # Concatenate the outputs
        teacher_outputs = np.concatenate(teacher_outputs, axis=0)  # Shape: [num_samples, vocab_size]
        outputs_save_path = f"model/llm/teacher_models_seed{args.seed}/teacher_{i}_outputs.npz"
        np.savez(outputs_save_path, outputs=teacher_outputs)

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    return

def load_teacher_outputs(args):
    """Load teacher outputs from saved files."""
    all_teacher_outputs = []
    for i in range(args.n_teachers):
        outputs_save_path = f"model/llm/teacher_models_seed{args.seed}/teacher_{i}_outputs.npz"
        data = np.load(outputs_save_path)
        all_teacher_outputs.append(data['outputs'])

    # Stack teacher outputs
    teacher_outputs = np.stack(all_teacher_outputs, axis=1)  # Shape: [num_samples, n_teachers, vocab_size]
    return teacher_outputs

def train_student_model(student_model, train_dataloader, teacher_outputs, args, device):
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    student_model.to(device)

    for epoch in range(args.num_epochs):
        student_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_idx = step * args.batch_size
            end_idx = min(batch_idx + args.batch_size, len(teacher_outputs))
            batch_teacher_probs = torch.tensor(teacher_outputs[batch_idx:end_idx]).to(device)

            optimizer.zero_grad()
            outputs = student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            # The answer token is at the last position
            answer_position = batch['input_ids'].size(1) - 1
            student_logits = logits[:, answer_position, :]  # Shape: [batch_size, vocab_size]
            student_log_probs = torch.log_softmax(student_logits, dim=-1)

            teacher_probs = batch_teacher_probs  # Shape: [batch_size, vocab_size]
            # Use top-k tokens
            topk = torch.topk(teacher_probs, k=min(teacher_probs.size(-1), args.top_k), dim=-1)
            teacher_probs_topk = torch.zeros_like(teacher_probs)
            teacher_probs_topk.scatter_(1, topk.indices, topk.values)
            teacher_probs_topk /= teacher_probs_topk.sum(dim=-1, keepdim=True)  # Normalize

            # Compute loss
            kl_loss = KLDivLoss(reduction="batchmean")
            loss = kl_loss(student_log_probs, teacher_probs_topk) * args.lambda_teacher

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})

        print(f'Epoch {epoch+1} average loss: {total_loss / len(train_dataloader)}')

def generate_student_model(train_df, pseudo_df, config, args, num_classes, epsilon):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer for student model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.mistral_models_path,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher outputs
    print("Loading teacher outputs...")
    teacher_outputs = load_teacher_outputs(args)

    # Compute sigma for differential privacy
    delta = get_delta(args, len(train_df))
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # Aggregate teacher outputs
    print(f"Aggregating teacher outputs with epsilon {epsilon} and sigma {sigma}")
    aggregated_outputs = aggregate_teacher_distributions(teacher_outputs, sigma)  # Shape: [num_samples, vocab_size]

    # Initialize and prepare student model with LoRA
    print("Initializing student model with LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.mistral_models_path
    )

    student_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(student_model, peft_config)
    student_model = student_model.to(device)
    student_model = student_model.to(torch.bfloat16)

    # Prepare dataset and dataloader
    train_dataset = TabularDataset(pseudo_df, tokenizer, config, config['y']['name'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the student model
    print("Training student model with teacher supervision...")
    train_student_model(student_model, train_dataloader, aggregated_outputs, args, device)

    # Save the whole state dict of the student model
    output_dir = f"model/llm/exp-lora-syn_mistral_{args.dataset}_seqpate_{epsilon}_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    # state = student_model.state_dict()
    # torch.save(state, os.path.join(output_dir, "model_final.pt"))
    student_model.save_pretrained(output_dir)
    print(f"Student model saved to {output_dir}")

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(f"model/llm/teacher_models_seed{args.seed}", exist_ok=True)

    # Load configuration and data
    print("Loading configuration and data...")
    with open(f"src/dataset/{args.dataset}/{args.dataset}.json", 'r') as f:
        config = json.load(f)

    # Determine number of classes
    target_column = config['y']['name']
    num_classes = len(config['y']['categories']) if config['y']['type'] == "categorical" else None

    # Load train and pseudo datasets
    train_df = pd.read_csv(f"data/{args.dataset}/clean/{args.dataset}_train_onehot.csv")
    pseudo_df = pd.read_csv(f"data/{args.dataset}/clean/syn_{args.dataset}_train_onehot.csv")

    # Apply format_number to all items
    train_df = train_df.applymap(format_number)
    pseudo_df = pseudo_df.applymap(format_number)

    if args.n_instances is not None:
        train_df = train_df.sample(n=args.n_instances)
        pseudo_df = pseudo_df.sample(n=args.n_instances)
    else:
        # Sample same number of instances as real training data
        if len(pseudo_df) > len(train_df):
            pseudo_df = pseudo_df.sample(n=len(train_df))

    print(f"Train dataset size: {len(train_df)}")
    print(f"Pseudo dataset size: {len(pseudo_df)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.mistral_models_path,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training teacher models one by one and saving their outputs
    print("Training teacher models...")
    train_teacher_models(args, config, tokenizer, train_df, target_column, pseudo_df)

    for epsilon in args.epsilon:
        print(f"\nProcessing epsilon = {epsilon}")
        generate_student_model(train_df, pseudo_df, config, args, num_classes, epsilon)

if __name__ == "__main__":
    main()
