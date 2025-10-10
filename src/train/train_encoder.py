import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm
import argparse
from transformers import get_constant_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn, get_dataset

def train_encoder(models_path, model_name, dataset_type, save_dir, batch_size, gradient_accumulation_steps,
                   num_epochs, save_period, learning_rate, warmup_steps, no_checkpoint):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if accelerator.is_local_main_process:
        wandb.init(project="SiT", name="train_encoder")
        wandb.config.update({
            "models_path": models_path,
            "model_name": model_name,
            "dataset_type": dataset_type,
            "save_dir": save_dir,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "save_period": save_period,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "no_checkpoint": no_checkpoint
        })

    if save_dir is None:
        save_dir = f"model/llm/encoder_{dataset_type}"

    data_file = f"data/{dataset_type}/text/syn_{dataset_type}_train_text_filtered.csv"
    dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_path, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token

    dataset = get_dataset(dataset_type, data_file, encoder_tokenizer=tokenizer, dataset_json=dataset_json, add_features=True)

    custom_dataloader = DataLoader(dataset, batch_size=batch_size // gradient_accumulation_steps,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id, add_features=True))

    print("Training the model...")

    torch.cuda.empty_cache()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, custom_dataloader, scheduler
    )

    model.train()
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    labels = batch['token_labels']
                    outputs = model(tokens, attention_mask=attn_mask, labels=labels, past_key_values=None, use_cache=False)
                    loss = outputs.loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            save_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )

                    if accelerator.is_local_main_process:
                        wandb.log({"loss": loss.item()})

        if not no_checkpoint and save_dir is not None and accelerator.is_local_main_process and epoch != num_epochs - 1:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

        if accelerator.is_local_main_process:
            avg_loss = total_loss / len(training_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")

    if save_dir is not None and accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Mistral Model")

    parser.add_argument("--models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="gpt2-large", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model, None for default")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients, it should divide batch_size")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--save_period", type=int, default=3000000, help="Save the model every save_period steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument('--no_checkpoint', action='store_true', help='Whether to save checkpoints')

    args = parser.parse_args()

    train_encoder(args.models_path, args.model_name, args.dataset, args.save_dir,
                  args.batch_size, args.gradient_accumulation_steps, args.num_epochs, args.save_period, args.learning_rate,
                  args.warmup_steps, args.no_checkpoint)


