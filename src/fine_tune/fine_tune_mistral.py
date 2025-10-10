import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torch.optim as optim
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn_for_baseline, get_dataset_for_baseline


def fine_tune_mistral(mistral_models_path: str, model_name: str, dataset: str, save_dir: Optional[str] = None,
                      batch_size: int = 64, gradient_accumulation_steps: int = 4, num_epochs: int = 4,
                      learning_rate: float = 5e-5, warmup_steps: int = 500, tune_lora: bool = False,
                      dp_type: str = None, epsilon: str = None, save_lora_only: bool = False, seed: int = 42):

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if save_lora_only and not tune_lora:
        raise ValueError("LoRA can't be saved if not using it.")

    if save_dir is None:
        if dp_type is None:
            save_dir = f"model/llm/exp-lora-real_mistral_{dataset}"
        else:
            save_dir = f"model/llm/exp-lora-syn_mistral_{dataset}_{dp_type}_{epsilon}"

    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=f"runs/fine_tune_mistral_{dataset}")

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)


    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token

    if tune_lora:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )

        model = get_peft_model(model, config)

    model = model.to(torch.bfloat16)

    if dp_type is None:
        data_file = f"data/{dataset}/text/{dataset}_train_text.csv"
    else:
        # if dp_type not in ['dp-wgan', 'pate-gan', 'prompt-pate', 'table-diffusion']:
        #     raise ValueError(f"Unknown differential privacy type: {dp_type}")
        data_file = f"data/{dataset}/dpsyn/eps{epsilon}/{dp_type}/synthetic_data.csv"

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"The file {data_file} does not exist.")

    train_dataset = get_dataset_for_baseline(dataset, data_file, tokenizer=tokenizer, syn_mode=dp_type is not None)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size // gradient_accumulation_steps,
                                  collate_fn=get_collate_fn_for_baseline(tokenizer.pad_token_id))

    torch.cuda.empty_cache()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_epochs * len(train_dataloader))

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler)

    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        train_loss = 0
        for index, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    token_labels = batch['token_labels']

                    outputs = model(input_ids=tokens, attention_mask=attn_mask, labels=token_labels,
                                    past_key_values=None, use_cache=False)
                    loss = outputs.loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()

                    if accelerator.is_local_main_process:
                        writer.add_scalar('Modeling Loss/train', loss.item(), global_step)

            global_step += 1

        if accelerator.is_local_main_process:
            avg_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")
            writer.add_scalar('Average Loss/train', avg_loss, epoch)

    if accelerator.is_local_main_process:
        if not save_lora_only:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_final.pt")
        else:
            model = accelerator.unwrap_model(model)
            model.save_pretrained(save_dir)

        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Mistral Model")

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model, None for default")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps to accumulate gradients, it should divide batch_size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--dp_type', type=str, default=None, help='Differential privacy type')
    parser.add_argument('--epsilon', type=str, default=None, help='Epsilon for DP')
    parser.add_argument('--save_lora_only', action='store_true', help='Whether to save only the LoRA part of the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(args)

    fine_tune_mistral(args.mistral_models_path, args.model_name, args.dataset, args.save_dir, args.batch_size,
                      args.gradient_accumulation_steps, args.num_epochs, args.learning_rate, args.warmup_steps,
                      args.tune_lora, args.dp_type, args.epsilon, args.save_lora_only, args.seed)

