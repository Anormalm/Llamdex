import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import deepspeed
import wandb
from tqdm.auto import tqdm
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.dataset.adult.InstructTextDataset import AdultTextDatasetForPerTokenTraining, get_collate_fn


def train_domain_mistral(mistral_models_path: str, model_name: str, expert_dir: str, system_prompt,
                         save_dir: Optional[str] = None, data_file: str = None,
                         ffn_hidden_size: int = 2048, num_heads: int = 8, dropout: float = 0.1, use_norm: bool = True,
                         expert_input_size: int = 105, expert_output_size: int = 1, per_device_train_batch_size: int = 64,
                         gradient_accumulation_steps: int = 8, num_epochs: int = 2, save_period: int = 500,
                         learning_rate: float = 1e-4, auto_find_batch_size: bool = False):
    """
    Train the Mistral model with a domain expert.
    Args:
        mistral_models_path: path to the model cache
        model_name: model ID or path to the model
        expert_dir: path to the domain expert
        save_dir: path to save the model
        ffn_hidden_size: hidden size of the feed-forward neural network in the Domain Expert
        num_heads: number of heads in the multi-head self-attention layer in the MappingBlock
        dropout: dropout rate in the MappingBlock
        use_norm: whether to use layer normalization after attention in the MappingBlock
        expert_input_size: input size of the Domain Expert
        expert_output_size: output size of the Domain Expert
        system_prompt: system prompt
        data_file: path to the training data
        per_device_train_batch_size: batch size per device
        gradient_accumulation_steps: number of steps to accumulate gradients, it should divide batch_size
        num_epochs: number of epochs
        save_period: save the model every save_period steps
        learning_rate: learning rate

    Returns:

    """
    wandb.init(project="SiT", name="train_domain_mistral_per_token")

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    model = DomainMistralForCausalLM.from_pretrained_mistral(model_name,
                                                             cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token

    embed_size = model.config.hidden_size

    print("Adding the custom expert to the model...")

    for p in model.parameters():
        p.requires_grad = False

    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["gate"],
    )
    model = get_peft_model(model, config)

    model.base_model.model.add_expert_(embed_size, ffn_hidden_size)

    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    print("Loading the dataset...")
    dataset = AdultTextDatasetForPerTokenTraining(pd.read_csv(data_file), tokenizer, system_prompt=system_prompt)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # output the count of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    # print the ratio of trainable parameters to the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters ratio: {trainable_params / total_params:.4f}")

    print("Training the model...")

    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_strategy="no",
        save_total_limit=3,
        learning_rate=learning_rate,
        # deepspeed="../../conf/ds_config.json",
        bf16=True,
        report_to="wandb",
        logging_dir=f"{save_dir}/logs",
        auto_find_batch_size=auto_find_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )


    # Trainer issue in https://github.com/nlp-with-transformers/notebooks/issues/31
    old_collator = trainer.data_collator
    trainer.data_collator = lambda data: dict(old_collator(data))

    trainer.train()

    if save_dir is not None:
        torch.save(model.state_dict(), f"{save_dir}/domain_mistral_model_1.pt")

    wandb.finish()


if __name__ == '__main__':
    mistral_models_path = "/disk1/zhaomin/SiT/model/llm"  # path to the model cache
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # model name
    expert_dir = "/disk1/zhaomin/SiT/model/experts/syn_adult_mlp.pth"  # path to the domain expert
    save_dir = "/disk1/zhaomin/SiT/model/llm/tmp_train"  # path to save the model
    ffn_hidden_size = 2048  # hidden size of the feed-forward neural network in the Domain Expert
    num_heads = 8  # number of heads in the multi-head self-attention layer in the MappingBlock
    dropout = 0.1  # dropout rate in the MappingBlock
    use_norm = True  # whether to use layer normalization after attention in the MappingBlock
    expert_input_size = 105  # input size of the Domain Expert
    expert_output_size = 1  # output size of the Domain Expert
    system_prompt = "Respond the question in only one word: Yes or No. Please do not refuse to answer or output any other word. "
    data_file = "/disk1/zhaomin/SiT/data/adult/processed/syn_adult_train_text.csv"
    per_device_train_batch_size = 4  # batch size per device
    gradient_accumulation_steps = 16  # number of steps to accumulate gradients, it should divide batch_size
    # batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_devices
    num_epochs = 2  # number of epochs
    save_period = 100000 # save the model every save_period steps, error for auto saving now: TypeError: Object of type dtype is not JSON serializable
    learning_rate = 1e-4  # learning rate
    auto_find_batch_size = True  # whether to automatically find the batch size, deepspeed doesn't support this feature
    train_domain_mistral(mistral_models_path, model_name, expert_dir, system_prompt, save_dir, data_file,
                         ffn_hidden_size, num_heads, dropout, use_norm, expert_input_size, expert_output_size,
                         per_device_train_batch_size, gradient_accumulation_steps, num_epochs, save_period, 
                         learning_rate, auto_find_batch_size)