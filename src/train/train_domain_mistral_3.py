import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor, PreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from typing import List, Optional, Tuple, Union
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
import argparse
from transformers import get_constant_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.model.DomainExpert import DomainExpert
from src.dataset.adult.InstructTextDataset import AdultTextDataset, get_collate_fn
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN


def train_domain_mistral(mistral_models_path: str, model_name: str, expert_dir: str, system_prompt,
                         save_dir: Optional[str] = None, data_file: str = None,
                         ffn_hidden_size: int = 2048, num_heads: int = 8, dropout: float = 0.1, use_norm: bool = True,
                         expert_input_size: int = 105, expert_output_size: int = 1,
                         max_new_tokens: int = 3000, batch_size: int = 64,
                         gradient_accumulation_steps: int = 8, num_epochs: int = 2, save_period: int = 300,
                         learning_rate: float = 1e-4, dataset_json=None, dataset_columns=None,
                         expert_input_size_scaled=None, mapping_hidden_size=64, layer_to_add=[0], modeling_loss_ratio=0.1, warmup_steps=500, tune_lora=False):
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
        max_new_tokens: maximum number of tokens to generate
        system_prompt: system prompt
        data_file: path to the training data
        batch_size: batch size
        gradient_accumulation_steps: number of steps to accumulate gradients, it should divide batch_size
        num_epochs: number of epochs
        save_period: save the model every save_period steps
        learning_rate: learning rate

    Returns:

    """
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if accelerator.is_local_main_process:
        wandb.init(project="SiT", name="train_domain_mistral")
        wandb.config.update({
            "mistral_models_path": mistral_models_path,
            "model_name": model_name,
            "expert_dir": expert_dir,
            "save_dir": save_dir,
            "ffn_hidden_size": ffn_hidden_size,
            "num_heads": num_heads,
            "dropout": dropout,
            "use_norm": use_norm,
            "expert_input_size": expert_input_size,
            "expert_output_size": expert_output_size,
            "max_new_tokens": max_new_tokens,
            "system_prompt": system_prompt,
            "data_file": data_file,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "save_period": save_period,
            "learning_rate": learning_rate,
            "dataset_json": dataset_json,
            "dataset_columns": dataset_columns,
            "expert_input_size_scaled": expert_input_size_scaled,
            "mapping_hidden_size": mapping_hidden_size,
            "layer_to_add": layer_to_add,
            "modeling_loss_ratio": modeling_loss_ratio,
            "warmup_steps": warmup_steps,
            "tune_lora": tune_lora,
        })

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

    if tune_lora:
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, config)
        base_model = model.base_model.model
    else:
        base_model = model

    if layer_to_add is None:
        layer_to_add = list(range(len(base_model.model.layers)))

    # Get the token ids for "Yes" and "No"
    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    embed_yes = base_model.model.embed_tokens(torch.tensor(yes_token_list))[0]
    embed_no = base_model.model.embed_tokens(torch.tensor(no_token_list))[0]

    expert = DomainExpert(embed_size, ffn_hidden_size, expert_input_size, expert_output_size, expert_dir, embed_yes, embed_no,
                          num_heads, dropout, use_norm, max_new_tokens, dataset_json, dataset_columns,
                          expert_input_size_scaled, mapping_hidden_size)

    base_model.add_expert_(expert, [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in layer_to_add])

    model = model.to(torch.bfloat16)

    print("Loading the dataset...")
    dataset = AdultTextDataset(pd.read_csv(data_file), tokenizer,
                               system_prompt=system_prompt, dataset_json=dataset_json, add_features=True)
    custom_dataloader = DataLoader(dataset, batch_size=batch_size // gradient_accumulation_steps,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id, add_features=True))

    if accelerator.is_local_main_process:
        # output the count of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")
        # print the ratio of trainable parameters to the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters ratio: {trainable_params / total_params:.4f}")

    print("Training the model...")

    torch.cuda.empty_cache()

    os.makedirs(save_dir, exist_ok=True)
    '''
    if accelerator.is_local_main_process:
        def get_model_param_dict(model):
            param_dict = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    param_dict[name] = param.clone().detach()
            return param_dict

        # Save the initial state of the parameters
        initial_param_dict = get_model_param_dict(model)
    '''
    for layer in layer_to_add:
        base_model.setup_encoder(layer, 0, True)
        base_model.setup_decoder(layer, 0, False)

    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, custom_dataloader, scheduler
    )

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features']
                    token_labels = batch['token_labels']
                    
                    outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)
                    logits = outputs.logits
                    yes_logits = logits[:, -1, yes_token_list].mean(dim=-1)
                    no_logits = logits[:, -1, no_token_list].mean(dim=-1)
                    logits = torch.stack([no_logits, yes_logits], dim=-1)

                    labels = batch['labels']
                    modeling_loss = torch.nn.CrossEntropyLoss()(logits, labels)

                    auxiliary_loss = 0

                    for layer in layer_to_add:
                        if tune_lora:
                            expert_input = model.module.base_model.model.get_expert_input(layer, 0).to(torch.bfloat16)
                        else:
                            expert_input = model.module.get_expert_input(layer, 0).to(torch.bfloat16)
                        features = features.to(torch.bfloat16)
                        auxiliary_loss += nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                    loss = auxiliary_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_0_{epoch + 1}_{index + 1}.pt")

                        # model.save_checkpoint(save_dir)

                    if accelerator.is_local_main_process:
                        wandb.log({"auxiliary_loss": auxiliary_loss.item()})   

    model = accelerator.unwrap_model(model)

    if tune_lora:
        base_model = model.base_model.model
    else:
        base_model = model

    for layer in layer_to_add:
        base_model.setup_encoder(layer, 0, False)
        base_model.setup_decoder(layer, 0, True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, custom_dataloader, scheduler
    )

    model.train()
  
    for epoch in range(num_epochs):
        total_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features']
                    token_labels = batch['token_labels']
                    
                    outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)
                    logits = outputs.logits
                    yes_logits = logits[:, -1, yes_token_list].mean(dim=-1)
                    no_logits = logits[:, -1, no_token_list].mean(dim=-1)
                    logits = torch.stack([no_logits, yes_logits], dim=-1)

                    labels = batch['labels']
                    modeling_loss = torch.nn.CrossEntropyLoss()(logits, labels)

                    loss = modeling_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_1_{epoch + 1}_{index + 1}.pt")

                        # model.save_checkpoint(save_dir)

                    if accelerator.is_local_main_process:
                        wandb.log({"modeling_loss": modeling_loss.item()})  

        if save_dir is not None and accelerator.is_local_main_process and epoch != num_epochs - 1:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_epoch_{epoch + 1}.pt")

        if accelerator.is_local_main_process:
            avg_loss = total_loss / len(training_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")

    if save_dir is not None and accelerator.is_local_main_process:
        state = accelerator.get_state_dict(model)
        accelerator.save(state, f"{save_dir}/model_final.pt")
        '''
        def compare_model_params(initial_params, model):
            for name, initial_param in initial_params.items():
                current_param = model.state_dict()[name].to(initial_param.device)
                # print(current_param)
                if not torch.equal(initial_param, current_param):
                    print(f"Parameter {name} has changed.")
                # else:
                #    print(f"Parameter {name} has not changed.")

        # Compare the parameters after training
        print(initial_param_dict.keys())
        compare_model_params(initial_param_dict, model)
        '''

    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Mistral Model")

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name")
    parser.add_argument("--expert_dir", type=str, default="model/experts/syn_adult_mlp.pth", help="Path to the domain expert")
    parser.add_argument("--save_dir", type=str, default="model/llm/domain_single_token", help="Path to save the model")
    parser.add_argument("--ffn_hidden_size", type=int, default=14336, help="Hidden size of the feed-forward neural network in the Domain Expert")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the multi-head self-attention layer in the MappingBlock")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the MappingBlock")
    parser.add_argument("--use_norm", type=bool, default=True, help="Whether to use layer normalization after attention in the MappingBlock")
    parser.add_argument("--expert_input_size", type=int, default=105, help="Input size of the Domain Expert")
    parser.add_argument("--expert_output_size", type=int, default=1, help="Output size of the Domain Expert")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--syn_text_data_file", type=str, default="data/adult/text/syn_adult_train_text.csv", help="Path to the synthetic text data file")
    parser.add_argument("--batch_size", type=int, default=21, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="Number of steps to accumulate gradients, it should divide batch_size")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--save_period", type=int, default=3000000, help="Save the model every save_period steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--data_json", type=str, default="src/dataset/adult/adult.json", help="Path to the data JSON file")
    parser.add_argument("--mapping_hidden_size", type=int, default=32, help="Hidden size for the mapping")
    parser.add_argument("--layer", type=int, nargs='*', default=None, help="Layers to add the expert, negative indexing supported")
    parser.add_argument("--modeling_loss_ratio", type=float, default=1e-5, help="Modeling loss ratio")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')

    args = parser.parse_args()

    train_domain_mistral(args.mistral_models_path, args.model_name, args.expert_dir, args.system_prompt, args.save_dir, args.syn_text_data_file,
                         args.ffn_hidden_size, args.num_heads, args.dropout, args.use_norm, args.expert_input_size, args.expert_output_size,
                         args.max_new_tokens, args.batch_size, args.gradient_accumulation_steps, args.num_epochs, args.save_period,
                         args.learning_rate, args.data_json, None, None, args.mapping_hidden_size, args.layer, args.modeling_loss_ratio, args.warmup_steps, args.tune_lora)
