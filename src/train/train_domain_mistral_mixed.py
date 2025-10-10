import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torch.optim as optim
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
from transformers import get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.model.DomainExpert import DomainExpert
from src.dataset.utils import get_collate_fn, get_dataset, get_class_tokens

def train_domain_mistral(mistral_models_path: str, model_name: str, dataset_type_list: List[str], expert_dir: str,
                         encoder_model: str, system_prompt: str,
                         save_dir: Optional[str] = None, ffn_hidden_size: int = 2048, num_tokens: int = 10,
                         num_heads: int = 8, dropout: float = 0.0, use_norm: bool = True,
                         expert_input_size: int = 105, expert_output_size: int = 1,
                         max_new_tokens: int = 3000, batch_size: int = 64, gradient_accumulation_steps: int = 8,
                         num_epochs: int = 2, save_period: int = 300,
                         learning_rate: float = 1e-4, mapping_hidden_size=64, layer_to_add=None,
                         modeling_loss_ratio=0.1, warmup_steps=500, tune_lora=False, no_checkpoint=False):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=f"runs/domain_mistral_{'_'.join(dataset_type_list)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    model = DomainMistralForCausalLM.from_pretrained_mistral(model_name,
                                                             cache_dir=mistral_models_path, torch_dtype=torch.bfloat16,
                                                             tokenizer=tokenizer)

    model.num_tokens = num_tokens
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

    for dataset_type in dataset_type_list:
        if expert_dir is None:
            expert_dir = f"model/experts/syn_{dataset_type}_mlp.pth"

        if save_dir is None:
            save_dir = f"model/llm/domain_mistral_{dataset_type}"

        data_file = f"data/{dataset_type}/text/syn_{dataset_type}_train_text_filtered.csv"
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
        dataset_info_json = f"src/dataset/dataset_info.json"
        with open(dataset_info_json, 'r') as f:
            dataset_info_json = json.load(f)

        if dataset_type not in dataset_info_json:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if system_prompt is None:
            system_prompt = dataset_info_json[dataset_type]['system_prompt']

        if expert_input_size is None:
            expert_input_size = dataset_info_json[dataset_type]['expert_input_size']

        if expert_output_size is None:
            expert_output_size = dataset_info_json[dataset_type]['expert_output_size']

        class_tokens = get_class_tokens(dataset_type)

        expert = DomainExpert(embed_size=embed_size, ffn_hidden_size=ffn_hidden_size, expert_input_size=expert_input_size,
                              expert_output_size=expert_output_size, expert_dir=expert_dir, num_heads=num_heads,
                              dropout=dropout, use_norm=use_norm, max_length=max_new_tokens, dataset_json=dataset_json,
                              dataset_columns=None, expert_input_size_scaled=None, mapping_hidden_size=mapping_hidden_size,
                              num_tokens=num_tokens, cache_dir=mistral_models_path, encoder_model_id=encoder_model)

        base_model.add_expert_(expert,
                               [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in layer_to_add])

    model = model.to(torch.bfloat16)

    print("Loading the dataset...")

    dataset = get_dataset(dataset_type, data_file, tokenizer=tokenizer, dataset_json=dataset_json,
                          system_prompt=system_prompt)

    custom_dataloader = DataLoader(dataset, batch_size=batch_size // gradient_accumulation_steps,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id))

    if accelerator.is_local_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters ratio: {trainable_params / total_params:.4f}")

    print("Training the model encoder...")

    torch.cuda.empty_cache()

    for layer in layer_to_add:
        model.setup_encoder(layer, 0, True)
        model.setup_decoder(layer, 0, False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, custom_dataloader, scheduler
    )

    model.train()
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features']
                    outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)

                    auxiliary_loss = 0

                    if hasattr(model, 'module'):
                        model_to_use = model.module
                    else:
                        model_to_use = model

                    for layer in layer_to_add:
                        if tune_lora:
                            expert_input = model_to_use.base_model.model.get_expert_input(layer, 0).to(torch.bfloat16)
                        else:
                            expert_input = model_to_use.get_expert_input(layer, 0).to(torch.bfloat16)
                        features = features.to(torch.bfloat16)
                        auxiliary_loss += torch.nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                    loss = auxiliary_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_{epoch + 1}_{index + 1}.pt")

                    if accelerator.is_local_main_process:
                        writer.add_scalar('Auxiliary Loss/train', loss.item(), global_step)

            global_step += 1

        if not no_checkpoint and save_dir is not None and accelerator.is_local_main_process and epoch != num_epochs - 1:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_encoder_epoch_{epoch + 1}.pt")

        if accelerator.is_local_main_process:
            avg_loss = total_loss / len(training_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")
            writer.add_scalar('Average Loss/train', avg_loss, epoch)

    model = accelerator.unwrap_model(model)

    print("Training the model decoder...")

    torch.cuda.empty_cache()

    for layer in layer_to_add:
        model.setup_encoder(layer, 0, False)
        model.setup_decoder(layer, 0, True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)
                    logits = outputs.logits
                    class_logits = logits[:, -1, class_tokens]

                    labels = batch['labels']
                    modeling_loss = torch.nn.CrossEntropyLoss()(class_logits, labels)
                    loss = modeling_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_{epoch + 1}_{index + 1}.pt")

                    if accelerator.is_local_main_process:
                        writer.add_scalar('Modeling Loss/train', loss.item(), global_step)

            global_step += 1

        if not no_checkpoint and save_dir is not None and accelerator.is_local_main_process and epoch != num_epochs - 1:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_decoder_epoch_{epoch + 1}.pt")

        if accelerator.is_local_main_process:
            avg_loss = total_loss / len(training_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")
            writer.add_scalar('Average Loss/train', avg_loss, epoch)

    if save_dir is not None and accelerator.is_local_main_process:
        state = accelerator.get_state_dict(model)
        accelerator.save(state, f"{save_dir}/model_final.pt")

    if accelerator.is_local_main_process:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Mistral Model")

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name")
    parser.add_argument("--dataset", type=str, nargs='*', required=True, help="Dataset type")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--encoder_model", type=str, default="roberta-large", help="Encoder model name")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model, None for default")
    parser.add_argument("--ffn_hidden_size", type=int, default=512, help="Hidden size of the feed-forward neural network in the Domain Expert, -1 to use Linear layer instead of SwiGLU")
    parser.add_argument('--num_tokens', type=int, default=30, help='Number of tokens that expert appends to the input')
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the multi-head self-attention layer in the MappingBlock")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the MappingBlock")
    parser.add_argument("--use_norm", type=bool, default=True, help="Whether to use layer normalization after attention in the MappingBlock")
    parser.add_argument("--expert_input_size", type=int, default=None, help="Input size of the Domain Expert")
    parser.add_argument("--expert_output_size", type=int, default=None, help="Output size of the Domain Expert")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients, it should divide batch_size")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--save_period", type=int, default=3000000, help="Save the model every save_period steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--mapping_hidden_size", type=int, default=32, help="Hidden size for the mapping")
    parser.add_argument("--layer", type=int, nargs='*', default=None, help="Layers to add the expert, negative indexing supported")
    parser.add_argument("--modeling_loss_ratio", type=float, default=0.01, help="Modeling loss ratio")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--no_checkpoint', action='store_true', help='Whether to save checkpoints')

    args = parser.parse_args()

    train_domain_mistral(args.mistral_models_path, args.model_name, args.dataset, args.expert_dir, args.encoder_model,
                         args.system_prompt, args.save_dir,
                         args.ffn_hidden_size, args.num_tokens, args.num_heads, args.dropout, args.use_norm,
                         args.expert_input_size, args.expert_output_size,
                         args.max_new_tokens, args.batch_size, args.gradient_accumulation_steps, args.num_epochs,
                         args.save_period,
                         args.learning_rate, args.mapping_hidden_size, args.layer, args.modeling_loss_ratio,
                         args.warmup_steps, args.tune_lora, args.no_checkpoint)