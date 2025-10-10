import sys
import os
import types

import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PrefixTuningConfig, TaskType, PeftModel, PeftModelForCausalLM
import torch.optim as optim
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import json
import gorilla

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.model.DomainExpert import DomainExpert
from src.dataset.utils_generation import get_collate_fn_generation, get_dataset_generation
from src.preprocess.DataScaler import TableScaler
from util import custom_get_prompt

patch = gorilla.Patch(peft.PeftModel, 'get_prompt', custom_get_prompt, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)


def train_domain_mistral_generation(mistral_models_path: str, model_name: str, dataset_type: str, expert_dir: str,
                         encoder_model: str, system_prompt: str,
                         save_dir: Optional[str] = None, ffn_hidden_size: int = 2048, num_tokens: int = 10,
                         num_heads: int = 8, dropout: float = 0.0, use_norm: bool = True,
                         expert_input_size: int = 105, expert_output_size: int = 1,
                         max_new_tokens: int = 3000, batch_size: int = 64, gradient_accumulation_steps: int = 8,
                         num_encoder_epochs: int = 2, save_period: int = 300,
                         learning_rate_encoder: float = 1e-3, mapping_hidden_size=64, layer_to_add=None,
                         modeling_loss_ratio=0.1, warmup_steps=500, tune_lora=False, no_checkpoint=False,
                         num_decoder_epochs=2, learning_rate_decoder=1e-4, prefix_tuning=False,
                         eval_when_train=False, early_stopping=False, enc_patience=3, ablation_suffix: str = None, 
                         disable_emb_translate=False, llamdex_padding='gaussian', expert_type='mlp'):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=f"runs/domain_mistral_generation_{dataset_type}")

    # Handle ablation studies
    dataset_suffix = ""
    if ablation_suffix:
        dataset_suffix = f"_ablation{ablation_suffix}"
        print(f"Training with ablation suffix: {ablation_suffix}")
    
    if expert_dir is None:
        if ablation_suffix:
            # Handle both MLP and GBDT for ablation studies
            if expert_type == 'mlp':
                expert_dir = f"model/experts/syn_{dataset_type}{dataset_suffix}_mlp.pth"
            elif expert_type == 'gbdt':
                expert_dir = f"model/experts/syn_{dataset_type}{dataset_suffix}_gbdt.json"
            else:
                raise ValueError(f"Invalid expert type: {expert_type}")
        else:
            if expert_type == 'mlp':
                expert_dir = f"model/experts/syn_{dataset_type}_mlp.pth"
            elif expert_type == 'gbdt':
                expert_dir = f"model/experts/syn_{dataset_type}_gbdt.json"
            else:
                raise ValueError(f"Invalid expert type: {expert_type}")

    if save_dir is None:
        save_dir = f"model/llm/domain_mistral_generation_{dataset_type}{dataset_suffix}"

    # Construct data file paths with ablation support - use generation-specific data
    if ablation_suffix:
        syn_data_file = f"data/{dataset_type}/text_generation_ablation/syn_{dataset_type}{dataset_suffix}_train_text_generation.csv"
        val_data_file = f"data/{dataset_type}/text_generation_ablation/syn_{dataset_type}{dataset_suffix}_test_text_generation.csv"
        real_data_file = f"data/{dataset_type}/text_generation/{dataset_type}_test_text_generation.csv"  # Use original real data for testing
    else:
        syn_data_file = f"data/{dataset_type}/text_generation/syn_{dataset_type}_train_text_generation.csv"
        real_data_file = f"data/{dataset_type}/text_generation/{dataset_type}_test_text_generation.csv"
        val_data_file = f"data/{dataset_type}/text_generation/syn_{dataset_type}_test_text_generation.csv"
    
    print(f"Training data file: {syn_data_file}")
    print(f"Validation data file: {val_data_file}")
    print(f"Test data file: {real_data_file}")
    
    # Check if training data file exists
    if not os.path.exists(syn_data_file):
        raise FileNotFoundError(f"Training data file not found: {syn_data_file}")
    if not os.path.exists(val_data_file):
        raise FileNotFoundError(f"Validation data file not found: {val_data_file}")
    
    # Use age-specific JSON configuration for ablation studies
    if ablation_suffix and dataset_type == "titanic":
        # Extract age bound from ablation suffix (e.g., "_age20" -> "20")
        age_bound = ablation_suffix.replace("_age", "")
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}_ablation_age{age_bound}.json"
        print(f"Using age-specific configuration: {dataset_json}")
        
        # Check if age-specific JSON exists, fallback to original if not
        if not os.path.exists(dataset_json):
            print(f"Warning: Age-specific JSON not found: {dataset_json}")
            print(f"Falling back to original configuration")
            dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
    else:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
    
    dataset_info_json = f"src/dataset/dataset_info_generation.json"
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

    assert not (tune_lora and prefix_tuning), "Only one of tune_lora and prefix_tuning can be True"
    if tune_lora:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )

        model = get_peft_model(model, config)
        base_model = model.base_model.model
    if prefix_tuning:
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=10,

        )
        model = get_peft_model(model, config)

        base_model = model.base_model
    else:
        base_model = model

    if layer_to_add is None:
        layer_to_add = list(range(len(base_model.model.layers)))

    expert_input_scaler = TableScaler(dataset_json).tensor_map

    expert = DomainExpert(embed_size=embed_size, ffn_hidden_size=ffn_hidden_size, expert_input_size=expert_input_size,
                          expert_output_size=expert_output_size, expert_dir=expert_dir, num_heads=num_heads,
                          dropout=dropout, use_norm=use_norm, max_length=max_new_tokens, dataset_json=dataset_json,
                          dataset_columns=None, expert_input_size_scaled=None, mapping_hidden_size=mapping_hidden_size,
                          num_tokens=num_tokens, cache_dir=mistral_models_path, encoder_model_id=encoder_model,
                          expert_input_scaler=expert_input_scaler)

    base_model.add_expert_(expert,
                           [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in layer_to_add])

    model = model.to(torch.bfloat16)

    print("Loading the dataset...")

    train_dataset = get_dataset_generation(dataset_type, syn_data_file, tokenizer=tokenizer, dataset_json=dataset_json,
                                system_prompt=system_prompt)
    test_dataset = get_dataset_generation(dataset_type, real_data_file, tokenizer=tokenizer, dataset_json=dataset_json,
                               system_prompt=system_prompt)
    val_dataset = get_dataset_generation(dataset_type, val_data_file, tokenizer=tokenizer, dataset_json=dataset_json,
                              system_prompt=system_prompt)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size // gradient_accumulation_steps,
                                  collate_fn=get_collate_fn_generation(tokenizer.pad_token_id))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size // gradient_accumulation_steps,
                                 collate_fn=get_collate_fn_generation(tokenizer.pad_token_id))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size // gradient_accumulation_steps,
                                collate_fn=get_collate_fn_generation(tokenizer.pad_token_id))

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

    encoder_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_encoder)
    encoder_scheduler = get_cosine_schedule_with_warmup(encoder_optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_encoder_epochs * len(train_dataloader))

    model, encoder_optimizer, train_dataloader, test_dataloader, val_dataloader, encoder_scheduler = accelerator.prepare(
        model, encoder_optimizer, train_dataloader, test_dataloader, val_dataloader, encoder_scheduler)

    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_encoder_epochs):
        # train encoder
        train_loss = 0
        model.train()
        for index, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    encoder_optimizer.zero_grad()
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features']
                    outputs = model(tokens,
                                    attention_mask=attn_mask, use_cache=False)

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
                        features = features.to(torch.bfloat16).to(model.device)
                        auxiliary_loss += torch.nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                    # For generation, we use the language modeling loss as a small component
                    token_labels = batch['token_labels']
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = token_labels[..., 1:].contiguous()
                    modeling_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    loss = auxiliary_loss + modeling_loss_ratio * modeling_loss

                    accelerator.backward(loss)
                    encoder_optimizer.step()
                    encoder_scheduler.step()

                    train_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_{epoch + 1}_{index + 1}.pt")

                    if accelerator.is_local_main_process:
                        writer.add_scalar('Total Loss/train', loss.item(), global_step)
                        writer.add_scalar('Auxiliary Loss/train', auxiliary_loss.item(), global_step)
                        writer.add_scalar('Modeling Loss/train', modeling_loss.item(), global_step)

            global_step += 1

        test_loss = 0
        if eval_when_train:
            model.eval()
            with torch.no_grad():
                for index, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                                         disable=not accelerator.is_main_process):
                    with accelerator.accumulate(model):
                        with accelerator.autocast():
                            tokens = batch['tokens']
                            attn_mask = batch['attention_mask']
                            features = batch['features']
                            outputs = model(tokens, attention_mask=attn_mask,
                                            use_cache=False)

                            auxiliary_loss = 0

                            if hasattr(model, 'module'):
                                model_to_use = model.module
                            else:
                                model_to_use = model

                            for layer in layer_to_add:
                                if tune_lora:
                                    expert_input = model_to_use.base_model.model.get_expert_input(layer, 0).to(
                                        torch.bfloat16)
                                else:
                                    expert_input = model_to_use.get_expert_input(layer, 0).to(torch.bfloat16)
                                features = features.to(torch.bfloat16)
                                auxiliary_loss += torch.nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                            test_loss += auxiliary_loss.item()

                            if accelerator.is_local_main_process:
                                writer.add_scalar('Auxiliary Loss/test', auxiliary_loss.item(), global_step)

        avg_val_loss = 0
        if early_stopping:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, disable=not accelerator.is_main_process):
                    with accelerator.autocast():
                        tokens = batch['tokens']
                        attn_mask = batch['attention_mask']
                        features = batch['features']
                        outputs = model(tokens, attention_mask=attn_mask, use_cache=False)

                        auxiliary_loss = 0

                        if hasattr(model, 'module'):
                            model_to_use = model.module
                        else:
                            model_to_use = model

                        for layer in layer_to_add:
                            if tune_lora:
                                expert_input = model_to_use.base_model.model.get_expert_input(layer, 0).to(
                                    torch.bfloat16)
                            else:
                                expert_input = model_to_use.get_expert_input(layer, 0).to(torch.bfloat16)
                            features = features.to(torch.bfloat16).to(model.device)
                            auxiliary_loss += torch.nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                        val_loss += auxiliary_loss.item()

            avg_val_loss = val_loss / len(val_dataloader)

            if accelerator.is_local_main_process:
                writer.add_scalar('Auxiliary Loss/val', avg_val_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

        model.train()

        if not no_checkpoint and save_dir is not None and accelerator.is_local_main_process and epoch != num_encoder_epochs - 1:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_encoder_epoch_{epoch + 1}.pt")

        if accelerator.is_local_main_process:
            avg_train_loss = train_loss / len(train_dataloader)
            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Epoch {epoch + 1}/{num_encoder_epochs} train Loss: {avg_train_loss:.4f}, test Loss: {avg_test_loss:.4f}, val Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Average Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Average Loss/test', avg_test_loss, epoch)

        if early_stopping and patience_counter >= enc_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    model = accelerator.unwrap_model(model)

    print("Training the model decoder...")

    torch.cuda.empty_cache()

    for layer in layer_to_add:
        model.setup_encoder(layer, 0, False)
        model.setup_decoder(layer, 0, True)

    decoder_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_decoder)
    decoder_scheduler = get_cosine_schedule_with_warmup(decoder_optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_decoder_epochs * len(train_dataloader))
    model, decoder_optimizer, train_dataloader, val_dataloader, decoder_scheduler = accelerator.prepare(
        model, decoder_optimizer, train_dataloader, val_dataloader, decoder_scheduler)

    global_step = 0

    for epoch in range(num_decoder_epochs):
        train_loss = 0
        for index, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    decoder_optimizer.zero_grad()
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features'].to(torch.bfloat16).to(model.device)
                    token_labels = batch['token_labels']
                    
                    outputs = model(tokens, expert_inputs=(features,),
                                    attention_mask=attn_mask, use_cache=False)
                    
                    logits = outputs.logits
                    # Standard language modeling loss for sequence generation
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = token_labels[..., 1:].contiguous()
                    modeling_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    loss = modeling_loss

                    accelerator.backward(loss)
                    decoder_optimizer.step()
                    decoder_scheduler.step()

                    train_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_{epoch + 1}_{index + 1}.pt")

                    if accelerator.is_local_main_process:
                        writer.add_scalar('Modeling Loss/train', loss.item(), global_step)

            global_step += 1

        if not no_checkpoint and save_dir is not None and accelerator.is_local_main_process and epoch != num_decoder_epochs - 1:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"{save_dir}/model_decoder_epoch_{epoch + 1}.pt")

        if accelerator.is_local_main_process:
            avg_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{num_decoder_epochs} Loss: {avg_loss:.4f}")
            writer.add_scalar('Average Loss/train', avg_loss, epoch)

    if save_dir is not None and accelerator.is_local_main_process:
        state = accelerator.get_state_dict(model)
        accelerator.save(state, f"{save_dir}/model_final.pt")

    if accelerator.is_local_main_process:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Mistral Model for Generation")

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--encoder_model", type=str, default="roberta-large", help="Encoder model name")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model, None for default")
    parser.add_argument("--ffn_hidden_size", type=int, default=512,
                        help="Hidden size of the feed-forward neural network in the Domain Expert, -1 to use Linear layer instead of SwiGLU")
    parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens that expert appends to the input')
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads in the multi-head self-attention layer in the MappingBlock")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the MappingBlock")
    parser.add_argument("--use_norm", type=bool, default=True,
                        help="Whether to use layer normalization after attention in the MappingBlock")
    parser.add_argument("--expert_input_size", type=int, default=None, help="Input size of the Domain Expert")
    parser.add_argument("--expert_output_size", type=int, default=None, help="Output size of the Domain Expert")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients, it should divide batch_size")
    parser.add_argument("--num_epochs_enc", type=int, default=4, help="Number of epochs to train encoder")
    parser.add_argument("--num_epochs_dec", type=int, default=4, help="Number of epochs to train decoder")
    parser.add_argument("--save_period", type=int, default=3000000, help="Save the model every save_period steps")
    parser.add_argument("--learning_rate_enc", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--learning_rate_dec", type=float, default=5e-5, help="Learning rate for the decoder")
    parser.add_argument("--mapping_hidden_size", type=int, default=32, help="Hidden size for the mapping")
    parser.add_argument("--layer", type=int, nargs='*', default=None,
                        help="Layers to add the expert, negative indexing supported")
    parser.add_argument("--modeling_loss_ratio", type=float, default=0.01, help="Modeling loss ratio")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--prefix-tuning', action='store_true', help='Whether to use prefix tuning')
    parser.add_argument('--no_checkpoint', action='store_true', help='Whether to save checkpoints')
    parser.add_argument('--eval_when_train', action='store_true', help='Whether to evaluate when training')
    parser.add_argument('--early_stopping', action='store_true', help='Whether to use early stopping for encoder')
    parser.add_argument('--enc_patience', type=int, default=3,
                        help='Number of epochs with no improvement after which training will be stopped for encoder')
    parser.add_argument('--ablation_suffix', type=str, default=None,
                        help='Suffix for ablation study (e.g., "_age20" for age ablation)')
    parser.add_argument('--expert_type', type=str, default='mlp', choices=['mlp', 'gbdt'],
                        help='Type of expert model to use (mlp or gbdt)')

    args = parser.parse_args()

    print(args)

    train_domain_mistral_generation(args.mistral_models_path, args.model_name, args.dataset, args.expert_dir, args.encoder_model,
                         args.system_prompt, args.save_dir,
                         args.ffn_hidden_size, args.num_tokens, args.num_heads, args.dropout, args.use_norm,
                         args.expert_input_size, args.expert_output_size,
                         args.max_new_tokens, args.batch_size, args.gradient_accumulation_steps, args.num_epochs_enc,
                         args.save_period,
                         args.learning_rate_enc, args.mapping_hidden_size, args.layer, args.modeling_loss_ratio,
                         args.warmup_steps, args.tune_lora, args.no_checkpoint,
                         num_decoder_epochs=args.num_epochs_dec, learning_rate_decoder=args.learning_rate_dec,
                         prefix_tuning=args.prefix_tuning, eval_when_train=args.eval_when_train,
                         early_stopping=args.early_stopping, enc_patience=args.enc_patience, 
                         ablation_suffix=args.ablation_suffix, expert_type=args.expert_type) 