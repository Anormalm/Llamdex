import sys
import os

from transformers import AutoTokenizer
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.dataset.adult.InstructTextDataset import AdultTextDataset, get_collate_fn
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN
from src.preprocess.DataScaler import TableScaler
from src.model.DomainExpert import DomainExpert


def train_domain_mistral(mistral_models_path: str, model_name: str, expert_dir: str, system_prompt,
                         save_dir: Optional[str] = None, data_file: str = None, test_data_file: str = None,
                         ffn_hidden_size: int = 2048, num_heads: int = 8, dropout: float = 0.1, use_norm: bool = True,
                         expert_input_size: int = 105, expert_output_size: int = 1, top_k: int = 256,
                         max_new_tokens: int = 3000, batch_size: int = 64,
                         gradient_accumulation_steps: int = 8, num_epochs: int = 2, save_period: int = 300,
                         learning_rate: float = 1e-4, dataset_json=None, dataset_columns=None,
                         expert_input_size_scaled=None, mapping_hidden_size=64):
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
        top_k: number of tokens to route to the Domain Expert
        max_new_tokens: maximum number of tokens to generate
        system_prompt: system prompt
        data_file: path to the training data
        batch_size: batch size
        gradient_accumulation_steps: number of steps to accumulate gradients, it should divide batch_size
        num_epochs: number of epochs
        save_period: save the model every save_period steps
        learning_rate: learning rate
        dataset_json: JSON file containing the dataset information
        dataset_columns: The ordered columns of the dataset to be scaled

    Returns:

    """
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")

    if accelerator.is_local_main_process:
        wandb.init(project="SiT", name="train_domain_mistral_stage_2")

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

    '''
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["gate"],
    )

    model = get_peft_model(model, config)
    '''
    for p in model.parameters():
        p.requires_grad = False

    expert = DomainExpert(embed_size, ffn_hidden_size, expert_input_size, expert_output_size, expert_dir,
                          num_heads, dropout, use_norm, max_new_tokens, dataset_json, dataset_columns,
                          expert_input_size_scaled, mapping_hidden_size)

    model.add_expert_(expert, 0)

    model = model.to(torch.bfloat16)

    # Get the token ids for "Yes" and "No"
    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    print("Loading the dataset...")
    dataset = AdultTextDataset(pd.read_csv(data_file), tokenizer,
                               system_prompt=system_prompt, dataset_json=dataset_json)
    test_dataset = AdultTextDataset(pd.read_csv(test_data_file), tokenizer,
                                    system_prompt=system_prompt, dataset_json=dataset_json)
    custom_train_dataloader = DataLoader(dataset, batch_size=batch_size // gradient_accumulation_steps,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id))
    custom_test_dataloader = DataLoader(test_dataset, batch_size=batch_size // gradient_accumulation_steps,
                                        collate_fn=get_collate_fn(tokenizer.pad_token_id))

    if accelerator.is_local_main_process:
        # output the count of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")
        # print the ratio of trainable parameters to the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters ratio: {trainable_params / total_params:.4f}")

    print("Training the model...")

    torch.cuda.empty_cache()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model, optimizer, training_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, custom_train_dataloader, custom_test_dataloader)

    model.train()
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        train_loss = 0
        for index, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader),
                                 disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    tokens = batch['tokens']
                    attn_mask = batch['attention_mask']
                    features = batch['features']
                    outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)

                    '''
                    if accelerator.is_local_main_process:
                        print(f"Logits: {logits}")
                        print(f"Labels: {labels}")
                        print(f"Loss: {loss.item()}")
                    '''
                    # use bfloat16 for the expert input
                    expert_input = model.get_expert_input(0, 0)
                    expert_input = torch.sigmoid(expert_input).to(torch.bfloat16)
                    features = features.to(torch.bfloat16)
                    loss = nn.MSELoss()(expert_input, features).to(torch.bfloat16)

                    accelerator.backward(loss)
                    optimizer.step()

                    train_loss += loss.item()

                    if (index + 1) % save_period == 0 and accelerator.is_local_main_process:
                        state = accelerator.get_state_dict(model)
                        accelerator.save(state, f"{save_dir}/model_{epoch + 1}_{index + 1}.pt")

                        # model.save_checkpoint(save_dir)

                    if accelerator.is_local_main_process:
                        wandb.log({"loss": loss.item()})
        train_avg_loss = train_loss / len(training_dataloader)

        # testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for index, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                                     disable=not accelerator.is_main_process):
                tokens = batch['tokens']
                attn_mask = batch['attention_mask']
                features = batch['features']
                outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)

                # use bfloat16 for the expert input
                expert_input = model.get_expert_input(0, 0)
                features = features.to(torch.bfloat16)
                loss = nn.MSELoss()(expert_input, features).to(torch.bfloat16)
                test_loss += loss.item()

                if accelerator.is_local_main_process:
                    wandb.log({"test_loss": loss.item()})

        if accelerator.is_local_main_process:
            test_avg_loss = test_loss / len(test_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")


    if save_dir is not None and accelerator.is_local_main_process:
        state = accelerator.get_state_dict(model)
        accelerator.save(state, f"{save_dir}/model_final.pt")

    wandb.finish()


if __name__ == '__main__':
    mistral_models_path = "model/llm"  # path to the model cache
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # model name
    expert_dir = "model/experts/syn_adult_mlp.pth"  # path to the domain expert
    save_dir = "model/llm/domain_encoder/"  # path to save the model
    ffn_hidden_size = 1024  # hidden size of the feed-forward neural network in the Domain Expert
    num_heads = 8  # number of heads in the multi-head self-attention layer in the MappingBlock
    dropout = 0.1  # dropout rate in the MappingBlock
    use_norm = True  # whether to use layer normalization after attention in the MappingBlock
    expert_input_size = 105  # input size of the Domain Expert
    expert_output_size = 1  # output size of the Domain Expert
    top_k = 256  # number of tokens to route to the Domain Expert
    max_new_tokens = 300  # maximum number of tokens to generate
    system_prompt = "Respond the user's question in only one word: Yes or No."
    syn_text_data_file = "data/adult/text/syn_adult_train_text.csv"
    real_text_data_file = "data/adult/text/syn_adult_test_text.csv"
    dummy_data_file = None
    syn_tab_data_file = "data/adult/clean/syn_adult_train.csv"
    batch_size = 128  # batch size
    gradient_accumulation_steps = 8  # number of steps to accumulate gradients, it should divide batch_size
    num_epochs = 3  # number of epochs
    save_period = 3000000  # save the model every save_period steps
    learning_rate = 2e-3  # learning rate
    data_json = "src/dataset/adult/adult.json"
    mapping_hidden_size = 32
    train_domain_mistral(mistral_models_path, model_name, expert_dir, system_prompt, save_dir, syn_text_data_file, real_text_data_file,
                         ffn_hidden_size, num_heads, dropout, use_norm, expert_input_size, expert_output_size, top_k,
                         max_new_tokens, batch_size, gradient_accumulation_steps, num_epochs, save_period,
                         learning_rate, data_json, None, None, mapping_hidden_size)
