import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

import argparse
import json

from src.preprocess.DataScaler import TableScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn, get_dataset, get_class_tokens
from src.model.DomainLlamaModel import DomainLlamaForCausalLM
from src.model.DomainExpert import DomainExpert


def evaluate_domain_llama(llama_models_path, model_name, dataset_type, expert_dir, encoder_dir,
                          encoder_tokenizer_name, ffn_hidden_size, num_tokens, num_heads, dropout, use_norm,
                          expert_input_size, expert_output_size, max_new_tokens, batch_size, n_instances,
                          system_prompt,
                          model_state_dict_dir, output_file, mapping_hidden_size,
                          expert_input_size_scaled, dataset_columns, tune_lora, use_baseline, layer_to_add,
                          direct_input):
    if expert_dir is None:
        expert_dir = f"model/experts/{dataset_type}_mlp.pth"

    if encoder_dir is None:
        # encoder_dir = f"model/llm/encoder_{dataset_type}"
        encoder_dir = "roberta-large"

    if encoder_tokenizer_name is None:
        encoder_tokenizer_name = "roberta-large"

    if model_state_dict_dir is None:
        model_state_dict_dir = f"model/llm/domain_llama_{dataset_type}/model_final.pt"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/domain_llama_test.csv"

    data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
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

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=llama_models_path, torch_dtype=torch.bfloat16)
    if use_baseline:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir=llama_models_path, torch_dtype=torch.bfloat16)
    else:
        model = DomainLlamaForCausalLM.from_pretrained_llama(model_name,
                                                             cache_dir=llama_models_path,
                                                             torch_dtype=torch.bfloat16,
                                                             tokenizer=tokenizer)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    embed_size = model.config.hidden_size

    class_tokens = get_class_tokens(dataset_type, model="llama2")

    if not use_baseline:
        model.num_tokens = num_tokens

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

        embed_pad = base_model.model.embed_tokens(torch.tensor([tokenizer.pad_token_id]))[0]

        lm_head = model.get_output_embeddings()
        norm = model.model.norm

        expert_input_scaler = TableScaler(dataset_json).tensor_map

        expert = DomainExpert(embed_size=embed_size, ffn_hidden_size=ffn_hidden_size,
                              expert_input_size=expert_input_size,
                              expert_output_size=expert_output_size, expert_dir=expert_dir, num_heads=num_heads,
                              dropout=dropout, use_norm=use_norm, max_length=max_new_tokens, dataset_json=dataset_json,
                              dataset_columns=None, expert_input_size_scaled=None,
                              mapping_hidden_size=mapping_hidden_size,
                              num_tokens=num_tokens, cache_dir=llama_models_path, encoder_model_id=encoder_tokenizer_name,
                              expert_input_scaler=expert_input_scaler)

        base_model.add_expert_(expert, [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in
                                        layer_to_add])

        print("Loading the model state dict...")

        model.load_state_dict(torch.load(model_state_dict_dir, map_location='cpu'))

        if tune_lora:
            model.base_model.model.replace_expert_(expert_dir)
        else:
            model.replace_expert_(expert_dir)

    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    print("Loading the dataset...")

    dataset = get_dataset(dataset_type, data_file, tokenizer=tokenizer,
                          dataset_json=dataset_json, n_instances=n_instances, system_prompt=system_prompt)

    custom_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id))

    print("Testing the model...")

    torch.cuda.empty_cache()

    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []

    expert_input_loss = 0
    with torch.no_grad():
        for batch in tqdm(custom_dataloader, desc="Evaluating"):
            tokens = batch['tokens'].to(torch.device('cuda'))
            attn_mask = batch['attention_mask'].to(torch.device('cuda'))
            labels = batch['labels'].to(torch.device('cuda'))
            features = batch['features'].to(torch.device('cuda')).to(torch.bfloat16)

            if direct_input:
                outputs = model(tokens, expert_inputs=(features,),
                                attention_mask=attn_mask, past_key_values=None, use_cache=False)
            else:
                outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)
                if not use_baseline:
                    expert_input = model.get_expert_input(layer_to_add[0], 0).to(torch.bfloat16)   # debug: get the expert input
                    expert_input_loss += torch.nn.MSELoss()(expert_input, features).item()

            logits = outputs.logits
            class_logits = logits[:, -1, class_tokens]
            probs = class_logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    print(all_preds)
    if not use_baseline:
        print(f"Expert input loss: {expert_input_loss / len(custom_dataloader):.4f}")

    num_classes = len(class_tokens)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy on the test set: {accuracy:.4f}")

    results_df = pd.DataFrame({
        'ground_truth': all_labels,
        'predictions': all_preds,
    })
    if num_classes == 2:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Weighted F1 Score on the test set: {f1:.4f}")

        y_bin = label_binarize(all_labels, classes=range(num_classes))
        if num_classes == 2:
            auc_roc = roc_auc_score(y_bin, all_probs[:, 1])
        else:
            auc_roc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')
        print(f"AUC-ROC Score on the test set: {auc_roc:.4f}")

        results_df['weighted_f1'] = f1
        results_df['auc_roc'] = auc_roc

    for i in range(num_classes):
        results_df[f'prob_class_{i}'] = all_probs[:, i]

    results_df['accuracy'] = accuracy


    results_df.to_csv(output_file, index=False)
    print(f"Predictions, probabilities, and ground truth saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Domain Llama model.')

    parser.add_argument('--llama_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model name')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--encoder_dir", type=str, default=None, help="Path to the encoder model, None for default")
    parser.add_argument("--encoder_tokenizer", type=str, default=None,
                        help="Path to the encoder tokenizer, None for default")
    parser.add_argument('--ffn_hidden_size', type=int, default=512,
                        help='Hidden size of the feed-forward neural network in the Domain Expert')
    parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens that expert appends to the input')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of heads in the multi-head self-attention layer in the MappingBlock')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in the MappingBlock')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='Whether to use layer normalization after attention in the MappingBlock')
    parser.add_argument('--expert_input_size', type=int, default=None, help='Input size of the Domain Expert')
    parser.add_argument('--expert_output_size', type=int, default=None, help='Output size of the Domain Expert')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of tokens to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt for the model')
    parser.add_argument('--model_state_dict_dir', type=str, default=None, help='Path to the model state dict')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--mapping_hidden_size', type=int, default=32, help='Hidden size for the mapping')
    parser.add_argument('--expert_input_size_scaled', type=int, default=None,
                        help='Scaled input size of the Domain Expert')
    parser.add_argument('--dataset_columns', type=str, default=None, help='Columns of the dataset')
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model')
    parser.add_argument("--layer", type=int, nargs='*', default=None,
                        help="Layers to add the expert, negative indexing supported")
    parser.add_argument("--direct_input", action='store_true',
                        help="Whether to directly use the ground truth features as input to the expert")

    args = parser.parse_args()

    evaluate_domain_llama(args.llama_models_path, args.model_name, args.dataset, args.expert_dir, args.encoder_dir,
                          args.encoder_tokenizer, args.ffn_hidden_size, args.num_tokens, args.num_heads,
                          args.dropout, args.use_norm, args.expert_input_size, args.expert_output_size,
                          args.max_new_tokens,
                          args.batch_size, args.n_instances, args.system_prompt, args.model_state_dict_dir,
                          args.output_file, args.mapping_hidden_size, args.expert_input_size_scaled,
                          args.dataset_columns, args.tune_lora, args.baseline, args.layer, args.direct_input)

