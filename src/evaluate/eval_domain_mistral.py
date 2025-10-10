"""
Evaluation script for Domain Mistral model with support for synthetic data and experts.

Usage examples:
1. Standard evaluation (original data + original expert):
   python src/evaluate/eval_domain_mistral.py --dataset titanic

2. Evaluate with synthetic text and expert:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --use_syn

3. Evaluate with synthetic data and custom suffix:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --use_syn --syn_data_suffix "filtered"

4. Ablation study with synthetic data:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --ablation_suffix "_age20" --use_syn

5. GBDT expert evaluation with synthetic data:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --expert_type gbdt --use_syn

6. Evaluate model trained on Nebius-generated synthetic data:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --use_nebius_model

7. Ablation study with Nebius-trained model:
   python src/evaluate/eval_domain_mistral.py --dataset titanic --ablation_suffix "_age20" --use_nebius_model
"""

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
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.analysis.TokenMonitor import DomainMistralTokenMonitor
from src.model.DomainExpert import DomainExpert


def evaluate_domain_mistral(mistral_models_path, model_name, dataset_type, expert_dir, encoder_dir,
                            encoder_tokenizer_name, ffn_hidden_size, num_tokens, num_heads, dropout, use_norm,
                            expert_input_size, expert_output_size, max_new_tokens, batch_size, n_instances,
                            system_prompt,
                            model_state_dict_dir, output_file, mapping_hidden_size,
                            expert_input_size_scaled, dataset_columns, tune_lora, use_baseline, layer_to_add,
                            direct_input, expert_weight, ablation_suffix=None, disable_emb_translate=False,
                            llamdex_padding='gaussian', expert_type="mlp", init_expert_type=None, init_expert_dir=None,
                            use_syn=False, syn_data_suffix="", use_nebius_model=False):
    # Handle ablation studies
    dataset_suffix = ""
    if ablation_suffix:
        dataset_suffix = f"_ablation{ablation_suffix}"
        print(f"Evaluating with ablation suffix: {ablation_suffix}")
    
    # Handle synthetic evaluation scenarios
    if use_syn:
        print(f"Synthetic evaluation mode:")
        print(f"  - Using synthetic text and expert")
        if syn_data_suffix:
            print(f"  - Synthetic data suffix: {syn_data_suffix}")
    
    # Handle Nebius model evaluation
    if use_nebius_model:
        print(f"Nebius model evaluation mode:")
        print(f"  - Using model trained on Nebius-generated synthetic data")
    
    if expert_dir is None:
        expert_prefix = "syn_" if use_syn else ""
        # For both ablation and non-ablation studies, use the same expert file
        # Age-specific behavior is handled by age-specific JSON configurations and scalers
        if expert_type == "mlp":
            expert_dir = f"model/experts/{expert_prefix}{dataset_type}_mlp.pth"
        elif expert_type == "gbdt":
            expert_dir = f"model/experts/{expert_prefix}{dataset_type}_gbdt.json"
        else:
            raise ValueError(f"Invalid expert type: {expert_type}")

    if encoder_dir is None:
        # encoder_dir = f"model/llm/encoder_{dataset_type}"
        encoder_dir = "roberta-large"

    if encoder_tokenizer_name is None:
        encoder_tokenizer_name = "roberta-large"

    if model_state_dict_dir is None:
        # Add layer suffix and nebius suffix to match training script expectations
        layer_suffix = f"_layer{layer_to_add[0]}" if layer_to_add and len(layer_to_add) > 0 else ""
        nebius_suffix = "_nebius" if use_nebius_model else ""
        model_state_dict_dir = f"model/llm/domain_mistral_{dataset_type}{dataset_suffix}{nebius_suffix}{layer_suffix}/model_final.pt"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        if ablation_suffix:
            # Include layer information in ablation output filename to match bash script expectations
            layer_suffix = f"_layer{layer_to_add[0]}" if layer_to_add and len(layer_to_add) > 0 else ""
            nebius_suffix = "_nebius" if use_nebius_model else ""
            output_file = f"data/{dataset_type}/res/domain_mistral{dataset_suffix}{nebius_suffix}{layer_suffix}_test.csv"
        elif use_syn:
            # Add synthetic indicator to output filename
            syn_suffix = "syn"
            if syn_data_suffix:
                syn_suffix += f"_{syn_data_suffix}"
            output_file = f"data/{dataset_type}/res/domain_mistral_{syn_suffix}_test.csv"
        elif use_nebius_model:
            # Add nebius indicator to output filename
            output_file = f"data/{dataset_type}/res/domain_mistral_nebius_test.csv"
        else:
            output_file = f"data/{dataset_type}/res/domain_mistral_test.csv"

    # Construct data file path with synthetic support
    if use_syn:
        if ablation_suffix:
            # Use synthetic ablation text data
            if syn_data_suffix:
                data_file = f"data/{dataset_type}/text_ablation/syn_{dataset_type}{dataset_suffix}_{syn_data_suffix}_test_text_filtered.csv"
            else:
                data_file = f"data/{dataset_type}/text_ablation/syn_{dataset_type}{dataset_suffix}_test_text_filtered.csv"
            print(f"Using synthetic ablation test data: {data_file}")
        else:
            # Use synthetic text data
            if syn_data_suffix:
                data_file = f"data/{dataset_type}/text/syn_{dataset_type}_{syn_data_suffix}_test_text_filtered.csv"
            else:
                data_file = f"data/{dataset_type}/text/syn_{dataset_type}_test_text_filtered.csv"
            print(f"Using synthetic test data: {data_file}")
    elif ablation_suffix:
        # For ablation studies, we can evaluate on either ablation test set or original test set
        # Default to original test set to compare performance across different age ranges
        data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
        print(f"Using original test data for ablation evaluation: {data_file}")
    else:
        data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
    
    print(f"Model checkpoint: {model_state_dict_dir}")
    print(f"Expert checkpoint: {expert_dir}")
    print(f"Data file: {data_file}")
    print(f"Output file: {output_file}")
    
    # Check if model file exists
    if not os.path.exists(model_state_dict_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {model_state_dict_dir}")
    if not use_baseline and not os.path.exists(expert_dir):
        raise FileNotFoundError(f"Expert checkpoint not found: {expert_dir}")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
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
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    if use_baseline:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    else:
        model = DomainMistralForCausalLM.from_pretrained_mistral(model_name,
                                                                 cache_dir=mistral_models_path,
                                                                 torch_dtype=torch.bfloat16,
                                                                 tokenizer=tokenizer)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    embed_size = model.config.hidden_size

    class_tokens = get_class_tokens(dataset_type)

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
                              num_tokens=num_tokens, cache_dir=mistral_models_path, encoder_model_id=encoder_tokenizer_name,
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

    # monitor = DomainMistralTokenMonitor(model, tokenizer)

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
                                attention_mask=attn_mask, past_key_values=None, use_cache=False, expert_weight=expert_weight)
            else:
                outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False, expert_weight=expert_weight)
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
    parser = argparse.ArgumentParser(description='Evaluate the Domain Mistral model.')

    parser.add_argument('--mistral_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
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
    parser.add_argument("--expert_weight", type=float, default=1.0, help="Weight of the expert output")
    parser.add_argument('--ablation_suffix', type=str, default=None,
                        help='Suffix for ablation study (e.g., "_age20" for age ablation)')
    parser.add_argument('--expert_type', type=str, default='mlp', choices=['mlp', 'gbdt'],
                        help='Type of expert model to use (mlp or gbdt)')
    parser.add_argument('--use_syn', action='store_true', help='Whether to use synthetic text and expert')
    parser.add_argument('--syn_data_suffix', type=str, default='', help='Suffix for synthetic data')
    parser.add_argument('--use_nebius_model', action='store_true', 
                        help='Whether to use model trained on Nebius-generated synthetic data')

    args = parser.parse_args()

    print(args)

    evaluate_domain_mistral(mistral_models_path=args.mistral_models_path,
                            model_name=args.model_name,
                            dataset_type=args.dataset,
                            expert_dir=args.expert_dir,
                            encoder_dir=args.encoder_dir,
                            encoder_tokenizer_name=args.encoder_tokenizer,
                            ffn_hidden_size=args.ffn_hidden_size,
                            num_tokens=args.num_tokens,
                            num_heads=args.num_heads,
                            dropout=args.dropout,
                            use_norm=args.use_norm,
                            expert_input_size=args.expert_input_size,
                            expert_output_size=args.expert_output_size,
                            max_new_tokens=args.max_new_tokens,
                            batch_size=args.batch_size,
                            n_instances=args.n_instances,
                            system_prompt=args.system_prompt,
                            model_state_dict_dir=args.model_state_dict_dir,
                            output_file=args.output_file,
                            mapping_hidden_size=args.mapping_hidden_size,
                            expert_input_size_scaled=args.expert_input_size_scaled,
                            dataset_columns=args.dataset_columns,
                            tune_lora=args.tune_lora,
                            use_baseline=args.baseline,
                            layer_to_add=args.layer,
                            direct_input=args.direct_input,
                            expert_weight=args.expert_weight,
                            ablation_suffix=args.ablation_suffix,
                            expert_type=args.expert_type,
                            use_syn=args.use_syn,
                            syn_data_suffix=args.syn_data_suffix,
                            use_nebius_model=args.use_nebius_model)

