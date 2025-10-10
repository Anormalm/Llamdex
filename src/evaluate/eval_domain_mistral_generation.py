#!/usr/bin/env python3
"""
Evaluation script for Domain Mistral Generation models.
Evaluates free-form text generation instead of classification.
"""

import sys
import os
import json
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.model.DomainExpert import DomainExpert
from src.dataset.utils_generation import get_dataset_for_evaluation_generation
from src.preprocess.DataScaler import TableScaler


class GenerationEvaluator:
    """Evaluate generation models by extracting answers from free-form text."""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
    
    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """Extract the actual answer from a free-form response."""
        response_lower = response.lower()
        
        # For binary classification datasets - simple detection
        if self.dataset_type in ['adult', 'titanic', 'bank_marketing']:
            # Check if "yes" appears in the text
            if "yes" in response_lower:
                return "1"  # Yes = 1
            # Check if "no" appears in the text
            elif "no" in response_lower:
                return "0"  # No = 0
        
        # For multiclass datasets - not supported right now
        else:
            print(f"Warning: Multiclass evaluation not supported for dataset {self.dataset_type}")
            return None
        
        return None
    
    def evaluate_responses(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Evaluate generated responses against ground truth."""
        extracted_predictions = []
        successful_extractions = 0
        
        for i, response in enumerate(predictions):
            extracted = self.extract_answer_from_response(response)
            if extracted is not None:
                extracted_predictions.append(extracted)
                successful_extractions += 1
            else:
                # If extraction fails, predict the most common class
                if self.dataset_type in ['adult', 'titanic', 'bank_marketing']:
                    extracted_predictions.append("0")  # Default to No
                else:
                    # For multiclass, skip this prediction
                    extracted_predictions.append("0")  # Default value, but won't be used
        
        # Convert ground truth to string format for comparison
        ground_truth_str = [str(int(float(gt))) for gt in ground_truth]
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth_str, extracted_predictions)
        
        # Calculate extraction success rate
        extraction_rate = successful_extractions / len(predictions)
        
        # Generate classification report
        try:
            report = classification_report(ground_truth_str, extracted_predictions, 
                                         output_dict=True, zero_division=0)
        except:
            report = {}
        
        return {
            'accuracy': accuracy,
            'extraction_rate': extraction_rate,
            'successful_extractions': successful_extractions,
            'total_samples': len(predictions),
            'classification_report': report
        }


def evaluate_model(model_path: str, dataset_type: str, data_file: str, 
                  model_name: str, mistral_models_path: str, expert_dir: str,
                  encoder_model: str, max_new_tokens: int = 100, batch_size: int = 8,
                  dataset_json: str = None, system_prompt: str = None):
    """Evaluate a generation model on the given dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if dataset is supported for evaluation
    supported_datasets = ['adult', 'titanic', 'bank_marketing']
    if dataset_type not in supported_datasets:
        raise ValueError(f"Dataset {dataset_type} not supported for generation evaluation. "
                        f"Only binary classification datasets are supported: {supported_datasets}")
    
    # Load dataset info
    dataset_info_json = "src/dataset/dataset_info_generation.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info = json.load(f)
    
    if dataset_type not in dataset_info:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    if system_prompt is None:
        system_prompt = dataset_info[dataset_type]['system_prompt']
    
    if dataset_json is None:
        dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=mistral_models_path)
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token
    
    # Load model
    print("Loading model...")
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        model_name, cache_dir=mistral_models_path, torch_dtype=torch.bfloat16, tokenizer=tokenizer
    )
    
    # Load expert
    expert_input_scaler = TableScaler(dataset_json).tensor_map
    expert_input_size = dataset_info[dataset_type]['expert_input_size']
    expert_output_size = dataset_info[dataset_type]['expert_output_size']
    
    expert = DomainExpert(
        embed_size=model.config.hidden_size, 
        ffn_hidden_size=512,
        expert_input_size=expert_input_size,
        expert_output_size=expert_output_size, 
        expert_dir=expert_dir,
        num_heads=8, dropout=0.0, use_norm=True, 
        max_length=max_new_tokens,
        dataset_json=dataset_json, dataset_columns=None, expert_input_size_scaled=None,
        mapping_hidden_size=64, num_tokens=10, cache_dir=mistral_models_path,
        encoder_model_id=encoder_model, expert_input_scaler=expert_input_scaler
    )
    
    # Add expert to model
    layer_to_add = list(range(len(model.model.layers)))
    model.add_expert_(expert, layer_to_add)
    
    # Load trained weights
    if model_path and os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No trained weights loaded, using base model")
    
    model = model.to(device)
    model.eval()
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = get_dataset_for_evaluation_generation(dataset_type, data_file, dataset_json)
    
    # Generate responses
    print("Generating responses...")
    predictions = []
    ground_truth = []
    
    evaluator = GenerationEvaluator(dataset_type)
    
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        batch_end = min(i + batch_size, len(eval_dataset))
        batch_items = [eval_dataset[j] for j in range(i, batch_end)]
        
        batch_texts = [item['text'] for item in batch_items]
        batch_features = torch.stack([item['features'] for item in batch_items]).to(device)
        batch_expected = [item['expected_response'] for item in batch_items]
        
        # Generate responses for batch
        for j, (text, features) in enumerate(zip(batch_texts, batch_features)):
            # Prepare input messages
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            else:
                messages = [{"role": "user", "content": text}]
            
            input_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_tokens,
                    expert_inputs=(features.unsqueeze(0),),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            response_tokens = outputs[0][input_tokens.shape[1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            predictions.append(response)
            
            # Extract ground truth from expected response
            ground_truth_label = evaluator.extract_answer_from_response(batch_expected[j])
            if ground_truth_label is None:
                # Fallback: use original label format
                if dataset_type in ['adult', 'titanic', 'bank_marketing']:
                    # Try to extract from dataset
                    gt_item = eval_dataset.raw_data[i + j]
                    answer_col = dataset_info[dataset_type]['answer_column']
                    ground_truth_label = str(int(float(gt_item[answer_col])))
                else:
                    # For categorical, assume A=0, B=1, etc.
                    gt_item = eval_dataset.raw_data[i + j]
                    answer_col = dataset_info[dataset_type]['answer_column']
                    ground_truth_label = str(int(float(gt_item[answer_col])))
            
            ground_truth.append(ground_truth_label)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    results = evaluator.evaluate_responses(predictions, ground_truth)
    
    # Print results
    print(f"\n=== Evaluation Results for {dataset_type} ===")
    print(f"Total samples: {results['total_samples']}")
    print(f"Successful extractions: {results['successful_extractions']}")
    print(f"Extraction rate: {results['extraction_rate']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    if results['classification_report']:
        print("\nClassification Report:")
        if 'macro avg' in results['classification_report']:
            macro_avg = results['classification_report']['macro avg']
            print(f"Macro F1: {macro_avg['f1-score']:.4f}")
            print(f"Macro Precision: {macro_avg['precision']:.4f}")
            print(f"Macro Recall: {macro_avg['recall']:.4f}")
    
    # Save detailed results
    output_file = f"evaluation_results_generation_{dataset_type}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'dataset_type': dataset_type,
            'model_path': model_path,
            'results': results,
            'sample_predictions': predictions[:10],  # Save first 10 predictions as examples
            'sample_ground_truth': ground_truth[:10]
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Domain Mistral Generation Model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--data_file", type=str, required=True, help="Path to evaluation data file")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model name")
    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to model cache")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to expert model")
    parser.add_argument("--encoder_model", type=str, default="roberta-large", help="Encoder model")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--dataset_json", type=str, default=None, help="Dataset JSON config")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt override")
    
    args = parser.parse_args()
    
    # Set default expert_dir if not provided
    if args.expert_dir is None:
        args.expert_dir = f"model/experts/syn_{args.dataset}_mlp.pth"
    
    evaluate_model(
        model_path=args.model_path,
        dataset_type=args.dataset,
        data_file=args.data_file,
        model_name=args.model_name,
        mistral_models_path=args.mistral_models_path,
        expert_dir=args.expert_dir,
        encoder_model=args.encoder_model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dataset_json=args.dataset_json,
        system_prompt=args.system_prompt
    )


if __name__ == "__main__":
    main() 