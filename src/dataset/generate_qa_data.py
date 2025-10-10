#!/usr/bin/env python3
"""
Script to generate semi-free QA data from existing classification datasets.
Converts yes/no and letter classifications into free-form responses.
"""

import pandas as pd
import json
import os
import argparse
import random
from typing import Dict, List, Tuple


class ResponseGenerator:
    """Generate free-form responses that convey yes/no or classification meanings."""
    
    def __init__(self):
        # Templates for Yes/No responses
        self.yes_templates = [
            "Yes, based on the given information.",
            "The answer is yes.",
            "Yes, this appears to be the case.",
            "Based on the data provided, yes.",
            "The evidence suggests yes.",
            "Yes, that's correct.",
            "The answer would be yes.",
            "Yes, according to the information.",
            "That's affirmative.",
            "Yes, this is likely.",
        ]
        
        self.no_templates = [
            "No, based on the given information.",
            "The answer is no.",
            "No, this does not appear to be the case.",
            "Based on the data provided, no.",
            "The evidence suggests no.",
            "No, that's not correct.",
            "The answer would be no.",
            "No, according to the information.",
            "That's negative.",
            "No, this is unlikely.",
        ]
        
        # Templates for categorical responses (A, B, C, etc.)
        self.categorical_templates = [
            "The appropriate classification is {category}.",
            "Based on the information, the answer is {category}.",
            "The rating would be {category}.",
            "This falls into category {category}.",
            "The classification is {category}.",
            "I would categorize this as {category}.",
            "The appropriate grade is {category}.",
            "This belongs in group {category}.",
            "The evaluation is {category}.",
            "The proper category is {category}.",
        ]
        
    def generate_response(self, label: str, dataset_type: str) -> str:
        """Generate a free-form response based on the label and dataset type."""
        
        # Handle binary classification (Yes/No)
        if dataset_type in ['adult', 'titanic', 'bank_marketing']:
            if str(label).lower() in ['1', '1.0', 'yes', 'true']:
                return random.choice(self.yes_templates)
            else:
                return random.choice(self.no_templates)
        
        # Handle categorical classification (A, B, C, etc.)
        else:
            # Convert numeric labels to letters
            if isinstance(label, (int, float)):
                label = int(label)
                category = chr(ord('A') + label)
            else:
                category = str(label).upper()
                
            template = random.choice(self.categorical_templates)
            return template.format(category=category)


def convert_dataset_to_generation(input_file: str, output_file: str, dataset_type: str):
    """Convert a classification dataset to generation format."""
    
    print(f"Converting {input_file} to generation format...")
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Get answer column from dataset info
    dataset_info_file = "src/dataset/dataset_info.json"
    with open(dataset_info_file, 'r') as f:
        dataset_info = json.load(f)
    
    if dataset_type not in dataset_info:
        raise ValueError(f"Dataset type {dataset_type} not found in dataset_info.json")
    
    answer_column = dataset_info[dataset_type]["answer_column"]
    
    if answer_column not in df.columns:
        raise ValueError(f"Answer column {answer_column} not found in dataset")
    
    # Initialize response generator
    response_gen = ResponseGenerator()
    
    # Generate response_text column
    df['response_text'] = df[answer_column].apply(
        lambda x: response_gen.generate_response(x, dataset_type)
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the converted dataset
    df.to_csv(output_file, index=False)
    print(f"Saved generation dataset to {output_file}")
    print(f"Generated {len(df)} samples")


def process_all_datasets():
    """Process all datasets and convert them to generation format."""
    
    # Dataset types to process (only binary classification supported for now)
    # datasets = ['adult', 'titanic', 'bank_marketing']
    datasets = ['titanic']
    
    print("Note: Only processing binary classification datasets.")
    print("Multiclass datasets (wine_quality, abalone, nursery) are not supported for generation evaluation yet.")
    
    for dataset_type in datasets:
        print(f"\n=== Processing {dataset_type} ===")
        
        # Define input and output paths
        input_dir = f"data/{dataset_type}/text"
        output_dir = f"data/{dataset_type}/text_generation"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert training data (synthetic)
        train_input = f"{input_dir}/syn_{dataset_type}_train_text_filtered.csv"
        train_output = f"{output_dir}/syn_{dataset_type}_train_text_generation.csv"
        
        if os.path.exists(train_input):
            convert_dataset_to_generation(train_input, train_output, dataset_type)
        else:
            print(f"Warning: Training file {train_input} not found")
        
        # Convert test data (synthetic)
        test_input = f"{input_dir}/syn_{dataset_type}_test_text_filtered.csv"
        test_output = f"{output_dir}/syn_{dataset_type}_test_text_generation.csv"
        
        if os.path.exists(test_input):
            convert_dataset_to_generation(test_input, test_output, dataset_type)
        else:
            print(f"Warning: Test file {test_input} not found")
        
        # Convert real test data
        real_input = f"{input_dir}/{dataset_type}_test_text.csv"
        real_output = f"{output_dir}/{dataset_type}_test_text_generation.csv"
        
        if os.path.exists(real_input):
            convert_dataset_to_generation(real_input, real_output, dataset_type)
        else:
            print(f"Warning: Real test file {real_input} not found")


def main():
    parser = argparse.ArgumentParser(description="Generate QA data for sequence generation training")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--input_file", type=str, help="Input CSV file")
    parser.add_argument("--output_file", type=str, help="Output CSV file")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    
    args = parser.parse_args()
    
    if args.all:
        process_all_datasets()
    elif args.dataset and args.input_file and args.output_file:
        convert_dataset_to_generation(args.input_file, args.output_file, args.dataset)
    elif args.dataset:
        # Process single dataset with default paths
        dataset_type = args.dataset
        input_dir = f"data/{dataset_type}/text"
        output_dir = f"data/{dataset_type}/text_generation"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert all files for this dataset
        files_to_convert = [
            (f"{input_dir}/syn_{dataset_type}_train_text_filtered.csv", f"{output_dir}/syn_{dataset_type}_train_text_generation.csv"),
            (f"{input_dir}/syn_{dataset_type}_test_text_filtered.csv", f"{output_dir}/syn_{dataset_type}_test_text_generation.csv"),
            (f"{input_dir}/{dataset_type}_test_text.csv", f"{output_dir}/{dataset_type}_test_text_generation.csv")
        ]
        
        for input_file, output_file in files_to_convert:
            if os.path.exists(input_file):
                convert_dataset_to_generation(input_file, output_file, dataset_type)
            else:
                print(f"Warning: {input_file} not found")
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 