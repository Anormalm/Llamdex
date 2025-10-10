import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json

def syn_titanic_ablation(dataset_json_path, age_upper_bound, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    """
    Generate synthetic Titanic dataset with modified age range for ablation study.
    
    Args:
        dataset_json_path: Path to the original titanic.json file
        age_upper_bound: Upper bound for age range [0, age_upper_bound]
        n_samples: Number of samples to generate
        random_state: Random seed
        cat_na_ratio: Ratio of categorical values to set as NaN
        test_ratio: Ratio of test set size
    """
    # Load the original dataset configuration
    with open(dataset_json_path, 'r') as f:
        dataset_config = json.load(f)
    
    # Create a deep copy to avoid modifying the original
    modified_config = deepcopy(dataset_config)
    
    # Modify the age range to [0, age_upper_bound]
    if 'Age' in modified_config['X']:
        modified_config['X']['Age']['range'] = [0, age_upper_bound]
        print(f"Modified age range to [0, {age_upper_bound}]")
    else:
        raise KeyError("Age field not found in dataset configuration")
    
    # Generate synthetic data with modified configuration
    syn_titanic_table = make_classification_from_json(
        n_samples=n_samples, 
        dataset_json=modified_config, 
        random_state=random_state
    )
    
    # Get categorical columns
    cat_columns = [col for col in syn_titanic_table.columns if syn_titanic_table[col].dtype == 'category']

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_titanic_table.loc[syn_titanic_table.sample(frac=cat_na_ratio, random_state=random_state).index, col] = np.nan

    # Split the dataset into train and test
    test_size = int(len(syn_titanic_table) * test_ratio)
    syn_titanic_test = syn_titanic_table[:test_size]
    syn_titanic_train = syn_titanic_table[test_size:]

    return syn_titanic_train, syn_titanic_test, modified_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic Titanic dataset with modified age range for ablation study')
    parser.add_argument('-t', '--age_upper_bound', type=int, required=True, 
                        help='Upper bound for age range [0, t]')
    parser.add_argument('-n', '--n_samples', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('-s', '--random_state', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--cat_na_ratio', type=float, default=0.1,
                        help='Ratio of categorical values to set as NaN (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set size (default: 0.2)')
    parser.add_argument('--output_dir', type=str, default='data/titanic/clean',
                        help='Output directory for generated files (default: data/titanic/clean)')
    
    args = parser.parse_args()
    
    # Validate age_upper_bound
    if args.age_upper_bound <= 0:
        raise ValueError("Age upper bound must be positive")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Path to original titanic.json
    titanic_json_path = 'src/dataset/titanic/titanic.json'
    
    # Generate synthetic data with modified age range
    syn_titanic_train, syn_titanic_test, modified_config = syn_titanic_ablation(
        dataset_json_path=titanic_json_path,
        age_upper_bound=args.age_upper_bound,
        n_samples=args.n_samples,
        random_state=args.random_state,
        cat_na_ratio=args.cat_na_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save the datasets with age upper bound in filename
    train_filename = f'{args.output_dir}/syn_titanic_ablation_age{args.age_upper_bound}_train.csv'
    test_filename = f'{args.output_dir}/syn_titanic_ablation_age{args.age_upper_bound}_test.csv'
    config_filename = f'{args.output_dir}/syn_titanic_ablation_age{args.age_upper_bound}_config.json'
    
    syn_titanic_train.to_csv(train_filename, index=False)
    syn_titanic_test.to_csv(test_filename, index=False)
    
    # Save the modified configuration for reference
    with open(config_filename, 'w') as f:
        json.dump(modified_config, f, indent=2)
    
    print(f"Synthetic Titanic ablation dataset saved:")
    print(f"  Train: {train_filename} ({len(syn_titanic_train)} samples)")
    print(f"  Test: {test_filename} ({len(syn_titanic_test)} samples)")
    print(f"  Config: {config_filename}")
    print(f"  Age range: [0, {args.age_upper_bound}]")

# Example usages:
# python src/preprocess/syn/syn_titanic_ablation.py -t 50
# python src/preprocess/syn/syn_titanic_ablation.py -t 30 -n 5000 -s 42
# python src/preprocess/syn/syn_titanic_ablation.py -t 80 --output_dir data/titanic/ablation 