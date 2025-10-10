import sys
import os
import argparse


sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from util import tabular_to_text


QUERY_PREFIXES = {
    'adult': """Convert the following information of certain group of people into natural language, ensure and 
    double check that you do not miss any information, add some irrelevant context and ask if the salary of such
    group is above 50,000 USD in the end without answering, # please:\n""",

    'titanic': """Convert the following information about a Titanic passenger into natural language. Ensure and 
    double-check that you do not miss any information, add some irrelevant context, and ask if the passenger survived
    or not at the end without answering, # please:\n""",

    'wine_quality': """Convert the following information about a wine into natural language, ensure and double check 
    that you do not miss any information, add some irrelevant context and ask about the quality of the wine in the end 
    without answering, # please:\n""",

    'bank_marketing': """Convert the following information about a bank marketing client into natural language, ensure
    and double check that you do not miss any information, add some irrelevant context and ask if the client has subscribed
    a term deposit in the end without answering, # please:\n""",

    'abalone': """Convert the following information about an abalone into natural language. Ensure and double-check 
    that you do not miss any information, add some irrelevant context about marine life, and ask about the age group 
    of the abalone at the end without answering, # please:\n""",
    
    'nursery': """Convert the following information about a nursery school student into natural language. Ensure and
    double-check that you do not miss any information, add some irrelevant context about nursery schools, and ask about
    the evaluation of the student at the end without answering, # please:\n"""
}


if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Generate text from tabular data')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name')
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='number of times to repeat the generation')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('--synthetic', action='store_true',
                        help='generate text for synthetic data. if false, generate text for real data')
    parser.add_argument('--ablation_suffix', type=str, default='',
                        help='suffix for ablation study files (e.g., "_age40" for age ablation)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='custom input directory for data files (overrides default data/{dataset}/clean/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='custom output directory for text files (overrides default data/{dataset}/text)')
    args = parser.parse_args()

    if args.mode not in ['train', 'test']:
        raise ValueError("Mode must be either 'train' or 'test'")

    dataset = args.dataset
    
    # Set input and output directories with custom paths if provided
    dataset_clean_root = args.input_dir if args.input_dir else f'data/{dataset}/clean/'
    dataset_processed_root = args.output_dir if args.output_dir else f"data/{dataset}/text"
    dataset_json_path = f"src/dataset/{dataset}/{dataset}.json"

    os.makedirs(dataset_clean_root, exist_ok=True)
    os.makedirs(dataset_processed_root, exist_ok=True)

    # generate text from synthetic data
    query_prefix = QUERY_PREFIXES[dataset]
    
    # Handle ablation studies
    if args.ablation_suffix:
        dataset_id = f'syn_{dataset}_ablation{args.ablation_suffix}' if args.synthetic else f'{dataset}_ablation{args.ablation_suffix}'
        print(f"Running ablation study with suffix: {args.ablation_suffix}")
    else:
        dataset_id = f'syn_{dataset}' if args.synthetic else dataset
    
    filtered_suffix = '_filtered' if args.synthetic else ''
    
    input_file = f'{dataset_clean_root}/{dataset_id}_{args.mode}.csv'
    output_file = f'{dataset_processed_root}/{dataset_id}_{args.mode}_text{filtered_suffix}.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    tabular_to_text(dataset, input_file, output_file,
                    'model/llm', 'mistralai/Mistral-7B-Instruct-v0.3', query_prefix, 300,
                    seed=args.seed, batch_size=args.batch_size, check_contradiction=args.synthetic, n_repeat=args.repeat)
    
    ablation_info = f" with ablation suffix '{args.ablation_suffix}'" if args.ablation_suffix else ""
    print(f"Text generation completed for {dataset} {args.mode} dataset{ablation_info}.")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Example usages:
    # python src/preprocess/gentext/gentext.py -d adult --mode train -r 100 -s 0 -b 8
    # python src/preprocess/gentext/gentext.py -d titanic --mode test -r 1 --synthetic
