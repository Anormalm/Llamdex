import sys
import os
import argparse
import pandas as pd
import numpy as np
import random
import time
import json
from tqdm import tqdm
from typing import List, Dict, Optional
import openai

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import interpret_to_human_readable


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


def get_label_name_from_dataset(dataset: str, dataset_json_path: str = None) -> str:
    """
    Get the label name from the dataset json file.
    """
    if dataset_json_path is None:
        dataset_json_path = os.path.join(os.path.dirname(__file__), f"../../dataset/{dataset}/{dataset}.json")
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    label_name = dataset_json['y']['name']
    return label_name


def load_nebius_api_key(api_key_file: str = "Nebius_APIkey") -> str:
    """Load the Nebius API key from file."""
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    return api_key


def setup_nebius_client(api_key: str) -> openai.OpenAI:
    """Setup the Nebius API client using OpenAI-compatible interface."""
    client = openai.OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=api_key
    )
    return client


def create_batch_requests(data_batch: List[Dict], query_prefix: str, label_name: str, 
                         max_new_tokens: int = 300) -> List[Dict]:
    """Create batch requests for the Nebius API."""
    requests = []
    
    for i, row in enumerate(data_batch):
        # Shuffle the items like in original script
        items = list(row.items())
        random.shuffle(items)
        
        # Create the query string
        query_str = query_prefix + " ".join(
            f'#{key}: {value}' for key, value in items 
            if (key != label_name and not pd.isnull(value))
        )
        
        # Create the request in OpenAI batch format
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "google/gemma-2-2b-it",
                "messages": [{"role": "user", "content": query_str}],
                "max_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        requests.append(request)
    
    return requests


def submit_batch_job(client: openai.OpenAI, requests: List[Dict], 
                    batch_description: str = "Text generation batch") -> str:
    """Submit a batch job to Nebius API and return the batch ID."""
    # Create a temporary file for the batch requests
    batch_file_name = f"batch_requests_{int(time.time())}.jsonl"
    
    with open(batch_file_name, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    try:
        # Upload the batch file
        with open(batch_file_name, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create the batch job
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": batch_description}
        )
        
        print(f"✅ Batch job submitted successfully!")
        print(f"   Batch ID: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Requests: {len(requests)}")
        
        return batch.id
        
    finally:
        # Clean up the temporary file
        if os.path.exists(batch_file_name):
            os.remove(batch_file_name)


def monitor_batch_status(client: openai.OpenAI, batch_id: str, 
                        check_interval: int = 30) -> Dict:
    """Monitor the batch job status and return results when completed."""
    print(f"🔄 Monitoring batch job: {batch_id}")
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if batch.status == "validating":
                print(f"[{current_time}] ⏳ Status: VALIDATING - Checking batch requests...")
            elif batch.status == "in_progress":
                completed = batch.request_counts.completed if hasattr(batch.request_counts, 'completed') else 0
                total = batch.request_counts.total if hasattr(batch.request_counts, 'total') else 0
                print(f"[{current_time}] 🔄 Status: IN_PROGRESS - Completed: {completed}/{total}")
            elif batch.status == "running":
                print(f"[{current_time}] 🔄 Status: RUNNING - Processing requests...")
            elif batch.status == "finalizing":
                print(f"[{current_time}] 🔄 Status: FINALIZING - Processing results...")
            elif batch.status in ["completed", "done"]:
                print(f"[{current_time}] ✅ Status: {batch.status.upper()}!")
                if hasattr(batch, 'request_counts'):
                    completed = getattr(batch.request_counts, 'completed', 'N/A')
                    failed = getattr(batch.request_counts, 'failed', 'N/A')
                    print(f"   Completed requests: {completed}")
                    print(f"   Failed requests: {failed}")
                return batch
            elif batch.status == "failed":
                print(f"[{current_time}] ❌ Status: FAILED!")
                if hasattr(batch, 'errors') and batch.errors:
                    for error in batch.errors:
                        print(f"   Error: {error}")
                raise Exception(f"Batch job failed: {batch_id}")
            elif batch.status == "cancelled":
                print(f"[{current_time}] ⚠️ Status: CANCELLED!")
                raise Exception(f"Batch job was cancelled: {batch_id}")
            else:
                print(f"[{current_time}] ❓ Status: {batch.status}")
            
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"❌ Error monitoring batch: {e}")
            time.sleep(check_interval)


def download_batch_results(client: openai.OpenAI, batch: Dict) -> List[Dict]:
    """Download and parse the batch results."""
    if not batch.output_file_id:
        raise Exception("No output file available for completed batch")
    
    print(f"📥 Downloading batch results...")
    
    # Download the results file
    file_response = client.files.content(batch.output_file_id)
    results_content = file_response.content.decode('utf-8')
    
    # Parse the JSONL results
    results = []
    for line in results_content.strip().split('\n'):
        if line:
            result = json.loads(line)
            results.append(result)
    
    print(f"✅ Downloaded {len(results)} results")
    return results


def parse_batch_results(results: List[Dict], original_data_batch: List[Dict]) -> List[Dict]:
    """Parse the batch results and match them with original data."""
    parsed_results = []
    
    # Sort results by custom_id to match original order
    results_by_id = {result['custom_id']: result for result in results}
    
    for i, original_row in enumerate(original_data_batch):
        custom_id = f"request-{i}"
        
        if custom_id in results_by_id:
            result = results_by_id[custom_id]
            generated_text = extract_generated_text_robust(result, custom_id)
        else:
            generated_text = "Error: Result not found"
            print(f"⚠️ Missing result for {custom_id}")
        
        parsed_results.append({
            'original_data': original_row,
            'generated_text': generated_text
        })
    
    return parsed_results


def extract_generated_text_robust(result: Dict, custom_id: str) -> str:
    """Extract generated text from a single result with robust error handling."""
    try:
        # Check if there's a response
        if 'response' in result and result['response'] is not None:
            response = result['response']
            # Nebius API has choices directly in response, not in body
            if 'choices' in response:
                choices = response['choices']
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    if message and 'content' in message:
                        return message['content']
                    else:
                        return "Error: No message content in response"
                else:
                    return "Error: No choices in response"
            else:
                return "Error: No choices in response"
        
        # Check if there's an error
        elif 'error' in result and result['error'] is not None:
            error_info = result['error']
            if isinstance(error_info, dict):
                message = error_info.get('message', 'Unknown error')
                code = error_info.get('code', 'No code')
                return f"Error: {message} (Code: {code})"
            else:
                return f"Error: {str(error_info)}"
        
        else:
            return f"Error: Unknown result format for {custom_id}"
            
    except Exception as e:
        return f"Error: Exception parsing result - {str(e)}"


def tabular_to_text_nebius(dataset: str, input_file: str, output_file_path: str, 
                          query_prefix: str, max_new_tokens: int = 300, 
                          seed: int = 0, check_contradiction: bool = False, n_repeat: int = 1,
                          api_key_file: str = "Nebius_APIkey"):
    """
    Convert tabular data to text using Nebius API
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load API key and setup client
    api_key = load_nebius_api_key(api_key_file)
    client = setup_nebius_client(api_key)
    
    # Load and process data
    df = pd.read_csv(input_file)
    new_df = pd.DataFrame(columns=df.columns.tolist() + ['prompt_to_llm'] + ['formatted_text'])
    
    df_readable = interpret_to_human_readable(df.copy(), dataset)
    label_name = get_label_name_from_dataset(dataset)
    
    print(f"🚀 Starting text generation for {dataset}")
    print(f"   Input file: {input_file}")
    print(f"   Output file: {output_file_path}")
    print(f"   Total rows: {len(df_readable)}")
    print(f"   Processing as single batch: ALL DATA")
    print(f"   Repeats: {n_repeat}")
    print(f"   Model: google/gemma-2-2b-it")
    
    for repeat in range(n_repeat):
        if n_repeat > 1:
            print(f"\n🔄 Repeat {repeat + 1}/{n_repeat}")
        
        # Process all data as a single batch
        readable_batch = df_readable.to_dict('records')
        
        # Create batch requests for all data
        requests = create_batch_requests(
            readable_batch, query_prefix, label_name, max_new_tokens
        )
        
        # Submit batch job for all data
        batch_description = f"{dataset} text generation - complete dataset ({len(readable_batch)} samples)"
        batch_id = submit_batch_job(client, requests, batch_description)
        
        # Monitor batch status
        completed_batch = monitor_batch_status(client, batch_id)
        
        # Download and parse results
        results = download_batch_results(client, completed_batch)
        parsed_results = parse_batch_results(results, readable_batch)
        
        # Add results to dataframe
        for j, parsed_result in enumerate(parsed_results):
            original_row = df.iloc[j]
            query_str = query_prefix + " ".join(
                f'#{key}: {value}' for key, value in parsed_result['original_data'].items() 
                if (key != label_name and not pd.isnull(value))
            )
            
            new_row = list(original_row) + [query_str, parsed_result['generated_text']]
            new_df.loc[len(new_df)] = new_row
        
        # Save after each repeat
        new_df.to_csv(output_file_path, index=False)
        print(f"💾 Repeat {repeat + 1} completed: {len(new_df)} total rows processed")
    
    # Final save
    new_df.to_csv(output_file_path, index=False)
    print(f"✅ Text generation completed!")
    print(f"   Total generated samples: {len(new_df)}")
    print(f"   Output saved to: {output_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from tabular data using Nebius API')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--mode', type=str, required=True, help='train or test')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='number of times to repeat the generation')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('--synthetic', action='store_true',
                        help='generate text for synthetic data. if false, generate text for real data')
    parser.add_argument('--ablation_suffix', type=str, default='',
                        help='suffix for ablation study files (e.g., "_age40" for age ablation)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='custom input directory for data files (overrides default data/{dataset}/clean/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='custom output directory for text files (overrides default data/{dataset}/text)')
    parser.add_argument('--api_key_file', type=str, default='Nebius_APIkey',
                        help='file containing the Nebius API key')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='maximum tokens to generate per request')
    parser.add_argument('--check_interval', type=int, default=30,
                        help='interval in seconds to check batch status')
    
    args = parser.parse_args()

    if args.mode not in ['train', 'test']:
        raise ValueError("Mode must be either 'train' or 'test'")

    dataset = args.dataset
    
    # Set input and output directories with custom paths if provided
    dataset_clean_root = args.input_dir if args.input_dir else f'data/{dataset}/clean/'
    dataset_processed_root = args.output_dir if args.output_dir else f"data/{dataset}/text"

    os.makedirs(dataset_clean_root, exist_ok=True)
    os.makedirs(dataset_processed_root, exist_ok=True)

    # Generate text using Nebius API
    query_prefix = QUERY_PREFIXES[dataset]
    
    # Handle ablation studies
    if args.ablation_suffix:
        dataset_id = f'syn_{dataset}_ablation{args.ablation_suffix}' if args.synthetic else f'{dataset}_ablation{args.ablation_suffix}'
        print(f"Running ablation study with suffix: {args.ablation_suffix}")
    else:
        dataset_id = f'syn_{dataset}' if args.synthetic else dataset
    
    filtered_suffix = '_filtered' if args.synthetic else ''
    
    input_file = f'{dataset_clean_root}/{dataset_id}_{args.mode}.csv'
    output_file = f'{dataset_processed_root}/{dataset_id}_{args.mode}_text{filtered_suffix}_nebius.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Check if API key file exists
    if not os.path.exists(args.api_key_file):
        raise FileNotFoundError(f"API key file not found: {args.api_key_file}")
    
    print(f"🚀 Starting Nebius API text generation...")
    print(f"   Dataset: {dataset}")
    print(f"   Mode: {args.mode}")
    print(f"   API Key File: {args.api_key_file}")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    
    tabular_to_text_nebius(
        dataset=dataset,
        input_file=input_file,
        output_file_path=output_file,
        query_prefix=query_prefix,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
        check_contradiction=args.synthetic,  # Only check contradictions for synthetic data
        n_repeat=args.repeat,
        api_key_file=args.api_key_file
    )
    
    ablation_info = f" with ablation suffix '{args.ablation_suffix}'" if args.ablation_suffix else ""
    print(f"\n🎉 Text generation completed for {dataset} {args.mode} dataset{ablation_info}.")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")

    # Example usages:
    print(f"""
=== Usage Examples ===
# Generate text for real data
python src/preprocess/gentext/gentext_dataset_nebius.py -d adult --mode train -r 1 -s 0

# Generate text for synthetic data  
python src/preprocess/gentext/gentext_dataset_nebius.py -d titanic --mode test --synthetic

# Generate text for ablation study
python src/preprocess/gentext/gentext_dataset_nebius.py -d titanic --mode train --synthetic --ablation_suffix "_age20"

# Custom API key file and settings
python src/preprocess/gentext/gentext_dataset_nebius.py -d wine_quality --mode test --api_key_file custom_api_key.txt --max_tokens 400 --check_interval 60
""") 