import sys
import os
import argparse
import pandas as pd
import json
import openai
from typing import List, Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import interpret_to_human_readable


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


def get_label_name_from_dataset(dataset: str, dataset_json_path: str = None) -> str:
    """Get the label name from the dataset json file."""
    if dataset_json_path is None:
        dataset_json_path = os.path.join(os.path.dirname(__file__), f"../../dataset/{dataset}/{dataset}.json")
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    label_name = dataset_json['y']['name']
    return label_name


def download_batch_results(client: openai.OpenAI, batch_id: str) -> List[Dict]:
    """Download and parse the batch results."""
    print(f"📋 Retrieving batch information for: {batch_id}")
    
    # Get batch info
    batch = client.batches.retrieve(batch_id)
    print(f"   Status: {batch.status}")
    
    if hasattr(batch, 'request_counts'):
        completed = getattr(batch.request_counts, 'completed', 'N/A')
        failed = getattr(batch.request_counts, 'failed', 'N/A')
        total = getattr(batch.request_counts, 'total', 'N/A')
        print(f"   Completed: {completed}/{total}, Failed: {failed}")
    
    if batch.status not in ["completed", "done"]:
        raise Exception(f"Batch is not completed. Current status: {batch.status}")
    
    if not batch.output_file_id:
        raise Exception("No output file available for completed batch")
    
    print(f"📥 Downloading batch results from file: {batch.output_file_id}")
    
    # Download the results file
    file_response = client.files.content(batch.output_file_id)
    results_content = file_response.content.decode('utf-8')
    
    # Parse the JSONL results
    results = []
    for line_num, line in enumerate(results_content.strip().split('\n')):
        if line:
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Failed to parse line {line_num + 1}: {e}")
                continue
    
    print(f"✅ Downloaded {len(results)} results")
    return results


def parse_batch_results_robust(results: List[Dict], original_data: pd.DataFrame, 
                              dataset: str, query_prefix: str) -> pd.DataFrame:
    """Parse the batch results and create the final DataFrame with robust error handling."""
    
    # Get label name
    label_name = get_label_name_from_dataset(dataset)
    
    # Convert original data to readable format for query reconstruction
    df_readable = interpret_to_human_readable(original_data.copy(), dataset)
    readable_records = df_readable.to_dict('records')
    
    # Create result DataFrame
    new_df = pd.DataFrame(columns=original_data.columns.tolist() + ['prompt_to_llm'] + ['formatted_text'])
    
    # Sort results by custom_id to match original order
    results_by_id = {}
    for result in results:
        custom_id = result.get('custom_id', '')
        results_by_id[custom_id] = result
    
    print(f"🔍 Processing {len(original_data)} rows...")
    
    success_count = 0
    error_count = 0
    
    for i in range(len(original_data)):
        custom_id = f"request-{i}"
        original_row = original_data.iloc[i]
        
        # Reconstruct the query string
        if i < len(readable_records):
            readable_row = readable_records[i]
            query_str = query_prefix + " ".join(
                f'#{key}: {value}' for key, value in readable_row.items() 
                if (key != label_name and not pd.isnull(value))
            )
        else:
            query_str = f"Error: Could not reconstruct query for row {i}"
        
        # Get the generated text
        if custom_id in results_by_id:
            result = results_by_id[custom_id]
            generated_text = extract_generated_text(result, custom_id)
            if not generated_text.startswith("Error:"):
                success_count += 1
            else:
                error_count += 1
        else:
            generated_text = f"Error: Result not found for {custom_id}"
            error_count += 1
            print(f"⚠️ Missing result for {custom_id}")
        
        # Add to DataFrame
        new_row = list(original_row) + [query_str, generated_text]
        new_df.loc[len(new_df)] = new_row
    
    print(f"📊 Processing complete:")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📈 Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    
    return new_df


def extract_generated_text(result: Dict, custom_id: str) -> str:
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


def download_and_process_batch(batch_id: str, original_input_file: str, output_file: str,
                              dataset: str, query_prefix: str, api_key_file: str = "Nebius_APIkey"):
    """Main function to download and process batch results."""
    
    print(f"🚀 Starting batch result download and processing")
    print(f"   Batch ID: {batch_id}")
    print(f"   Dataset: {dataset}")
    print(f"   Original input: {original_input_file}")
    print(f"   Output file: {output_file}")
    print()
    
    # Setup client
    api_key = load_nebius_api_key(api_key_file)
    client = setup_nebius_client(api_key)
    
    # Load original data
    print(f"📂 Loading original data from: {original_input_file}")
    if not os.path.exists(original_input_file):
        raise FileNotFoundError(f"Original input file not found: {original_input_file}")
    
    original_data = pd.read_csv(original_input_file)
    print(f"   Loaded {len(original_data)} rows")
    
    # Download results
    results = download_batch_results(client, batch_id)
    
    # Process results
    final_df = parse_batch_results_robust(results, original_data, dataset, query_prefix)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved to: {output_file}")
    print(f"   Total rows: {len(final_df)}")
    print(f"🎉 Batch processing completed successfully!")


if __name__ == '__main__':
    # Query prefixes (same as in original script)
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
    
    parser = argparse.ArgumentParser(description='Download and process results from a completed Nebius batch job')
    parser.add_argument('--batch_id', type=str, required=True, help='Nebius batch ID to download')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--input_file', type=str, required=True, help='path to original input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='path to save processed results')
    parser.add_argument('--api_key_file', type=str, default='Nebius_APIkey', help='file containing the Nebius API key')
    
    args = parser.parse_args()
    
    if args.dataset not in QUERY_PREFIXES:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported: {list(QUERY_PREFIXES.keys())}")
    
    query_prefix = QUERY_PREFIXES[args.dataset]
    
    download_and_process_batch(
        batch_id=args.batch_id,
        original_input_file=args.input_file,
        output_file=args.output_file,
        dataset=args.dataset,
        query_prefix=query_prefix,
        api_key_file=args.api_key_file
    )
    
    print(f"""
=== Usage Example ===
python src/preprocess/gentext/download_nebius_results.py \\
    --batch_id batch_5a661956-34a6-4677-a2bc-5c473f931294 \\
    --dataset titanic \\
    --input_file data/titanic/clean/syn_titanic_train.csv \\
    --output_file data/titanic/text/syn_titanic_train_text_filtered_nebius.csv
""") 