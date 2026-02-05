import os
import sys
import json
import time
import openai
import requests
from tqdm import tqdm
from eval_score import eval_score, show_results

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


API_KEY = "sk-oUO8lmP8x5XsySaYtYLk6Xmb9O99jeXU32w2fZd7wGpoo3l9" 
API_URL = "https://api.2077ai.org/v1/messages"
REQUEST_TIMEOUT = 180  # Timeout for the API request in seconds

MODEL_NAME_MAPPING = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-4o": "gpt-4o-2024-05-13",
}
# The model you want to use for the judgments.
MODEL_TO_USE = "gpt-4o"

def process_entry_with_gpt(entry, query_key='question', prompt='', model=MODEL_TO_USE, test=False):
    """
    Sends a request to the GPT API for a given entry and handles exceptions.

    Args:
        entry (dict): A dictionary containing the data for a single prompt.
        answer_key (str): The key to access the model's response in the entry.
        model (str): The model to use for the completion.
        test (bool): If True, sends a simple test message instead of processing the entry.

    Returns:
        dict or None: The parsed JSON response from the API, or None if an error occurred.
    """
    # --- Setup API Client ---
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # --- Construct Messages ---
    if test:
        messages = [{"role": "user", "content": "Hello, this is a test. Are you ready?"}]
        print("Sending a simple test request...")
    else:
        # Extract query and response from the entry
        query = entry[query_key]
        pred = entry['pred']
        
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(query, pred)
            }
        ]

    payload = {
        "model": model,
        "max_tokens": 4096, # Adjusted for safety, Claude 3.5 Sonnet has a 200K context window
        "messages": messages
    }
    
    try:
        # --- Make API Call ---
        response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        response_data = response.json()
        return response_data

    # --- Exception Handling ---
    except openai.APITimeoutError:
        print(f"❌ Error: Request timed out after {REQUEST_TIMEOUT} seconds.", file=sys.stderr)
    except openai.APIConnectionError as e:
        print(f"❌ Error: Could not connect to the API at {API_URL}. Check URL/network.", file=sys.stderr)
    except openai.RateLimitError:
        print("❌ Error: Rate limit exceeded. Please check your plan and billing details.", file=sys.stderr)
    except openai.APIStatusError as e:
        print(f"❌ Error: API returned status {e.status_code}. Response: {e.response.text}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
    
    return None

def main(data_path):
    print("--- 🚀 Starting GPT API File Processor ---")

    # --- Run Initial API Test ---
    print("\n--- 🧪 Performing initial API connectivity test... ---")
    test_response = process_entry_with_gpt({}, test=True)
    if test_response:
        print("✔️ Success! API connection is working.")
    else:
        print("🚨 FATAL ERROR: API test failed. Please check your configuration and credentials. Exiting.", file=sys.stderr)
        sys.exit(1)

    output_path = data_path.replace('.jsonl', '_score.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(output_path):
        with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/prompt_for_answer_extraction.md') as f:
            prompt = f.read()

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f]
            print(f"Successfully loaded {len(all_data)} entries from {data_path}")
        except FileNotFoundError:
            print(f"🚨 FATAL ERROR: Input file not found at '{data_path}'. Please check the path.", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"🚨 FATAL ERROR: Could not parse JSON from '{data_path}'. Please ensure it's a valid JSONL file.", file=sys.stderr)
            sys.exit(1)
        
        # --- Check for Existing Output File ---
        finished_indexes = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    finished_entries = [json.loads(line) for line in f]
                finished_indexes = {entry.get('question') for entry in finished_entries}
                print(f"Found {len(finished_indexes)} already processed entries in {output_path}. Resuming...")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Warning: Could not read progress from {output_path}. Starting from scratch. Error: {e}", file=sys.stderr)

        # Filter out data that has already been processed
        data_to_process = [entry for entry in all_data if entry.get('question') not in finished_indexes]
        print(f"Total entries to process: {len(data_to_process)}")

        # ---  Process Entries ---
        print(f"\n--- 🔄 Starting processing of {len(data_to_process)} entries... ---")
        try:
            for entry in tqdm(data_to_process, desc="Processing Entries"):
                start_time = time.time()
                response_data = process_entry_with_gpt(entry, query_key='question', prompt=prompt)
                
                if response_data:
                    try:
                        # Extract the message content
                        message_content = response_data['content'][0]['text']
                        try:
                            pred_ans = message_content.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                            score = eval_score(entry['answer'], pred_ans, entry['answer_format'])
                        except:
                            pred_ans = "Failed to extract"
                            score = 0.0
                            
                        entry['extracted_answer'] = pred_ans
                        entry['score'] = score
                        # Write the successful result to the output file immediately
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                
                    except (KeyError, IndexError):
                        print(f"\n⚠️ Warning: Response for index {entry.get('question')} had an unexpected structure.", file=sys.stderr)
                else:
                    print(f"\n⚠️ Failed to get response for index {entry.get('question')}. Skipping.", file=sys.stderr)
                    # Optional: Add a delay before retrying the next one if failures are frequent
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n--- 🛑 User interrupted operation. Shutting down gracefully. ---")
            print("Progress has been saved. You can restart the script to resume.")

        print("\n--- ✅🏁 Processing Finished ---")
    else:
        print(f"Output file {output_path} already exists. Skipping processing.")
    
    print("Starting to evaluate the results...")
    with open(output_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
        
    show_results(samples, show_path=output_path.replace('.jsonl', '_results.txt'))
    print("Results evaluation completed. Results saved to:", output_path.replace('.jsonl', '_results.txt'))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="use GPT API to process a JSONL file and extract answers.")
    parser.add_argument("data_path", type=str, help="Path to the input JSONL file containing the data to process.")
    args = parser.parse_args()
    
    main(args.data_path)