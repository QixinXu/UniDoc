import json
import os
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Assuming these are available from your original script
from eval_score import eval_score, show_results 

# --- Configuration ---
# You may need to replace this with your actual API key
API_KEY = "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" 
# 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
API_URL = "https://api.2077ai.org/v1/messages"
REQUEST_TIMEOUT = 180  # Timeout for the API request in seconds
MAX_WORKERS = 5      # Number of concurrent threads for API calls

MODEL_TO_USE = "gpt-4o"

# Remove proxies to avoid connection issues
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

def extract_last_boxed_content(text):
    """
    提取字符串中最后一个 \boxed{} 里的内容。
    
    Args:
        text: 包含一个或多个 \boxed{} 的输入字符串。
        
    Returns:
        最后一个 \boxed{} 的内容，如果没有找到则返回 None。
    """
    matches = re.findall(r'\\boxed{(.*?)}', text)
    if matches:
        return matches[-1]  
    return None

def process_entry_with_gpt(entry, prompt, model=MODEL_TO_USE):
    """
    Sends a request to the GPT API for a single entry.
    This function is designed to be run by a worker thread.
    
    Args:
        entry (dict): A dictionary containing the data for a single prompt.
        prompt (str): The prompt text for the API call.
        model (str): The model to use for the completion.
        
    Returns:
        dict or None: The processed entry with score and extracted answer, or None if an error occurred.
    """
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    try:
        # Construct the messages payload
        query = entry.get('question', '')
        pred = entry.get('pred') if 'pred' in entry else entry.get('output')
        assert pred is not None, f"Entry missing 'pred' or 'output': {entry}"
        extracted_answer = extract_last_boxed_content(pred)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"\n\nQuestion:{query}\nAnalysis:{extracted_answer}\n"}
        ]
        print(messages)
        
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": messages
        }

        # Make API Call
        response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        message_content = response_data['content'][0]['text']
        
        # Extract and score the answer
        try:
            pred_ans = message_content.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            if pred_ans == "Not answerable":
                print(f"extracted answer: {extracted_answer}, but the model says it's not answerable")
            if pred_ans == "Failed to extract":
                print(f"extracted answer: {extracted_answer}, but the model failed to extract")
            score = eval_score(entry['answer'], pred_ans, entry['answer_format'])
        except (KeyError, IndexError):
            pred_ans = "Failed to extract"
            score = 0.0

        # Update the entry with new data
        entry['extracted_answer'] = pred_ans
        entry['score'] = score
        return entry

    except requests.exceptions.Timeout:
        print(f"\n❌ Error: Request timed out for {entry.get('question')}.", file=sys.stderr)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Error: API returned status {e.response.status_code} for {entry.get('question')}. Response: {e.response.text}", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred for {entry.get('question')}: {e}", file=sys.stderr)
    
    return None

def main(data_path, show_all=False):
    print("--- 🚀 Starting GPT API File Processor ---")

    # --- Run Initial API Test ---
    print("\n--- 🧪 Performing initial API connectivity test... ---")
    try:
        test_payload = {"model": MODEL_TO_USE, "max_tokens": 100, "messages": [{"role": "user", "content": "Hello, this is a test. Are you ready?"}]}
        response = requests.post(API_URL, headers={"x-api-key": API_KEY}, json=test_payload, timeout=10)
        response.raise_for_status()
        print("✔️ Success! API connection is working.")
    except Exception as e:
        print(f"🚨 FATAL ERROR: API test failed. Check config and credentials. Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    output_path = data_path.replace('.jsonl', '_score.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping processing.")
    else:
        with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/prompt_for_answer_extraction.md') as f:
            prompt = f.read()

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f]
                    
            print(f"Successfully loaded {len(all_data)} entries from {data_path}")
        except FileNotFoundError:
            print(f"🚨 FATAL ERROR: Input file not found at '{data_path}'.", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"🚨 FATAL ERROR: Could not parse JSON from '{data_path}'.", file=sys.stderr)
            sys.exit(1)
        
        # --- Check for Existing Output File to Resume ---
        finished_uids = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    finished_entries = [json.loads(line) for line in f]
                finished_uids = {entry.get('question') for entry in finished_entries}
                print(f"Found {len(finished_uids)} already processed entries. Resuming...")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Warning: Could not read progress from {output_path}. Starting from scratch. Error: {e}", file=sys.stderr)

        data_to_process = [entry for entry in all_data if entry.get('question') not in finished_uids]
        print(f"Total entries to process: {len(data_to_process)}")

        # --- Process entries using multi-threading ---
        print(f"\n--- 🔄 Starting processing of {len(data_to_process)} entries with {MAX_WORKERS} workers... ---")
        
        try:
            # Use ThreadPoolExecutor for concurrent API calls
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks to the executor
                future_to_entry = {executor.submit(process_entry_with_gpt, entry, prompt): entry for entry in data_to_process}

                # Process results as they are completed
                with open(output_path, 'a', encoding='utf-8') as f_out:
                    for future in tqdm(as_completed(future_to_entry), total=len(data_to_process), desc="Processing"):
                        try:
                            processed_entry = future.result()
                            if processed_entry:
                                f_out.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
                        except Exception as e:
                            print(f"\n⚠️ An exception occurred during processing: {e}", file=sys.stderr)
        
        except KeyboardInterrupt:
            print("\n\n--- 🛑 User interrupted operation. Shutting down gracefully. ---")
            print("Progress has been saved. You can restart the script to resume.")

        print("\n--- ✅🏁 Processing Finished ---")
    
    # --- Final Evaluation ---
    print("Starting to evaluate the results...")
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"🚨 FATAL ERROR: Output file {output_path} not found for evaluation.", file=sys.stderr)
        sys.exit(1)
        
    show_results(samples, show_path=output_path.replace('.jsonl', '_results.txt'), show_all=show_all)
    print("Results evaluation completed. Results saved to:", output_path.replace('.jsonl', '_results.txt'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Use GPT API to process a JSONL file and extract answers.")
    parser.add_argument("data_path", type=str, help="Path to the input JSONL file containing the data to process.")
    parser.add_argument("--show_all", action='store_true', help="Whether to show all results or only incorrect ones.")
    args = parser.parse_args()
    
    main(args.data_path, args.show_all)
