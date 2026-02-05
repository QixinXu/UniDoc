import os
import re
import sys
import json
import time
import openai
from tqdm import tqdm

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


API_KEY = os.environ.get("OPENAI_API_KEY", "sk-o7RCKovyezOE4sL5bZ4q0jHWejx0vWM5tzjWpo70yZlG52Al")
BASE_URL = "http://openai.infly.tech/v1/"
REQUEST_TIMEOUT = 180  # Timeout for the API request in seconds

MODEL_NAME_MAPPING = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-4o": "gpt-4o-2024-05-13",
}
# The model you want to use for the judgments.
MODEL_TO_USE = "gpt-4o"

DEFAULT_SYSTEM_TEMPLATE = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.

User:
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

def process_entry_with_gpt(entry, query_key='query', is_cot=False, test=False):
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
    final_model = "gpt-4o"
    openai.api_key = "placeholder-key-not-used"
    openai.base_url = BASE_URL
    
    # --- Construct Messages ---
    if test:
        messages = [{"role": "user", "content": "Hello, this is a test. Are you ready?"}]
        print("Sending a simple test request...")
    else:
        # Extract query and response from the entry
        query = entry[query_key]
        if is_cot:
            pred = entry['pred'].split('\nAnswer:')[-1].strip()
        else:
            pred = entry['pred'].strip()
        answer = entry['answer']
        
        messages = [
            {
                "role": "user",
                "content": DEFAULT_SYSTEM_TEMPLATE.format(query=query, reference_answer=answer, generated_answer=pred)
            }
        ]

    try:
        # --- Make API Call ---
        response = openai.chat.completions.create(
            model=final_model,
            messages=messages,
            extra_headers={"apikey": API_KEY},
            stream=False,
            temperature=0.1,
            top_p=1.0,
            max_tokens=256,
            timeout=REQUEST_TIMEOUT,
        )
        return json.loads(response.to_json())

    # --- Exception Handling ---
    except openai.APITimeoutError:
        print(f"❌ Error: Request timed out after {REQUEST_TIMEOUT} seconds.", file=sys.stderr)
    except openai.APIConnectionError as e:
        print(f"❌ Error: Could not connect to the API at {BASE_URL}. Check URL/network.", file=sys.stderr)
    except openai.RateLimitError:
        print("❌ Error: Rate limit exceeded. Please check your plan and billing details.", file=sys.stderr)
    except openai.APIStatusError as e:
        print(f"❌ Error: API returned status {e.status_code}. Response: {e.response.text}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
    
    return None

def main(args):
    print("--- 🚀 Starting GPT API File Processor ---")

    # --- Run Initial API Test ---
    print("\n--- 🧪 Performing initial API connectivity test... ---")
    test_response = process_entry_with_gpt({}, test=True)
    if test_response:
        print("✔️ Success! API connection is working.")
    else:
        print("🚨 FATAL ERROR: API test failed. Please check your configuration and credentials. Exiting.", file=sys.stderr)
        sys.exit(1)

    output_path = args.data_path.replace('.jsonl', '_score.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        print(f"Successfully loaded {len(all_data)} entries from {args.data_path}")
    except FileNotFoundError:
        print(f"🚨 FATAL ERROR: Input file not found at '{args.data_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"🚨 FATAL ERROR: Could not parse JSON from '{args.data_path}'. Please ensure it's a valid JSONL file.", file=sys.stderr)
        sys.exit(1)
    
    # --- Check for Existing Output File ---
    finished_indexes = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                finished_entries = [json.loads(line) for line in f]
            finished_indexes = {entry.get('qa_id') for entry in finished_entries}
            print(f"Found {len(finished_indexes)} already processed entries in {output_path}. Resuming...")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Warning: Could not read progress from {output_path}. Starting from scratch. Error: {e}", file=sys.stderr)

    # Filter out data that has already been processed
    data_to_process = [entry for entry in all_data if entry.get('qa_id') not in finished_indexes]
    print(f"Total entries to process: {len(data_to_process)}")

    # ---  Process Entries ---
    print(f"\n--- 🔄 Starting processing of {len(data_to_process)} entries... ---")
    try:
        for entry in tqdm(data_to_process, desc="Processing Entries"):
            start_time = time.time()
            response_data = process_entry_with_gpt(entry, query_key='question', is_cot=args.cot)
            
            if response_data:
                try:
                    # Extract the message content
                    message_content = response_data['choices'][0]['message']['content']
                    match = re.search(r'\d', message_content)
                    if match:
                        score = int(match.group(0))
                        passing = 1 if score >= 4 else 0
                        
                    entry['score'] = score
                    entry['passing'] = passing
                    entry['judge'] = message_content.strip()
                    # Write the successful result to the output file immediately
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            
                except (KeyError, IndexError):
                    print(f"\n⚠️ Warning: Response for index {entry.get('qa_id')} had an unexpected structure.", file=sys.stderr)
            else:
                print(f"\n⚠️ Failed to get response for index {entry.get('qa_id')}. Skipping.", file=sys.stderr)
                # Optional: Add a delay before retrying the next one if failures are frequent
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n--- 🛑 User interrupted operation. Shutting down gracefully. ---")
        print("Progress has been saved. You can restart the script to resume.")

    print("\n--- ✅🏁 Processing Finished ---")
    
    print("Starting to evaluate the results...")
    with open(output_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
        
    acc = sum(sample['passing'] for sample in samples) / len(samples)
    print(f"Evaluation complete. Accuracy: {acc * 100:.2f}%")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="use GPT API to process a JSONL file and extract answers.")
    parser.add_argument("data_path", type=str, help="Path to the input JSONL file containing the data to process.")
    parser.add_argument("--cot", action='store_true', help="whether data is generated with chain of thought instruction")
    args = parser.parse_args()
    
    main(args)