import argparse
import json
import re
from typing import List, Dict

from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict

# --- Judge Prompt Template ---
# This prompt is the core of the script. It instructs the LLM on how to perform its role as a judge.
# 1. Role-playing: Instructs the LLM to act as an objective evaluation expert.
# 2. Task Description: Clearly defines the task as judging semantic equivalence, not just string matching.
# 3. Core Principles: Asks the LLM to ignore superficial differences in wording and focus on the core meaning.
# 4. Format Constraint: Enforces a strict JSON output format for easy and reliable parsing.
JUDGE_PROMPT_TEMPLATE = """
You are an objective and rigorous AI judge.
Your task is to determine if the 'Model Output' is semantically equivalent to the 'Reference Answer' or expresses the same core information.

Please adhere to the following evaluation criteria:
1.  **Semantic Equivalence**: The prediction should be judged as 'Correct' if its core meaning is consistent with the ground truth.
2.  **Ignore Superficial Differences**: Do not penalize for differences in wording, grammar, punctuation, sentence order, or level of detail if the semantic meaning is correct. For example, 'Beijing is the capital of China' and 'The capital of China is Beijing' are equivalent.
3.  **Factual Accuracy**: The prediction should be judged as 'Incorrect' if it contains factual errors that contradict the ground truth.

[Reference Answer (Ground Truth)]:
{ground_truth}

[Model Output (Prediction)]:
{pred}

Carefully compare the two texts above and return your verdict in a strict JSON format. The JSON object must contain two fields: `judgement` and `reason`:
- `judgement`: A string with the value of either "Correct" or "Incorrect".
- `reason`: A string briefly explaining your reasoning (1-2 sentences).

**Your output must be a single, valid JSON object and nothing else. Do not include any explanatory text or code block markers.**

For example:
```json
{{
    "judgement": "Correct", 
    "reason": "The model's output, despite different wording, accurately conveys the core meaning of the reference answer."
}}
```
"""


def load_data(file_path: str) -> List[Dict]:
    """Loads data from a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line: {line.strip()}")
    return data


def main(args):
    """Main function."""
    print("=" * 20)
    print("Starting LLM-as-Judge Evaluation Task")
    print(f"Model Path: {args.model_path}")
    print(f"Input File: {args.input_file}")
    print(f"Output File: {args.output_file}")
    print("=" * 20)

    # 1. Initialize the vLLM model
    print("Loading model...")
    # trust_remote_code is required for Qwen models
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,
        trust_remote_code=True
    )
    print("Model loaded successfully.")

    # 2. Prepare the evaluation data and prompts
    data_to_judge = load_data(args.input_file)
    prompts = [
        JUDGE_PROMPT_TEMPLATE.format(
            ground_truth=item.get('answer', ''),
            pred=item.get('pred', '').split('<answer>')[-1].replace('</answer>', '').strip()  # Extract answer after 'Answer:' if present
        )
        for item in data_to_judge
    ]
    print(f"Loaded {len(prompts)} records for evaluation.")

    # 3. Configure sampling parameters
    # For deterministic tasks like judging, we use a low temperature.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256  # 256 tokens should be sufficient for the JSON output.
    )

    # 4. Generate judgments in a batch with vLLM
    print("Generating judgments in batch...")
    outputs = llm.generate(prompts, sampling_params)
    print("Inference complete.")

    # 5. Parse the outputs and save the results
    print("Parsing and saving results...")
    correct_count = 0
    incorrect_count = 0
    parse_failures = 0
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing results"):
            original_item = data_to_judge[i]
            generated_text = output.outputs[0].text.strip()
            # print(f"Generated Text for item {i}: {generated_text}")  # Debugging line to see the raw output
            
            # extract the json
            match = re.search(r'```json\n(.*?)\n```', generated_text, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()
            else:
                # If no code block is found, assume the entire output is the JSON
                print(f"Warning: No JSON code block found in output for item {i}. Using entire output as JSON.")
                continue

            try:
                # Try to parse the JSON output from the LLM
                judgement_data = json.loads(generated_text)
                judgement = judgement_data.get('judgement', 'Parse Failed')
                reason = judgement_data.get('reason', 'Could not extract reason from JSON.')

                if judgement == "Correct":
                    correct_count += 1
                elif judgement == "Incorrect":
                    incorrect_count += 1
                else:
                    # Handle cases where the 'judgement' field has an unexpected value
                    parse_failures += 1
                    judgement = "Parse Failed"

            except json.JSONDecodeError:
                # If the LLM output is not valid JSON, mark as a failure
                judgement = "Parse Failed"
                reason = f"Original output was not valid JSON: {generated_text}"
                parse_failures += 1

            original_item['judgement'] = judgement
            original_item['reason'] = reason

            json_line = json.dumps(original_item, ensure_ascii=False)
            f_out.write(json_line + '\n')

    # 6. Print the final summary report
    total = len(data_to_judge)
    print(f"correct ratio: {correct_count / total}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions using an LLM-as-Judge.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/map-vepfs/models/Qwen2.5-72B-Instruct/",
        help="Path to the judge LLM (Hugging Face hub name or local path)."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input file in JSON Lines format, containing 'ground_truth' and 'pred' fields."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="judgement_results.jsonl",
        help="Output file to save the evaluation results."
    )
    args = parser.parse_args()
    main(args)