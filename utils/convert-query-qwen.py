import argparse
import json
import re
from typing import List, Dict

from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- Judge Prompt Template ---
# This prompt is the core of the script. It instructs the LLM on how to perform its role as a judge.
# 1. Role-playing: Instructs the LLM to act as an objective evaluation expert.
# 2. Task Description: Clearly defines the task as judging semantic equivalence, not just string matching.
# 3. Core Principles: Asks the LLM to ignore superficial differences in wording and focus on the core meaning.
# 4. Format Constraint: Enforces a strict JSON output format for easy and reliable parsing.
JUDGE_PROMPT_TEMPLATE = """
### **Prompt: Synthesize Sequential Document Analysis Steps into a Cohesive Thought Process**

**1. Role and Goal**

You are an expert AI assistant specializing in text synthesis and logical reasoning. Your task is to merge a series of sequential "thinking" steps into a single, coherent, and seamless thought process. This process documents the analysis of a set of document images in chunks to determine which are relevant for answering a specific question.

**2. Task Overview**

You will be given a list of sequential `thinking` strings. Each string represents the analysis of a small batch of document images (e.g., images 1-5, 6-10, etc.). Your goal is to combine these individual analysis steps into one master `<think>` block that reads as a single, continuous stream of consciousness. You must preserve the core analysis of each image while ensuring the final text is fluent and logically structured from beginning to end.

**3. Detailed Instructions**

* **Unified Introduction:** The final output must begin with a single, comprehensive introduction. This introduction should be taken from the first `thinking` step. It typically outlines the user's question and defines the criteria for what constitutes a relevant document image. Do not repeat this introductory analysis for subsequent chunks.
* **Chronological and Seamless Merging:** Combine the analysis of each image in the strict chronological order they are presented in the input list (e.g., analysis of images 1-5, followed by 6-10, 11-15, and so on). Create smooth transitions between the merged segments. Avoid abrupt sentences like "Now analyzing the next set of images." Instead, simply continue the analysis flow. For example, after analyzing Image 5, directly proceed to the analysis of Image 6.
* **Preserve Core Analysis:** The substance of the analysis for each individual image is critical and **must not be altered, summarized, or embellished.** The reasoning, observations, and conclusion about whether an image is relevant must be kept exactly as provided in the original `thinking` steps.
* **Consistent Formatting:** Maintain a consistent format for the analysis of each image, typically using bullet points (e.g., `- **Image X**:`). Ensure that image numbers are sequential and correct in the final output.
* **Consolidated Conclusion:** The final output must end with a single, definitive conclusion. This conclusion should synthesize the findings from all analyzed images and provide the final, complete list of relevant page/image numbers. This final summary should replace the intermediate conclusions found at the end of each individual `thinking` step.

**4. Input Format**

You will receive the inputs containing the `question`, the ground-truth `answer`, and a `thinking_steps` list, where each element is a string of a sequential thinking process.

**5. Output Format**
You must respond with a single markdown code block containing the complete, synthesized `<think>` block. The output should look like this:

```<think>
[Your complete, seamless thought process analyzing all document images in order, preserving the original analysis for each image, and culminating in a final conclusion listing all relevant images.]
</think>```

### Input
Question: {question}
Answer: {answer}
Thinking Steps: {thinking_steps}

### Example Output
<think>\nThe question asks for the communication channel indicated as the easiest by the EASY-O-METER. The key concepts here are identifying the \"EASY-O-METER\" and determining which communication channel it designates as the easiest.\n\nAnalyzing the provided images:\n\n- **Image 1**: The content is focused on introducing autonomous customers and superagents, with no mention of an EASY-O-METER or any evaluation of communication channels. Therefore, it is not relevant to answering the question.\n\n- **Image 2**: This image discusses reducing customer effort and presents statistics related to customer service issues and preferences. While it touches on the theme of ease, it lacks any reference to an EASY-O-METER or a comparison of communication channels, making it unsuitable for addressing the specific query.\n\n- **Image 3**: This image introduces the concept of \"Net Easy\" and its benefits but does not include an EASY-O-METER or a comparison of communication channels. It focuses more on the operational quality and loyalty aspects rather than pinpointing the easiest channel.\n\n- **Image 4**: This image outlines factors driving customer effort, such as getting through to the right person, advisor attributes, and customer navigation. Although it relates to the broader topic of customer effort, it does not feature an EASY-O-METER or evaluate communication channels directly, thus it is not pertinent to the question at hand.\n\n- **Image 5**: This image contains a graphic labeled \"EASY-O-METER,\" which visually represents different communication channels along a spectrum from hard work to easy. The meter includes labels for various channels such as letters, email, voice auto, voice, online self-serve, social, and chat. The green arrow pointing towards \"Chat\" suggests that it is positioned at the highest point on the ease scale, making it the channel indicated as the easiest by the EASY-O-METER. This image is directly relevant to the question.\n\n- **Image 6**: This image discusses customer preferences for various communication channels like phone, email, web-chat, social media, and video-chat. It highlights statistics on consumer behavior but does not mention anything related to an \"EASY-O-METER.\" Therefore, this page is not relevant to answering the question.\n\n- **Image 7**: This image focuses on the challenges of omni-channel communication and channel switching. It lists various communication methods and their usage statistics but does not include any reference to an \"EASY-O-METER.\" Thus, this page is also not relevant.\n\n- **Image 8**: This image explores the relationship between smartphones and customer service, including app downloads, GPS sharing, and screen sharing preferences. There is no mention of an \"EASY-O-METER,\" so this page does not provide the necessary information to answer the question.\n\n- **Image 9**: This image emphasizes the role of phone calls in handling complex issues and building customer relationships. It includes statistics on call center usage and customer trust but does not refer to an \"EASY-O-METER.\" Consequently, this page is not relevant.\n\n- **Image 10**: This image covers the growth of web-chat as a communication channel and mentions user preferences and security concerns. Like the previous images, it does not include any reference to an \"EASY-O-METER,\" making it irrelevant to the question.\n\n- **Image 11**: This image discusses video culture and customer engagement through video-chat. It includes statistics related to video call preparation and preferences but does not mention anything about an \"EASY-O-METER\" or evaluate communication channels based on ease. Therefore, this page is not relevant to the question.\n\n- **Image 12**: This image focuses on social media as a platform for customer service. It highlights expectations and behaviors related to social media interactions but does not include any reference to an \"EASY-O-METER\" or assess communication channels based on their ease of use. Thus, this page is also not relevant.\n\n- **Image 13**: This image presents new models for contact centers, including visual depictions of agent workspaces and technology. There is no mention of an \"EASY-O-METER\" or evaluation of communication channels. Hence, this page is not relevant to the question.\n\n- **Image 14**: This image outlines the primary functions of contact centers in 2020, with a bar chart showing various functions like complaint handling, problem-solving, and inbound transactions. It does not discuss an \"EASY-O-METER\" or rank communication channels by ease. Consequently, this page is not relevant.\n\n- **Image 15**: This image addresses the skills required for super agents in 2020, emphasizing communication skills, problem-solving abilities, and product/service experience. It does not mention an \"EASY-O-METER\" or evaluate communication channels. Therefore, this page is not relevant either.\n\n- **Image 16**: This image presents \"Primary measures 2020,\" including metrics like customer satisfaction, call handling time, and easy/effort. While it mentions \"easy/effort,\" it does not specify any communication channels or reference an \"EASY-O-METER.\" Therefore, this page is not relevant to answering the question.\n\n- **Image 17**: This image shows \"Core channels 2020,\" listing various communication channels such as social media, webchat, email, and phone, along with their respective values. Although it lists communication channels, there is no indication of an \"EASY-O-METER\" or any measure of ease associated with these channels. Thus, this page is also not relevant.\n\n- **Image 18**: This image discusses the \"Location of centre 2020,\" focusing on different types of contact center locations like field/mobile, offshore, home-based, integrated hubs, and dedicated centers. It does not mention communication channels or an \"EASY-O-METER,\" making it irrelevant to the question.\n\n- **Image 19**: This image covers \"Key technologies 2020,\" highlighting technologies such as big data analytics, cloud, agent webchat, intelligent-based routing, and mobile technologies. There is no mention of communication channels or an \"EASY-O-METER,\" so this page is not relevant either.\n\nIn conclusion, out of all the analyzed document images, only **Image 5** contains the EASY-O-METER and explicitly indicates \"Chat\" as the easiest communication channel. Therefore, the answer to the question is based solely on Image 5.\n</think>
"""


def load_data(file_path: str) -> List[Dict]:
    """Loads data from a JSON Lines file."""
    data = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON file: {file_path}")
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
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    print("Model loaded successfully.")

    # 2. Prepare the evaluation data and prompts
    data_to_judge = load_data(args.input_file)
    if args.debug:
        data_to_judge = data_to_judge[:10]
    prompts = [
        JUDGE_PROMPT_TEMPLATE.format(
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            thinking_steps=item.get('responses', [])
        )
        for item in data_to_judge
    ]
    print(f"Loaded {len(prompts)} records for evaluation.")

    # 3. Configure sampling parameters
    # For deterministic tasks like judging, we use a low temperature.
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=4096  # 256 tokens should be sufficient for the JSON output.
    )

    # 4. Generate judgments in a batch with vLLM
    print("Generating judgments in batch...")
    outputs = llm.generate(prompts, sampling_params)
    print("Inference complete.")

    # 5. Parse the outputs and save the results
    print("Parsing and saving results...")

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing results"):
            original_item = data_to_judge[i]
            generated_text = output.outputs[0].text.strip()
            if generated_text.startswith('<think>') and generated_text.endswith('</think>'):
                original_item['new_thinking'] = generated_text
            else:
                # Attempt to extract the <think> block using regex
                match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
                if match:
                    original_item['new_thinking'] = f"<think>{match.group(1).strip()}</think>"
                else:
                    print(f"Warning: Output format unexpected for item {i}. Storing raw output.")
                    continue

            f_out.write(json.dumps(original_item, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {args.output_file}")


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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size for model loading."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Whether to run in debug mode with a smaller subset of data."
    )
    args = parser.parse_args()
    main(args)