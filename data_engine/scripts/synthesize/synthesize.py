
import argparse
import copy
import json
import logging
import multiprocessing
import os
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import ray
import transformers
import time

from helper import *
from model_utils import LM

default_sys = "You are a helpful assistant."
boxed_sysprompt = "Please reason step by step, and put your final answer within \\boxed{}."

templates = {
"student_generation_initial": """Here is a coding problem:\n{}
    
Generally, the response should include: 
- Explain what the problem is asking. Identify the core knowledge or concepts the problem is testing.
- Think about how to address the problem, and then summarize the high-level main idea to solve the problem.
- Provide detailed steps to solve the problem, try writing the pseudo codes for the critical steps, and debug the codes in mind.
Use a **first-person, self-thinking** tone in your response. Please think step by step and put your finalized code snippets within a final ```python and ``` block.""",
    # "student_generation": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
    # The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> finalized code here </answer>. 
    # The reasoning process of `<think></think>` often includes problem understanding, recall on relevant knowledge and theories, a sketch for problem solving, and then detailed calculations and derivations.
    # Within the reasoning block, try to use less than 10 paragraphs, separated using double newlines.
    # Solve the following problem step by step. 
    
    # {}
    # """,
    "student_direct_refinement": """You are an expert teacher. Your primary goal is to analyze a given problem and a student's solution, then **enrich or correct** the student's solution with clear, detailed, and accurate explanations.
Here's how to approach the task:
1.  Carefully read and understand the problem provided.
2.  Analyze the student's solution. 
3.  Correct or Refine/Enrich the student solution**: 

# Problem
{}

# Student Solution
{}

Imagine explaining to a student who has poor mastery of relevant knowledge.
Based on the problem context and the student solution, think about how to refine the given solution.
Put your thinking process in `<think></think>` tags, provide your refinements in the `<refine></refine>` tags. 
Please follow the initial replacement block to use a **first-person, self-thinking** tone in your refinements. Use simple language, provide extremely detailed steps in your calculations, derivations and reasoning, without skipping any steps.
""",
"student_smooth": """You are given a problem and a student solution. 
1. Understand the problem.
2. Stick to the student solution, polish and refine the solution.
    - Use **first-person, self-thinking** tone, explain the thought process.
    - elaborate on the reasoning, make the thinking process logical, and easy-to-follow for a reader with poor knowledge on relevant concepts.
    - keep the formatting of `<think></think>` and `<answer></answer>` tags.

# Problem
{}

# Student Solution
{}
""",
"student_elaboration_reference": """You are an expert teacher. Your primary goal is to analyze a given problem and a student's solution, then **enrich or correct** the student's solution with clear, detailed, and accurate explanations.
Here's how to approach the task:
1.  Carefully read and understand the problem provided.
2.  Analyze the student's solution and compare it with the reference solution. 
3.  Correct or Refine/Enrich the Replacement Block**: 

# Problem
{}

# Reference Solution
{}

# Student Solution
{}
----
<replace>
{}
</replace>
----
{}

Generally, the response should include: 
- Explain what the problem is asking. Identify the core knowledge or concepts the problem is testing.
- Think about how to address the problem, and then summarize the high-level main idea to solve the problem.
- Provide detailed steps to solve the problem, try writing the pseudo codes for the critical steps, and debug the codes in mind.

Now look at the below replace block:
<replace>
{}
</replace>

The goal is to correct and refine the replacement block, in order to help the student eventually get the correct final answer.
1. Determine the replacement block is in which stage in the above solution structure.
2. Comparing with the reference solution, check whether the student solution is correct and whether the replacement block is correct.
  - If the replacement block is correct, focus on providing details and enrich it.
  - If the replacement block is incorrect, focus on correcting it.

Put your thinking process in `<think></think>` tags, provide your refinements in the `<refine></refine>` tags. 
Please follow the initial replacement block to use a **first-person, self-thinking** tone in your refinements. 
""",}

think_prefix = "<think>\n"

@ray.remote
def generate(inputs, model: LM, num_rollouts=None, isgreedy=True, **kwargs):
    results = []
    
    if len(inputs)==1:
        completions, temperature = model.generate(inputs, num_rollouts, isgreedy, **kwargs)
        results = completions
    else: 
        for inp in inputs:
            completions, temperature = model.generate(inp, num_rollouts, isgreedy, **kwargs)
            results.append(completions) 
    return results


lm_tokenizer, lm_model = None, None
def get_model_output(
    template_role: str,
    system_prompt: Optional[str],
    template_inputs: List[List[Any]],
    tokenizer: Any,
    model: Any,
    prompt_suffix: str,
    # NOTE: use_fewshot is an unused parameter in this function.
    use_fewshot: bool,
    num_rollouts: int,
    is_greedy: bool = True,
    **kwargs,
) -> Tuple[Any, List[str]]:
    """
    Constructs prompts from templates and submits them to the model for generation.

    Args:
        template_role: The key for the desired prompt template.
        system_prompt: An optional system message to guide the model's behavior.
        template_inputs: A list of lists, where each inner list contains the arguments for a prompt template.
        tokenizer: The model's tokenizer.
        model: The language model instance.
        prompt_suffix: A string to append to each prompt after formatting.
        use_fewshot: (Unused) A flag that was likely intended for few-shot prompting.
        num_rollouts: The number of sequences to generate for each prompt.
        is_greedy: A flag to control the decoding strategy.
        **kwargs: Additional arguments passed to the generation function.

    Returns:
        A tuple containing the Ray object for the asynchronous generation task
        and the list of fully constructed prompts sent to the model.
    """
    template = templates[template_role]
    
    # Format each input using the specified template.
    formatted_queries = [template.format(*inp) for inp in template_inputs]
    
    prompts = []
    for query in formatted_queries:
        # Create the standard message format for model interaction.
        messages = [{"role": "user", "content": query}]
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Construct the final prompt string.
        prompt = make_prompt(tokenizer, messages)
        prompt += prompt_suffix
        prompts.append(prompt)

    # Launch the remote generation task using Ray.
    generation_task = generate.remote(prompts, model, num_rollouts, isgreedy=is_greedy, **kwargs)
    return generation_task, prompts

class RefinementProcessor:
    """Handles the iterative refinement process for a single generated response."""

    def __init__(self, node: Any, tokenizer: Any, model: Any, stop_threshold: float, max_steps: int, num_expansion: int = 2):
        self.node = node
        self.tokenizer = tokenizer
        self.model = model
        self.stop_threshold = stop_threshold
        self.max_steps = max_steps
        self.num_expansion = num_expansion

    def run(self, initial_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full iterative refinement loop on a generated "thinking" process.

        Args:
            initial_info: A dictionary containing the initial model rollout and its metadata.

        Returns:
            An updated info dictionary containing the results of the refinement process.
        """
        # The 'initial_ppl' is a tuple where the first element is the log perplexity.
        initial_perplexity = initial_info.get('initial_ppl', (1000.0,))
        print(f"===> Initial PPL: {initial_perplexity[0]}")
        if initial_perplexity[0] < self.stop_threshold:
            print("===> Skipping refinement due to low initial PPL.")
            return initial_info

        thinking_segments = initial_info.get('thinking_segments', [])
        if not thinking_segments:
            return initial_info

        print(f"====> Starting refinement on {len(thinking_segments)} thinking steps.")
        
        finalized_thinking_steps = []
        best_ppl_so_far = initial_perplexity
        
        # Iterate through each segment of the thinking process to refine it.
        for i in range(min(self.max_steps, len(thinking_segments))):
            # Reconstruct the thinking process parts: before, current, and after the segment being refined.
            before_segment = "\n\n".join(finalized_thinking_steps)
            segment_to_refine = thinking_segments[i].strip()
            after_segments = "\n\n".join(s.strip() for s in thinking_segments[i+1:])
            
            # Generate and evaluate several possible refinements for the current segment.
            best_candidate, best_ppl, all_candidates = self._refine_one_step(
                before_segment, segment_to_refine, after_segments
            )
            
            # Decide whether to keep the original segment or use a generated refinement.
            choice = 'original'
            chosen_text = segment_to_refine
            if best_candidate is not None and best_ppl[0] < best_ppl_so_far[0]:
                best_ppl_so_far = best_ppl
                chosen_text = best_candidate['refinement']
                choice = f"refinement_No.{best_candidate['id']}"

            finalized_thinking_steps.append(chosen_text)
            
            # Log the details of this refinement step.
            initial_info[f"refine_thinking_step_No.{i+1}"] = {
                'segment_to_refine': segment_to_refine,
                'choice': choice,
                'chosen_refinement': chosen_text,
                'after_avg_token_logp': best_ppl_so_far[0],
                'possible_refinements': all_candidates,
            }

            # Stop early if the perplexity has reached the desired threshold.
            if best_ppl_so_far[0] < self.stop_threshold:
                print("===> Stopping refinement early as PPL threshold reached.")
                # Append the rest of the original segments without refinement.
                finalized_thinking_steps.extend(s.strip() for s in thinking_segments[i+1:])
                break
        
        initial_info['corrected_thinking'] = finalized_thinking_steps
        
        # If any refinement was done, perform one final rollout with the complete refined thinking.
        if finalized_thinking_steps:
            initial_info['finalized_avg_token_logp'] = best_ppl_so_far[0]
            new_thinking_prompt = "<thint
            >\n" + "\n\n".join(finalized_thinking_steps)
            rollout_texts, _, _, _ = direct_rollout(self.node, new_thinking_prompt, 1)
            initial_info['finalized_response'] = rollout_texts[0]

        return initial_info

    def _refine_one_step(self, before: str, current: str, after: str) -> Tuple[Optional[Dict], Tuple[float, List], List[Dict]]:
        """Generates and evaluates possible refinements for a single thinking segment."""
        num_samples = self.num_expansion
        # The arguments are: question, reference_answer, text_before, text_to_replace, text_after, original_text
        prompt_args = (self.node.memory['q'], self.node.memory['ref'], before, current, after, current)
        
        rollouts_obj, _ = get_model_output(
            template_role='student_elaboration_reference', 
            system_prompt=default_sys, 
            template_inputs=[prompt_args], 
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix="",
            use_fewshot=False,
            num_rollouts=num_samples, 
            is_greedy=False
        )
        # The return from ray.get should be (texts, logps, offsets), but only texts are used here.
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        refinement_candidates = []
        best_candidate_info = None
        # Initialize with a high perplexity value.
        best_perplexity = (float('inf'), [])

        for i, rollout_text in enumerate(rollout_texts):
            # Extract the refined text from within the <refine> tags.
            last_block_start = rollout_text.rfind("<refine>")
            if last_block_start == -1:
                print("Warning: <refine> tag not found in output, skipping.")
                continue

            block_end = rollout_text.rfind("</refine>")
            start_pos = last_block_start + len("<refine>")
            refinement_text = rollout_text[start_pos:block_end if block_end != -1 else None].strip()
            
            # Create the full "thinking" process with the new refinement.
            recomposed_thinking = f"{refinement_text}"
            if before: recomposed_thinking = f"{before}\n\n{recomposed_thinking}"
            if after: recomposed_thinking = f"{recomposed_thinking}\n\n{after}"
            
            # Evaluate the new thinking process by calculating the perplexity of the reference answer.
            manager = PosteriorManager('student_generation', default_sys, [[self.node.memory['q']]], self.tokenizer, self.model, self.node.memory['ref'])
            _, posterior_prefix = manager.prepare(think_prefix, recomposed_thinking, "")
            ppl_obj, _ = manager.submit(posterior_prefix)
            # The 'compute' method returns a tuple (log_perplexity, debug_info).
            new_perplexity_result = manager.compute(ppl_obj)

            print(f"Refinement candidate {i} | New log PPL: {new_perplexity_result[0]}")
            
            candidate_info = {
                'id': i,
                'refinement': refinement_text,
                'generator': self.model.model_name,
                'raw_output': rollout_text,
                'raw_input_for_posterior': posterior_prefix,
                'avg_token_logp': new_perplexity_result[0],
            }
            refinement_candidates.append({f'expansion_No.{i}_of_segment': candidate_info})

            # If this candidate is better than the best one so far, update it.
            if new_perplexity_result[0] < best_perplexity[0]:
                best_perplexity = new_perplexity_result
                best_candidate_info = candidate_info
                
        return best_candidate_info, best_perplexity, refinement_candidates
    
class PosteriorManager:
    """Calculates the log probability of a reference answer given a thinking process."""
    def __init__(self, role: str, system_prompt: str, inputs: List[List[Any]], tokenizer: Any, model: Any, ref_answer: str):
        self.role = role
        self.system_prompt = system_prompt
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.model = model
        self.ref_answer = ref_answer
        
    def prepare(self, think_prefix: str, thinking_process: str, latent_thinking: str) -> Tuple[str, str]:
        """Prepares the prompt for posterior probability calculation."""
        # The part of the prompt before the reference answer.
        self.prefix_before_answer = f"{think_prefix}{thinking_process}\n{latent_thinking}</think>\n<answer>\n```python\n"
        # The full prompt including the reference answer.
        self.posterior_prefix = f"{self.prefix_before_answer}{self.ref_answer}\n```\n</answer>"
        return self.prefix_before_answer, self.posterior_prefix
    
    def submit(self, prefix: str) -> Tuple[Any, List[str]]:
        """Submits the prompt to the model to get token log probabilities."""
        self.rollouts_obj, self.real_input_prompts = get_model_output(
            self.role, self.system_prompt, self.inputs, self.tokenizer, self.model, prefix, 
            use_fewshot=False, num_rollouts=1, is_greedy=False, prompt_only=True
        )
        return self.rollouts_obj, self.real_input_prompts

    def compute(self, rollouts_obj: Any) -> Tuple[float, List[Dict]]:    
        """Computes the log perplexity of the reference answer from the model's output."""
        _, prompt_logprobs, _ = ray.get(rollouts_obj)
        if not prompt_logprobs: return 1000.0, []
        
        current_prompt_logprobs = prompt_logprobs[0]
        
        # This complex logic is used to find the exact tokens corresponding to the reference answer.
        offsets = self.tokenizer(self.posterior_prefix, return_offsets_mapping=True).offset_mapping 
        start_char_index = len(self.prefix_before_answer)
        end_char_index = start_char_index + len(self.ref_answer)

        # Find the number of tokens to count backward from the end of the prompt to find the answer's start.
        tokens_to_go_back = 0
        for i, (start, end) in enumerate(offsets[::-1]):
            if start <= start_char_index:
                tokens_to_go_back = i + 1
                break
        
        # Find how many tokens the reference answer spans.
        answer_token_span = 0
        for j, (start, end) in enumerate(offsets[-tokens_to_go_back:]):
            if end >= end_char_index:
                answer_token_span = j + 1
                break
        
        answer_logps = []
        # Sum the log probabilities of the tokens that make up the reference answer.
        logprob_slice = current_prompt_logprobs[-tokens_to_go_back : -tokens_to_go_back + answer_token_span]
        for logp_dict in logprob_slice: 
            # The dictionary may have multiple keys; the first one corresponds to the prompt token.
            major_key = list(logp_dict)[0]
            logp = logp_dict[major_key]['logprob']
            answer_logps.append(logp)
            
        # Calculate the negative mean log probability (log perplexity). A lower value is better.
        log_perplexity = -np.mean(answer_logps) if answer_logps else 1000.0
        
        # For debugging, gather the log probability info for tokens surrounding the answer.
        debug_slice = current_prompt_logprobs[-tokens_to_go_back-2 : -tokens_to_go_back + answer_token_span + 2]
        debug_logprob_info = [logp_dict[list(logp_dict)[0]] for logp_dict in debug_slice]
        
        return log_perplexity, debug_logprob_info
    
def direct_rollout(node, prefix, n_sample=1, role=None):
    
    role, sysp, in_keys = ('student_generation' if role is None else role), default_sys, ['q', ] 
    node.memory['gen_role'] = role
    
    inputs = [node.memory[k] for k in in_keys]
    tok, model = lm_tokenizer, lm_model
    node.memory['generator'] = model.model_name
    inp = [inputs]
    print(f"===> {role}: {n_sample} for {len(inp)} queries")
    rollouts_obj, real_input_prompts = get_model_output(role, sysp, inp, tok, model, prefix, False, n_sample, is_greedy=False)
    rollout_texts, rollout_logps, rollout_offsets = ray.get(rollouts_obj)
    del rollouts_obj
    return rollout_texts, rollout_logps, rollout_offsets, real_input_prompts



@ray.remote
def process_item(
    item: Dict[str, Any],
    file_prefix: str,
    rank: int,
    n_sample: int, 
    configs=dict()
):
    uid = item.get('uid', 'unknown')
    output_fname = f"{file_prefix}_{uid}_rk{rank}_meta.jsonl"
    if glob(f"{output_fname}*"):
        print(f"Skipping existing item: {uid}")
        return True
    
    stop_thresh = configs['processing']['stop_thresh']
    max_step = configs['processing']['max_step']
    num_expansion = configs['processing']['num_expansion']

    node = Node(ref=item['ref'], raw_q=item['q'], info={'uid': uid})
    node.memory.update({k:v for k,v in item.items() if not isinstance(v, np.ndarray)})
    
    fname = output_fname

    tok, model = lm_tokenizer, lm_model

    # 1. Initial Rollout
    # n_sample = 1
    rollout_texts, rollout_logps, rollout_offsets, real_input_prompts = direct_rollout(node, "", n_sample, role='student_generation_initial')

    outcomes = []
    expanded_prompts = [pp for pp in real_input_prompts for _ in range(n_sample)]
    
    for roll, inpprompt, token_logps, token_to_text_offsets in zip(rollout_texts, expanded_prompts, rollout_logps, rollout_offsets):
        ntoken = len(token_logps)
        info = dict(ntokens=ntoken,)
        
        # separate thinking and code answer
        thinking = "".join(roll.split('```python')[:-1])
        answer_code = '```python'+roll.split('```python')[-1]

        # breakdown thinking to steps 
        aa,bb,thinking_segments = breakdown_steps(thinking)
        
        # get PPL of initial rollout
        role, sysp = 'student_generation', default_sys 
        inp = [[node.memory['q']]]
        manager = PosteriorManager(role, sysp, inp, tok, model, node.memory['ref'])
        posterior_prefix1, posterior_prefix = manager.prepare(think_prefix, thinking, "")
        rollouts_obj, real_input_prompts = manager.submit(posterior_prefix)
        noreplace_log_ppl = manager.compute(rollouts_obj)
        ppl = noreplace_log_ppl
            
        # save info
        info.update(dict(initial_response=roll, 
                            answer=answer_code,
                            initial_ppl=ppl, 
                            generator=model.model_name,
                            thinking_segments=thinking_segments,
                        #  rawinput=inpprompt, 
                    ))
    
        outcomes.append(info)
    
    # 2. iterative refinement for each response
    refinement_processor = RefinementProcessor(
        node, tok, model, 
        stop_threshold=stop_thresh, 
        max_steps=max_step,
        num_expansion=num_expansion
    )
    
    updated_results = []
    for info in outcomes:
        refined_info = refinement_processor.run(info)
        updated_results.append(refined_info if refined_info is not None else info)
    
    node.memory['alist'] = updated_results
    
    flag = False 
    result_file = f"{fname}_meta.jsonl"
    with open(result_file, "w") as file:
        line = json.dumps(node.memory,indent=2)

        file.write(line + "\n")
    print(f'dumped to {result_file}')
      
    return flag
        

def main(args: argparse.Namespace):
    """Main function to load data, initialize models, and run the processing loop."""
    config = load_config(args.config_file)
    file_prefix = config["output"]["file_prefix"]
    folder = os.path.sep.join(file_prefix.split(os.path.sep)[:-1])
    os.makedirs(folder, exist_ok=True)
    num_rollouts = config["processing"]["num_rollouts"]

    # Create output directory
    os.makedirs(os.path.dirname(config["output"]["file_prefix"]), exist_ok=True)
    
    # Initialize Ray
    ray.init(num_cpus=args.num_cpus)

    # Load models and tokenizers
    print("Loading models and tokenizers...")
    global lm_tokenizer, lm_model
    config["model"]["model_args"]["port"] = args.port
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model"]["model_name"],
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    lm_model = LM(
        model_type=config["model"]["model_type"],
        model_name=config["model"]["model_name"],
        model_url=None, # lm_url,
        num_rollouts=num_rollouts,
        tokenizer=lm_tokenizer,
        **config["model"]["model_args"],
    )
    
    path = config['input']['file_path']
    df = pd.read_parquet(path)
    # df.rename(columns={'seq_id':'uid', 'instruction':'q', 'code':'ref', 'testcase':'gold'}, inplace=True)
    # sel = df['ref'].apply(len)>1000
    # df = df[sel]
    data = df.to_dict('records')
    
    num_each = len(data) // args.total_ranks + 1
    rank_data = data[args.rank * num_each : (args.rank + 1) * num_each]
    print(f"Rank {args.rank}/{args.total_ranks}: Processing {len(rank_data)} items.")
    st = time.time()
    task_futures = [
        process_item.remote(
                item,
                file_prefix,
                args.rank,
                num_rollouts,
                configs=config
            )
        for item in rank_data
    ]

    # Wait for all tasks to complete
    ray.get(task_futures)

    print(f"Total time elapsed: {(time.time() - st) / 60:.2f} minutes.")
    ray.shutdown()
    
    logging.info("Finished processing the JSON file.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run language model generation and refinement pipeline.")
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--port", type=int, required=True, help="Port.")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process.")
    parser.add_argument("--total-ranks", type=int, required=True, help="Total number of parallel processes.")
    parser.add_argument("--num-cpus", type=int, default=32, help="Number of CPUs to allocate for Ray.")
    
    args = parser.parse_args()
    main(args)
