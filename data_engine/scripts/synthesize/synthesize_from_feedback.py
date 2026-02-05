
import argparse
import copy
import json
import logging
import multiprocessing
import os
from glob import glob
from typing import Any, Dict, List, Tuple, Optional
import re
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
"student_generation_short": """
{}
Please reason step by step, first thinks about the reasoning process in the mind, then provides the user with the final answer within \\boxed{{}}. The reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{final answer}} here </answer>.
Solve the following question and put your final answer within \\boxed{{}}.
""",

"student_generation_short_mimo": """
Read the following question carefully, solve it step by step, and then output the final answer in the format of "\nAnswer: \\boxed{{final answer}}". You should put your final answer in \\boxed{{}}.
Question: {}
""",

"student_generation_initial_answerable":"""You are a skilled analyst. Your task is to answer a question based on a given document image by producing a detailed reasoning process. If the document does not contain the necessary information to answer the question, you should state your final answer as "Not Answerable."
    
Generally, your reasoning process should address the question as follows: 
- Begin your analysis with an introductory paragraph that provides a general overview of the document. Start by identifying what kind of document it is (e.g., a resume, an invoice, a report) and discuss its main purpose.
- Clearly state the user's question you are addressing. Explain what specific information or data points you will need to find within the document to arrive at a satisfactory answer. Briefly outline the approach you will take to locate and analyze this information.
- Provide your detailed reasoning process. 
    - Present your detailed reasoning. Begin by pinpointing the exact location of the relevant evidence within the document, using descriptive language to guide the reader.
    - To substantiate your analysis, quote key phrases or data directly from the document. Clearly describe any calculations or logical deductions you make, laying out the steps sequentially.
    - If, after a thorough search, you determine that the document does not contain the necessary information to answer the question, state your final answer as "Not Answerable." In such cases, you must still provide a detailed analysis explaining why the question cannot be answered.

The assistant first thinks about the reasoning process in the mind, then provides the user with the final answer within \\boxed{{}}. The reasoning process are enclosed within <think> </think> and final answer are enclose within <answer> </answer> tags, respectively, i.e., <think> reasoning and analysis prorcess here </think> <answer> \\boxed{{final answer}} here </answer>. Note that the final answer within <answer> </answer> tags should be a short phrase or a single number, not a long sentence and solve the following question and put your final answer within \\boxed{{}}.

Question: {}""",

"student_generation_initial":"""You are a skilled analyst. Your task is to answer a question based on a given document image by producing a detailed reasoning process. 
    
Generally, your reasoning process should address the question as follows: 
- Begin your analysis with an introductory paragraph that provides a general overview of the document. Start by identifying what kind of document it is (e.g., a resume, an invoice, a report) and discuss its main purpose.
- Clearly state the user's question you are addressing. Explain what specific information or data points you will need to find within the document to arrive at a satisfactory answer. Briefly outline the approach you will take to locate and analyze this information.
- Provide your detailed reasoning process. Explain exactly where in the document you located the relevant evidence, using descriptive language. To make your analysis robust, quote key phrases or data directly from the document within your sentences to support your findings. If you perform any calculations or logical steps, describe them clearly in sequence. Note that you may find you need evidence that wasn't analyzed in your initial approach outline, so you don't need to strictly adhere to the initial outline.

The assistant first thinks about the reasoning process in the mind, then provides the user with the final answer within \\boxed{{}}. The reasoning process are enclosed within <think> </think> and final answer are enclose within <answer> </answer> tags, respectively, i.e., <think> reasoning and analysis prorcess here </think> <answer> \\boxed{{final answer}} here </answer>. Note that the final answer within <answer> </answer> tags should be a short phrase or a single number, not a long sentence and solve the following question and put your final answer within \\boxed{{}}.

Question: {}""",

"student_generation_locate_interpret":"""You are a highly skilled and meticulous analyst. Your task is to provide a comprehensive answer to a question based on a given document image. Your response must not only state the final answer but also present a detailed, step-by-step reasoning process that shows how you arrived at that conclusion.

Your analysis should be structured as follows:

1.  **Document Identification & Contextualization:**
    * Begin by identifying the document type (e.g., a financial report, a research paper, a resume or something else).
    * Briefly describe the document's primary purpose. What is the main function of this document? What kind of information is it designed to convey?
    * This initial overview provides the essential context for your subsequent analysis.

2.  **Question Analysis & Strategy Formulation:**
    * Restate the user's question clearly and concisely.
    * Break down the question to identify the specific pieces of information or data points that are required to formulate a complete answer.
    * Outline your strategy: Explain where you anticipate finding this information within the document (e.g., "I will look for a table titled 'Sales Summary' or a line item under 'Total Amount'"). This demonstrates a logical and proactive approach.

3.  **Detailed Reasoning & Evidence Extraction:**
    * Locate: First, describe the exact location of the relevant information. The location could be a figure, a table, a subtitle of a figure or a specific paragraph.
    * Interpret: Next, explain the process of extracting the data. If the answer requires calculation, show the steps (e.g., "The total is calculated by adding the values from rows 1 and 2: $150 + $75 = $225").
    * Revisit evidence: Explain how the located data directly answers the question. Connect the specific numbers, text, or visual elements to the final conclusion.

4.  **Final Conclusion:**
    * Provide a clear and definitive answer to the original question.
    * The conclusion should be a direct result of your detailed reasoning process, reinforcing the credibility of your analysis.

The assistant first thinks about the reasoning process in the mind, then provides the user with the final answer within \\boxed{{}}. The reasoning process are enclosed within <think> </think> and final answer are enclose within <answer> </answer> tags, respectively, i.e., <think> reasoning and analysis prorcess here </think> <answer> \\boxed{{final answer}} here </answer>. Note that the final answer within <answer> </answer> tags should be a short phrase or a single number, not a long sentence and solve the following question and put your final answer within \\boxed{{}}.

Question: {}""",

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

"student_elaboration_reference": 
"""You are a meticulous and analytical Document VQA Expert. Your mission is to refine a specific part of an answer which tends to be incorrect to a question about a given document. You must focus *only* on the text provided within the `<replace>` tags and improve it based on the document's visual and textual evidence. There are some common mistakes in the original part, such as: 1. misperception of the document (for example, ignoring the existance of figures, tables, urls, etc.); 2. misinterpretation of the document (for example, misreading a number, or misinterpreting a table); 3. hallucination of the answer (for example, making up a number that is not in the document). Your ultimate goal is to provide user with a logical, detailed thinking process that leads to the provided reference answer, which is the ground truth of the question.

# Task Instructions:

1.  **Understand the Goal:** Carefully read the **Question**, analyze the provided Document Image, and understand the **Reference Answer**. Your task is to perfect a specific segment of the machine-generated **Full Answer**.
2.  **Isolate the Target:** Locate the text block marked by `<replace>` and `</replace>`. This is the *only* part of the **Full Answer** you are allowed to modify.
3.  **Analyze and Refine:** Follow the three-step analysis below to generate a new, improved version of the target block. The refined version should be more precise, factually accurate according to the document, and logically sound.

# Document Information:
**Question:**
{}
**Reference Answer (Ground Truth):**
{}
**Full Answer (to be partially refined):**
{}
----
<replace>
{}
</replace>
----
{}

---

Follow this three-step method to construct your response:

**Step 1: Contextual Analysis**
First, analyze the block within the `<replace>` tags in the context of the **Full Answer**. 
* **Structural Role:** Is it an introductory part to the document, an strategy outline, an intermediate calculation/reasoning step or a summarization of evidence?
What is its structure role?

**Step 2: Evidence-Based Revision**
Your goal is to perfect the paragraph *based on its structure role*, not to solve the entire problem within this single block.

- If the role of this specific paragraph is **introductory part to the document**, the `Target Answer` should **not** be used to guide the revision of this specific paragraph. Instead, evaluate the paragraph using this hierarchy:
  * **1. Factual Accuracy (Correction First):** Before anything else, is the information 100% correct according to the document? Identify and correct any factual errors, misinterpretations of text, or incorrect data extraction. If there are errors, correcting them is your primary task.
  * **2. Enrichment (Detail and Comprehensiveness):** If the paragraph is factually correct, can its descriptive richness be improved? Your sole focus here is to make this paragraph a more **comprehensive and exhaustive description** of the relevant content **directly observable from the document**. To achieve this, **ADD** relevant, specific details (such as exact figures, names, dates, numbers or other key characteristics) that enrich the existing correct information. **Crucially, completely disregard the user's question and the `Target Answer` in this step, as they are not relevant to this paragraph's function.** This enrichment is about building a rich, foundational understanding of the document's content itself, independent of the final question.

- If the role of this paragraph is a **strategy outline**, ensure it clearly and precisely articulates the overall plan for deriving the answer from the document. Evaluate its effectiveness based on these criteria:
  * **1. Strategic Accuracy & Coherence:** Check the question again, is the outlined strategy fundamentally sound and directly relevant to answering the Document VQA question, or it misunderstands the question and propose a irrelavant plan? Does it logically flow and accurately represent the intended approach? If current strategy misunderstands the question, read the question carefully and propose a new overall plan. If there are inaccuracies or logical gaps in the strategy, identify and correct them, ensuring it aligns perfectly with the problem-solving requirements.


- If this paragraph serves as an **intermediate reasoning step**, your primary task is to ensure its logical soundness and its ability to robustly lead to the **Reference Answer**. Compare your reasoning process with the **Reference Answer**. Check if your reasoning process logically and consistently leads to this answer. If it doesn't, Please evaluate, reflect and refine this paragraph based on the following three points, for each point, state your evluation process detailedly in the analysis: 
  * **1. Secure understanding of the question:**
    * 1. Re-read the question carefully: Thoroughly review the question you are addressing. Check if the misunderstanding of the question is the root cause of the problem.
    * 2. Assess relevance to reasoning step: Ensure that your current reasoning step is directly relevant to the question. This helps prevent misinterpretations, such as over-analyzing the question or fundamentally misunderstanding the core objective, which can lead to incorrect reasoning. 
  * **2. Accuracy of Information Location:** 
    * 1. Verify the located area: Confirm that the answer have accurately identified and located the specific area within the document that contains the relevant evidence (e.g., a particular table, a specific paragraph, or a figure). If the question cannot be answered, ensure that the answer have found no relevant information in the document to support a different conclusion.
    * 2. Correct immediately: If you find that the location is incorrect, please relocate it promptly.
  * **3. Accuracy of Information Interpretation:** After confirming that you have located the correct information (such as figures, specific texts, graphs, etc.), please verify the accuracy of your interpretation from these three perspectives: 
    * 1. Focus on core relevance: Check if you are focusing on the part of the located area that is most directly relevant to the question, for example, a specific column in a table or key data points in a graph. 
    * 2. Error-free information extraction: Ensure that the information you extract is completely faithful to the original source, avoiding any hallucination or deviation.

- If the role of this paragraph is **summarization of evidence**, ensure it accurately reflects the key points extracted from the document. If it is not summarizing the evidence correctly, identify the key points and summarize them in a more precise manner. If it is summarizing the evidence correctly, but lacks detail, add more detailed information that supports the summary.

Throughout the revision, you MUST ensure the paragraph sticks to its **original purpose**. Ensure the revised paragraph does **not** prematurely draw conclusions or state the final answer if that wasn't its original role. For example, if the paragraph is an initial data extraction or intermidiate reasoning steps, you MUST NOT derive the answer to the question in your revised version. Do not expand the scope of the paragraph.

**Step 3: Construct the Refined Block**
Based on your analysis in Step 2, rewrite the paragraph. Your new version must:
* **Be a drop-in replacement:** It must seamlessly fit back into the original **Full Answer** without creating redundancy or logical breaks with the surrounding text and maintain its primary role analyzed in Step 1 and Step 2.
* **Be factually grounded:** Every piece of information must be directly verifiable from the provided Document Image.
* **Correct all identified errors:** Fix any inaccuracies or logical fallacies from the original paragraph.

---

Based on the guide above, you are to refine **only** the section marked for replacement below.
<replace>
{}
</replace>

First, present your analysis following the three-step method within `<analysis></analysis>` tags. Then, provide the improved, refined paragraph within `<refine></refine>` tags.

<analysis>
[Your detailed analysis based on Step 1, 2, and 3 goes here.]
</analysis>

<refine>
[Your new, polished paragraph goes here.]
</refine> 
""",

"student_elaboration_reference_bbox": 
"""You are a meticulous and analytical Document VQA Expert. Your mission is to refine a specific part of an answer which tends to be incorrect to a question about a given document. You must focus *only* on the text provided within the `<replace>` tags and improve it based on the document's visual and textual evidence. There are some common mistakes in the original part, such as: 1. misperception of the document (for example, ignoring the existance of figures, tables, urls, etc.); 2. misinterpretation of the document (for example, misreading a number, or misinterpreting a table); 3. hallucination of the answer (for example, making up a number that is not in the document). Your ultimate goal is to provide user with a logical, detailed thinking process that leads to the provided reference answer, which is the ground truth of the question.

# Task Instructions:

1.  **Understand the Goal:** Carefully read the **Question**, analyze the provided Document Image, and understand the **Reference Answer**. Your task is to perfect a specific segment of the machine-generated **Full Answer**.
2.  **Isolate the Target:** Locate the text block marked by `<replace>` and `</replace>`. This is the *only* part of the **Full Answer** you are allowed to modify.
3.  **Analyze and Refine:** Follow the three-step analysis below to generate a new, improved version of the target block which contains **grounding box** in the reasoning trace and should be more precise, factually accurate according to the document, and logically sound.

# Document Information:
**Question:**
{}
**Reference Answer (Ground Truth):**
{}
**Reference Grounding Box:**
{}
**Full Answer (to be partially refined):**
{}
----
<replace>
{}
</replace>
----
{}

---

Follow this three-step method to construct your response:

**Step 1: Contextual Analysis**
First, analyze the block within the `<replace>` tags in the context of the **Full Answer**. 
* **Structural Role:** Is it an introductory part to the document, an strategy outline, an intermediate calculation/reasoning step or a summarization of evidence?
What is its structure role?

**Step 2: Evidence-Based Revision**
Your goal is to perfect the paragraph *based on its structure role*, not to solve the entire problem within this single block.

- If the role of this specific paragraph is **introductory part to the document**, the `Target Answer` should **not** be used to guide the revision of this specific paragraph. Instead, evaluate the paragraph using this hierarchy:
  * **1. Factual Accuracy (Correction First):** Before anything else, is the information 100% correct according to the document? Identify and correct any factual errors, misinterpretations of text, or incorrect data extraction. If there are errors, correcting them is your primary task.
  * **2. Enrichment (Detail and Comprehensiveness):** If the paragraph is factually correct, can its descriptive richness be improved? Your sole focus here is to make this paragraph a more **comprehensive and exhaustive description** of the relevant content **directly observable from the document**. To achieve this, **ADD** relevant, specific details (such as exact figures, names, dates, numbers or other key characteristics) that enrich the existing correct information. **Crucially, completely disregard the user's question and the `Target Answer` in this step, as they are not relevant to this paragraph's function.** This enrichment is about building a rich, foundational understanding of the document's content itself, independent of the final question.

- If the role of this paragraph is a **strategy outline**, ensure it clearly and precisely articulates the overall plan for deriving the answer from the document. Evaluate its effectiveness based on these criteria:
  * **1. Strategic Accuracy & Coherence:** Check the question again, is the outlined strategy fundamentally sound and directly relevant to answering the Document VQA question, or it misunderstands the question and propose a irrelavant plan? Does it logically flow and accurately represent the intended approach? If current strategy misunderstands the question, read the question carefully and propose a new overall plan. If there are inaccuracies or logical gaps in the strategy, identify and correct them, ensuring it aligns perfectly with the problem-solving requirements.


- If this paragraph serves as an **intermediate reasoning step**, your primary task is to ensure its logical soundness and its ability to robustly lead to the **Reference Answer**. Compare your reasoning process with the **Reference Answer**. Check if your reasoning process logically and consistently leads to this answer. If it doesn't, Please evaluate, reflect and refine this paragraph based on the following three points, for each point, state your evluation process detailedly in the analysis: 
  * **1. Secure understanding of the question:**
    * 1. Re-read the question carefully: Thoroughly review the question you are addressing. Check if the misunderstanding of the question is the root cause of the problem.
    * 2. Assess relevance to reasoning step: Ensure that your current reasoning step is directly relevant to the question. This helps prevent misinterpretations, such as over-analyzing the question or fundamentally misunderstanding the core objective, which can lead to incorrect reasoning. 
  * **2. Accuracy of Information Location:** 
    * 1. Verify the located area: Confirm that the answer have accurately identified and located the specific area within the document that contains the relevant evidence (e.g., a particular table, a specific paragraph, or a figure). If the question cannot be answered, ensure that the answer have found no relevant information in the document to support a different conclusion.
    * 2. Correct immediately: If you find that the location is incorrect, please relocate it promptly.
  * **3. Accuracy of Information Interpretation:** After confirming that you have located the correct information (such as figures, specific texts, graphs, etc.), please verify the accuracy of your interpretation from these three perspectives: 
    * 1. Focus on core relevance: Check if you are focusing on the part of the located area that is most directly relevant to the question, for example, a specific column in a table or key data points in a graph. 
    * 2. Error-free information extraction: Ensure that the information you extract is completely faithful to the original source, avoiding any hallucination or deviation.

- If the role of this paragraph is **summarization of evidence**, ensure it accurately reflects the key points extracted from the document. If it is not summarizing the evidence correctly, identify the key points and summarize them in a more precise manner. If it is summarizing the evidence correctly, but lacks detail, add more detailed information that supports the summary.

Throughout the revision, you MUST ensure the paragraph sticks to its **original purpose**. Ensure the revised paragraph does **not** prematurely draw conclusions or state the final answer if that wasn't its original role. For example, if the paragraph is an initial data extraction or intermidiate reasoning steps, you MUST NOT derive the answer to the question in your revised version. Do not expand the scope of the paragraph.

**Step 3: Add Grounding Box for Reasoning Path**
In this step, you will add grounding box information to the refinement part by only using the `Reference Grounding Box` information. Reference Grounding Box is a dictionary that contains the grounding box information for the reasoning path. The keys of the dictionary are specific texts in the document which might be useful to answering the question, and the values are the corresponding corresponding grounding box. You should add this information to the reasoning path in a way that is consistent with the original paragraph.
For example, if the provides `Reference Grounding Box` is: {{'Figure 1': [0.1, 0.3, 0.5, 0.7]}}, then you should add the grounding box information if 'Figure 1' is mentioned in the paragraph, like this: "Figure 1 (bounding box: [0.1, 0.3, 0.5, 0.7]) shows the relevant information." If the grounding box is not mentioned in the paragraph, you should not add it.

**Step 4: Construct the Refined Block**
Based on your analysis in Step 2 and Step3, rewrite the paragraph. Your new version must:
* **Be a drop-in replacement:** It must seamlessly fit back into the original **Full Answer** without creating redundancy or logical breaks with the surrounding text and maintain its primary role analyzed in Step 1 and Step 2.
* **Be factually grounded:** Every piece of information must be directly verifiable from the provided Document Image.
* **Correct all identified errors:** Fix any inaccuracies or logical fallacies from the original paragraph.

---

Based on the guide above, you are to refine **only** the section marked for replacement below.
<replace>
{}
</replace>

First, present your analysis following the three-step method within `<analysis></analysis>` tags. Then, provide the improved, refined paragraph within `<refine></refine>` tags.

<analysis>
[Your detailed analysis based on Step 1, 2, and 3 goes here.]
</analysis>

<refine>
[Your new, polished paragraph goes here.]
</refine> 
"""
}

think_prefix = "<think>\n"

@ray.remote
def generate(inputs, model: LM, num_rollouts=None, is_vl=True, isgreedy=True, **kwargs):
    results = []
    if is_vl:
        assert len(inputs)==1 
        completions, temperature = model.generate_vl_vllm_server(inputs[0], num_rollouts, isgreedy, **kwargs)
        return completions 

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
    text_inputs: List[List[Any]],
    images: List[Any],
    tokenizer: Any,
    model: Any,
    prompt_suffix: str,
    # NOTE: use_fewshot is an unused parameter in this function.
    use_fewshot: bool,
    num_rollouts: int,
    is_greedy: bool = True,
    **kwargs,
) :
    """
    Constructs prompts from templates and submits them to the model for generation.

    Args:
        template_role: The key for the desired prompt template.
        system_prompt: An optional system message to guide the model's behavior.
        text_inputs: A list of lists, where each inner list contains the arguments for a prompt template.
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
    formatted_queries = [template.format(*inp) for inp in text_inputs]
    
    prompts = []
    for query in formatted_queries:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ] + 
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(path)}"}}
                    for path in images
                ]
            }
        ]


        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Construct the final prompt string.
        # prompt = make_prompt(tokenizer, messages)
        # prompt += prompt_suffix
        # prompts.append(prompt)
        prompts.append(messages)

    # Launch the remote generation task using Ray.
    generation_task = generate.remote(prompts, model, num_rollouts, is_vl=True, isgreedy=is_greedy, **kwargs)
    return generation_task, prompts


class RefinementProcessor:
    """Handles the iterative refinement process for a single generated response."""

    def __init__(self, node: Any, tokenizer: Any, model: Any, stop_threshold: float, max_steps: int, num_expansion: int = 2, initial_template: str = "student_generation_initial"):
        self.node = node
        self.tokenizer = tokenizer
        self.model = model
        self.stop_threshold = stop_threshold
        self.max_steps = max_steps
        self.num_expansion = num_expansion
        self.initial_template = initial_template

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
            
            # use the refinement replace the original segment.
            chosen_text = best_candidate['refinement'] if best_candidate else segment_to_refine
            choice = best_candidate['id'] if best_candidate else "original"

            finalized_thinking_steps.append(chosen_text)
            
            # Log the details of this refinement step.
            initial_info[f"refine_thinking_step_No.{i+1}"] = {
                'segment_to_refine': segment_to_refine,
                'choice': choice,
                'chosen_refinement': chosen_text,
                'after_avg_token_logp': best_ppl_so_far[0],
                'possible_refinements': all_candidates,
            }

            # # Stop early if the perplexity has reached the desired threshold.
            # if best_ppl_so_far[0] < self.stop_threshold:
            #     print("===> Stopping refinement early as PPL threshold reached.")
            #     # Append the rest of the original segments without refinement.
            #     finalized_thinking_steps.extend(s.strip() for s in thinking_segments[i+1:])
            #     break
        
        initial_info['corrected_thinking'] = finalized_thinking_steps
        
        # If any refinement was done, perform one final rollout with the complete refined thinking.
        if finalized_thinking_steps:
            initial_info['finalized_avg_token_logp'] = best_ppl_so_far[0]
            new_thinking_prompt = "<think>\n" + "\n\n".join(finalized_thinking_steps)
            rollout_texts, _, _, _ = direct_rollout(self.node, new_thinking_prompt, n_sample=1, role='student_generation_initial')
            initial_info['finalized_response'] = rollout_texts[0]

        return initial_info

    def _refine_one_step(self, before: str, current: str, after: str) -> Tuple[Optional[Dict], Tuple[float, List], List[Dict]]:
        """Generates and evaluates possible refinements for a single thinking segment."""
        num_samples = self.num_expansion
        # The arguments are: question, reference_answer, text_before, text_to_replace, text_after, original_text
        prompt_args = (self.node.memory['q'], self.node.memory['ref'], before, current, after, current)
        images = self.node.memory.get('image', [])
        rollouts_obj, _ = get_model_output(
            template_role='student_elaboration_reference', 
            system_prompt=default_sys, 
            text_inputs=[prompt_args],
            images=images,  
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

        # prompt should change here when the template changes!
        student_generation_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": templates[self.initial_template].format(self.node.memory['q'])}
                ] + [
                    {"type": "image_url", "image_url" : {"url": f"data:image/jpeg;base64,{image_to_base64(path)}"}}
                    for path in images
                ]
            }
        ]
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
            
            # # Evaluate the new thinking process by calculating the perplexity of the reference answer.
            manager = PosteriorManager('student_generation', default_sys, student_generation_conversation, self.tokenizer, self.model, self.node.memory['ref'])
            _, posterior_prefix = manager.prepare(think_prefix, recomposed_thinking, "")
            # ppl_obj, _ = manager.submit(posterior_prefix)
            # # The 'compute' method returns a tuple (log_perplexity, debug_info).
            # new_perplexity_result = manager.compute(ppl_obj)

            # print(f"Refinement candidate {i} | New log PPL: {new_perplexity_result[0]}")
            
            candidate_info = {
                'id': i,
                'refinement': refinement_text,
                'generator': self.model.model_name,
                'raw_output': rollout_text,
                'raw_input_for_posterior': posterior_prefix,
                # 'avg_token_logp': new_perplexity_result[0],
            }
            refinement_candidates.append({f'expansion_No.{i}_of_segment': candidate_info})

            # # If this candidate is better than the best one so far, update it.
            # if new_perplexity_result[0] < best_perplexity[0]:
            #     best_perplexity = new_perplexity_result
            #     best_candidate_info = candidate_info
            best_candidate_info = candidate_info
            best_perplexity = (1000.0, [])  # Placeholder, as we are not computing perplexity here.
            
                
        return best_candidate_info, best_perplexity, refinement_candidates
    
class PosteriorManager:
    """Calculates the log probability of a reference answer given a thinking process."""
    def __init__(self, role: str, system_prompt: str, conversation: List[Any], tokenizer: Any, model: Any, ref_answer: str):
        self.role = role
        self.system_prompt = system_prompt
        self.conversation = conversation
        self.tokenizer = tokenizer
        self.model = model
        self.ref_answer = ref_answer
        
    def prepare(self, think_prefix: str, thinking_process: str, latent_thinking: str) -> Tuple[str, str]:
        """Prepares the prompt for posterior probability calculation."""
        # The part of the prompt before the reference answer.
        self.prefix_before_answer = f"{think_prefix}{thinking_process}\n{latent_thinking}</think>\n<answer>\\boxed{{"
        # The full prompt including the reference answer.
        self.posterior_prefix = f"{self.prefix_before_answer}{self.ref_answer}}}</answer>"
        return self.prefix_before_answer, self.posterior_prefix
    
    def submit(self, assitant_response: str) -> Tuple[Any, List[str]]:
        """Submits the prompt to the model to get token log probabilities."""
        conversation = self.conversation + [
            {'role':'assistant',
            'content': assitant_response}
        ]
        # TODO: need to tell the model not to sample the ending tokens.
        rollouts_obj = generate.remote([conversation], self.model, num_rollouts=1, is_vl=True, isgreedy=False, prompt_only=True)
    
        self.rollouts_obj = rollouts_obj
        return self.rollouts_obj, conversation

    def compute(self, rollouts_obj: Any) -> Tuple[float, List[Dict]]:    
        """Computes the log perplexity of the reference answer from the model's output."""
        _, prompt_logprobs, _ = ray.get(rollouts_obj)
        if not prompt_logprobs: return 1000.0, [], []
        current_prompt_logprobs = prompt_logprobs[0]
        
        # This complex logic is used to find the exact tokens corresponding to the reference answer.
        offsets = self.tokenizer(self.posterior_prefix, return_offsets_mapping=True).offset_mapping
        tokens = self.tokenizer.tokenize(self.posterior_prefix) 
        start_char_index = len(self.prefix_before_answer)
        end_char_index = start_char_index + len(self.ref_answer)
        assert self.posterior_prefix[start_char_index:end_char_index] == self.ref_answer

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
        # for logp_dict in logprob_slice: 
            # The dictionary may have multiple keys; the first one corresponds to the prompt token.
            # major_key = list(logp_dict)[0]
            # logp = logp_dict[major_key]['logprob']
        for logp in logprob_slice:
            answer_logps.append(logp)
            
        # Calculate the negative mean log probability (log perplexity). A lower value is better.
        log_perplexity = -np.mean(answer_logps) if answer_logps else 1000.0
        
        answer_tokens = tokens[-tokens_to_go_back : -tokens_to_go_back + answer_token_span]
        # For debugging, gather the log probability info for tokens surrounding the answer.
        debug_slice = current_prompt_logprobs[-tokens_to_go_back-2 : -tokens_to_go_back + answer_token_span + 2]
        # debug_logprob_info = [logp_dict[list(logp_dict)[0]] for logp_dict in debug_slice]
        debug_logprob_info = debug_slice        
        return log_perplexity, debug_logprob_info, answer_tokens


class NumpyEncoder(json.JSONEncoder):
    """
    A special JSON encoder for NumPy types.
    It converts NumPy integers, floats, and arrays into standard Python types.
    """
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Handle NumPy floats
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError for other types
        return super(NumpyEncoder, self).default(obj)

import base64

# 读取图像并转换为 Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def direct_rollout(node, prefix, n_sample=1, role=None):
    
    role, sysp, in_keys = ('student_generation' if role is None else role), default_sys, ['q', ] 
    node.memory['gen_role'] = role
    imagelist = node.memory['image'] 
    inputs = [node.memory[k] for k in in_keys]
    tok, model = lm_tokenizer, lm_model
    node.memory['generator'] = model.model_name
    inp = [inputs]
    print(f"===> {role}: {n_sample} for {len(inp)} queries")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": templates[role].format(*inputs)}
            ] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(path)}"}}
                for path in imagelist
            ]
        }
    ]
    rollouts_obj = generate.remote([conversation], model, n_sample, is_vl=True, isgreedy=False)
    rollout_texts, rollout_logps, rollout_offsets = ray.get(rollouts_obj)
    del rollouts_obj
    return rollout_texts, rollout_logps, rollout_offsets, conversation



@ray.remote
def process_item(
    item: Dict[str, Any],
    file_prefix: str,
    rank: int,
    n_sample: int, 
    initial_template: str, 
    configs=dict()
):
    uid = item.get('qid', 'unknown')
    imagelist = item['image']
    output_fname = f"{file_prefix}_{uid}_rk{rank}_meta.jsonl"
    if glob(f"{output_fname}*"):
        print(f"Skipping existing item: {uid}")
        return True
    
    stop_thresh = configs['processing']['stop_thresh']
    max_step = configs['processing']['max_step']
    num_expansion = configs['processing']['num_expansion']

    node = Node(ref=item['answer'], raw_q=item['question'], info={'qid': uid, 'image':imagelist})
    node.memory.update({k:v for k,v in item.items() if not isinstance(v, np.ndarray)})
    
    fname = output_fname

    tok, model = lm_tokenizer, lm_model

    # 1. Initial Rollout
    # n_sample = 1
    rollout_texts, rollout_logps, rollout_offsets, conversation = direct_rollout(node, "", n_sample, role=initial_template) # student_generation_short or student_generation_initial
    conversations = [conversation for _ in range(n_sample)]
    outcomes = []
    # expanded_prompts = [pp for pp in real_input_prompts for _ in range(n_sample)] # seems not needed, delete tmp
    
    for roll, token_logps, token_to_text_offsets, conversation in zip(rollout_texts, rollout_logps, rollout_offsets, conversations):
        ntoken = len(token_logps)
        info = dict(ntokens=ntoken,)
        
        # separate thinking and code answer
        think_process_match = re.search("<think>(.*?)</think>", roll, re.DOTALL)
        if think_process_match:
            think_process = think_process_match.group(1).strip('\n')
        else:
            think_process = roll.strip('\n')
        
        # think_process = roll.split("\nAnswer:")[0].strip('\n')
            
        match = re.search("(\\boxed{.*?})", roll, re.DOTALL)
        answer_code = None
        if match:
            answer_code = match.group(1)

        # breakdown thinking to steps 
        # aa,bb,thinking_segments = breakdown_steps(thinking)
        aa, bb, thinking_segments = breakdown_steps(think_process)
        
        # get PPL of initial rollout
        role, sysp = 'student_generation', default_sys 
        manager = PosteriorManager(role, sysp, conversation, tok, model, node.memory['ref'])
        posterior_prefix1, posterior_prefix = manager.prepare(think_prefix, think_process, "")
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
        num_expansion=num_expansion,
        initial_template=initial_template
    )
    
    updated_results = []
    for info in outcomes:
        refined_info = refinement_processor.run(info)
        updated_results.append(refined_info if refined_info is not None else info)
    
    node.memory['alist'] = updated_results
    # node.memory['alist'] = outcomes
    
    flag = False 
    result_file = f"{fname}_meta.jsonl"
    with open(result_file, "w") as file:
        line = json.dumps(node.memory,indent=2,ensure_ascii=False,cls=NumpyEncoder)

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
    initial_template = config["processing"]["initial_template"]

    # Create output directory
    os.makedirs(os.path.dirname(config["output"]["file_prefix"]), exist_ok=True)
    
    

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
    if path.endswith('.json'):
        print(f"Loading data from JSON file: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.endswith('.parquet'):
        print(f"Loading data from Parquet file: {path}")
        df = pd.read_parquet(path)
        data = df.to_dict('records') # question, image, answer, qid
    
    num_each = len(data) // args.total_ranks + 1
    rank_data = data[args.rank * num_each : (args.rank + 1) * num_each]
    print(f"Rank {args.rank}/{args.total_ranks}: Processing {len(rank_data)} items.")
    st = time.time()
    # Initialize Ray
    ray.init(num_cpus=args.num_cpus)
    task_futures = [
        process_item.remote(
                item,
                file_prefix,
                args.rank,
                num_rollouts,
                initial_template,
                configs=config
            )
        for item in rank_data
    ]

    # Wait for all tasks to complete
    ray.get(task_futures)
    
    # del task_futures to free memory
    del task_futures

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
