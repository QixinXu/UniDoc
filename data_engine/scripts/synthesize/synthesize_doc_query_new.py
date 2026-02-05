import io
import os
import re
import ray
import time
import yaml
import json
import base64
import random
import argparse
import jsonlines
import transformers

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from typing import List, Any, Optional

from model_utils import LM

default_sys = ""
templates = {
"extract": 
'''
**Role:** You are an expert academic analyst specializing in distilling complex information from research papers.

**Context:** You will be given an image (e.g., a chart, diagram, or figure) and a related text excerpt from a scientific article. The text directly explains or discusses the provided image.

**Task:** Your primary goal is to analyze both the image and the text to extract the most critical key points. These are not just keywords or simple nouns; they should be concise statements that capture the central theme, a specific methodology, a significant finding, or a core concept presented.

**Objective:** The extracted key points will serve as the foundation for generating insightful questions about the material. Therefore, they must be substantive and directly reflect the core information presented in both the image and the text.

**Instructions:**
- Carefully analyze the provided image and its accompanying text.
- Synthesize your understanding to identify 3-5 core ideas or findings.
- Present your output strictly in the following format.
- Your answer MUST use ENGLISH!

**Input**:
related text: {}

''',

"synthesize_easy": 
'''
# Role and Goal
You are an expert instructional designer, tasked with creating a high-quality question-answer pair based on a provided image and its accompanying text. Your goal is to assess a student's ability to synthesize information from visual and textual sources and their attention to detail.

# Input
- An image.
- A related text snippet: {}

# Critical Instructions
1.  **Question Crafting:**
    - The question MUST be answerable ONLY by using the information present in BOTH the image and the text.
    - It should encourage critical observation and information integration, not just simple fact retrieval from one source.
    - The question must be clear, concise, and unambiguous.
    - The question MUST NOT be subjective or open-ended; it should have a definitive answer based on the provided materials.
    - There should be **only** one "what/how/why" question in the output, you must not generate multiple questions, which means outputs like "What ... and how ...?" or "Why...? How...?" is strictly banned, because they contained two questions instead of one. 

2.  **The Golden Rule of Answering:**
    - The provided answer MUST be derived exclusively from the given materials.
    - Do NOT use any external knowledge, assumptions, or logical inferences that are not directly supported by the provided content. The answer should be a verifiable fact from the source.
    - The answer should be within few words or a short sentence, directly providing the answer without additional explanation or context.
''',

"output_format_instructions": '''
# Output Format
Your output must be a single, valid JSON object containing two keys: "question" and "answer". Do not add any text or explanation outside of the JSON structure.

Example:
```json
{
  "question": "Based on the text, what specific feature mentioned is visibly present on the object shown in the image?",
  "answer": "Lunar-grade titanium finish."
}
```
''',

"output_format_instructions_extract": '''
**Output Format**
Your output must be a single, valid JSON object containing two keys: "analysis" and "keypoints". Do not add any text or explanation outside of the JSON structure. The "keypoints" should be a list of strings, each representing a key point extracted from the image and text. The language used should be English.

Example:
```json
{
  "analysis": "Provide a brief analysis of how the image and text are interconnected and what their combined core message is.",
  "keypoints": ['key point A', 'key point B', 'key point C', ...]
}
```
''',

"synthesize_hard_v0": 
'''
# Role and Goal
You are an expert instructional designer, tasked with creating a high-quality question-answer pair based on a provided image and its accompanying text. Your goal is to assess a student's ability to synthesize information from visual and textual sources and their ability to perform REASONING with synthesized information.
Specifically, you will be given two kinds of text snippets:
1. A subtitle that describes the image.
2. A related text snippet that provides additional context or information about the image.

# Input
- An image.
- A subtitle of the image: {}
- A related text snippet: {}

# Critical Instructions
1.  **Question Crafting:**
    - The question MUST be answerable ONLY by using the information present in BOTH the image and the texts.
    - The question should require the student to synthesize information from the image, the subtitle and the related text. All three sources MUST be used to answer the question.
    - The question should test the student's ability from two steps:
        1. Attention to detail: Firstly, the student should observe the image and the texts to gather relevant information.
        2. Reasoning: Secondly, the student should use the gathered information to answer the question.
    - The question must be clear, concise, and unambiguous.
    - The question must not be subjective or open-ended; it should have a definitive answer based on the provided materials. For example, it should not ask "What do you think about the image?" or "What is your opinion on the text?". Instead, it should ask for specific information that can be found in the image and the texts.
    - There should be **only** one "what/how/why" question in the output, you must not generate multiple questions, which means outputs like "What ... and how ...?" or "Why...? How...?" is strictly banned, because they contained two questions instead of one. 
    

2.  **The Golden Rule of Answering:**
    - The provided answer MUST be derived exclusively from the given materials.
    - Do NOT use any external knowledge, assumptions, or logical inferences that are not directly supported by the provided content. The answer should be a verifiable fact from the source.
    - The answer is more preferred to be derived from the related text, rather than the image and the subtitle. The image and subtitle is used to provide necessary context or information.
    - The answer should be within a few words or a short sentence, but NOT single number or single word. It should directly providing the answer without additional explanation or context.
    
''',

"synthesize_hard_v1": 
'''
# Role and Goal
You are an expert instructional designer, tasked with creating a high-quality question-answer pair based on a provided image, its accompanying text and a set of keypoints. Your goal is to assess a student's ability to synthesize information from visual and textual sources and their ability to perform REASONING with synthesized information.
Specifically, you will be given three kinds of text snippets:
1. A subtitle that describes the image.
2. A related text snippet that provides additional context or information about the image.
3. A set of key points extracted from the image and text which are core ideas or findings of the image and text.

# Input
- An image.
- A subtitle of the image: {}
- A related text snippet: {}
- A set of key points extracted from the image and text: {}

# Critical Instructions
1.  **Question Crafting:**
    - The question MUST be answerable ONLY by using the information present in BOTH the image and the texts.
    - The question should require the student to synthesize information from the image, the subtitle and the related text. All three sources MUST be used to answer the question.
    - The question should test the student's ability from two steps:
        1. Attention to detail: Firstly, the student should observe the image and the texts to gather relevant information.
        2. Reasoning: Secondly, the student should use the gathered information to answer the question.
    - The keypoints are provided to give you direction on what kind of information is important in the image and texts, so you can use them to generate a more focused question.
    - It's preferrred to combine multiple key points to generate a challenging question.
    - It's preferred to generate a question that requires multiple steps of reasoning.
    - The question must be clear, concise, and unambiguous.
    - The question must not be subjective or open-ended; it should have a definitive answer based on the provided materials. For example, it should not ask "What do you think about the image?" or "What is your opinion on the text?". Instead, it should ask for specific information that can be found in the image and the texts.
    - There should be **only** one "what/how/why" question in the output, you must not generate multiple questions, which means outputs like "What ... and how ...?" or "Why...? How...?" is strictly banned, because they contained two questions instead of one. 
    

2.  **The Golden Rule of Answering:**
    - The provided answer MUST be derived exclusively from the given materials.
    - Do NOT use any external knowledge, assumptions, or logical inferences that are not directly supported by the provided content. The answer should be a verifiable fact from the source.
    - The answer should to be the ultimate result of a reasoning process using the key points, related text, subtitle and image, rather than just a simple fact from the image or text. The key points are used to provide necessary context or information.
    - The answer should be within a few words or a short sentence, but NOT single number or single word. It should directly providing the answer without additional explanation or context.
    
''',

"synthesize_table": '''
# Role and Goal
You are an expert instructional designer, tasked with creating a high-quality question-answer pair based on a provided table, its markdown content derived from OCR, its corresponding title and its accompanying text. Your goal is to assess a student's ability to extract, calculate, compare or analyze information from tabular and textual sources.

Specifically, you will be given three kinds of text snippets:
1. A markdown content derived from OCR that represents the table, which intends to help the you better understand the table.
2. A subtitle that describes the table.
3. A related text snippet that provides additional context or information about the table.

Also, you will be given an ability that specifies what kind of question to generate, it can be one of the following:
- extraction: The question should require the student to extract specific information from the table.
- calculation: The question should require the student to perform calculations based on the data in the table and information in the texts.
- comparison: The question should require the student to compare different data points or trends within the table and texts.    
- analysis: The question should require the student to analyze trends, patterns, or implications based on the table and texts.


# Input
- An image of a table.
- A markdown content derived from OCR that represents the table: {}
- A subtitle of the image: {}
- A related text snippet: {}
- An ability that specifies what kind of question to generate: {}

# Critical Instructions
1.  **Question Crafting:**
    - The question MUST be answerable ONLY by using the information present in the table and the texts.
    - For **extraction questions**, the question should require the student to extract specific information from the table. For example, the question might ask for a particular value, category, or data point that is clearly presented in the table.
    - For **calculation questions**, the question should require the student to perform calculations based on the data in the table and information in the texts. Typically, the answer should not be directly found in the table or texts, but rather derived from them. For example, the question might ask the difference between two values, the sum of a series of values, a percentage change or how many columns/rows meet a certain condition.
    - For **comparison questions**, the question should require the student to compare different data points or trends within the table and texts. This might involve identifying which value is higher or lower, comparing rates of change, or discussing differences in categories.
    - For **analysis questions**, the question should require the student to analyze trends, patterns, or implications based on the **table and texts**. This question usually requires higher-order thinking skills, such as interpreting data trends, making inferences, or discussing the significance of the data presented, the related text snippet is the most important source for analysis questions. The question MUST be objective and fact-based, rather than asking for personal opinions or subjective interpretations.

2.  **The Golden Rule of Answering:**
    - The provided answer MUST be derived exclusively from the given materials.
    - The answer should be a verifiable fact from the source.
    - The answer should to be the ultimate result of a extraction, calculation, comparison or analysis process using the table and texts.
    - For **extraction questions**, the answer should be a specific value, category, or data point that is clearly presented in the table.
    - For **calculation questions**, the answer should be a numerical value derived from calculations based on the data in the table and information in the texts. 
    - For **comparison questions**, the answer should clearly indicate the result of the comparison, such as which value is higher or lower, or the nature of the difference between categories. The answer should as concise as possible, for example:
        - If the question is "Which value is higher, A or B?", the answer should be "A" or "B", instead of "Value A is higher than Value B".
        - If the question is "What is the difference between A and B?", the answer should be "X" (the numerical difference), instead of "The difference between A and B is X".
    - For **analysis questions**, the answer should be a concise and short statement based on the data trends, patterns, or implications presented in the table and texts.
    
''',

"synthesize_text": '''
# Role and Goal
You are an expert instructional designer. Your primary role is to develop a comprehensive question-answer pair to evaluate a student's reading comprehension and reasoning skills. The questions you create should require the student to analyze and synthesize information presented in a given text snippet.

Specifically, you will be given one kinds of text snippets:
1. An OCR-derived text snippet that represents the text content, which provide you the source material for question generation.

# Input
- An OCR-derived text snippet that represents the text content: {}
- An image of the text which is intended to help you better understand the text layout.

# Critical Instructions
1.  **Question Crafting:**
    - The question MUST be answerable ONLY by using the information present in the texts.
    - The question should require the student to extract specific information from the text. For example, the question might ask for a particular value, category, or data point that is clearly presented in the text.
    - The question can also require the student to conduct simple reasoning based on the text, such as calculating a value given the total and a percentage, or comparing two values.
    - The question is preferred to be a single "what/how/why" question, which means outputs like "What ... and how ...?" or "Why...? How...?" is strictly banned, because they contained two questions instead of one.
    - The question MUST be objective and fact-based, rather than asking for personal opinions or subjective interpretations.

2.  **The Golden Rule of Answering:**
    - The provided answer MUST be derived exclusively from the given materials.
    - The answer should be a verifiable fact from the source.
    - The answer should to be the ultimate result of a extraction, reasoning process using the texts.
    - For extraction questions, the answer should be a specific value, term, category, or data point that is clearly presented in the text. It is critical that you provide this information by quoting the exact phrase or data verbatim from the source text. Do not rephrase or summarize the content.
    - For reasoning questions, the answer should be a numerical value derived from calculations based on the data in the text, or a concise statement that reflects the comparison or analysis of the information presented.
    
''',
}
image_root_dir = '/home/ma-user/work/haozhe/workspace/o1_reflection/release/pdf_images/'


def pretty_name(name: str) -> str:
    return name.replace(" ", "-").replace(".", "-")

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def load_json(file_path: str):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        file_content = json.load(file)
    data = file_content.get('data', {})
    nodes = data.get('nodes', [])
    nodes_num = len(nodes)
    file_id = file_path.split('/')[-1].replace('.json', '')
    return nodes, nodes_num, file_id

def convert_pdf_to_image(pdf_path: str, page_number: int):
    """Convert a specific page of a PDF to an image."""
    from pdf2image import convert_from_path
    # print(f"Converting page {page_number} of {pdf_path} to image...")
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    if images:
        return images[0]
    else:
        raise ValueError("No images found for the specified page number.")

def image_to_base64(image_pil: Image.Image, format: str = "PNG"):
    if not isinstance(image_pil, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    buffered = io.BytesIO()
    try:
        image_pil.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        print(f"Error converting PIL image to pure base64: {e}")
        return ""

def crop_image(image: Image.Image, bbox: list):
    """Crop an image using the provided bounding box."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if len(bbox) != 4:
        raise ValueError("Bounding box must have four elements: [left, upper, right, lower].")
    width, height = image.size
    coordinates = (width * bbox[0], height * bbox[1], width * bbox[2], height * bbox[3])
    return image.crop(coordinates)

def aggregate_childs(node: dict, pdf_path: str, file_id: str = None, idx: int = None):
    """Aggregate child nodes into a single string."""
    type_ = node.get('type', '')

    if type_ == 'apara':
        is_title = node.get('texttitle', False)
        is_figuretitle = node.get('figuretitle', False)
        page_num = node['refer']['locs'][0]['page']
        bbox = node['refer']['locs'][0]['bbox']
        # title
        if is_title:
            return "", 'title', page_num, bbox
        # figure title without childs
        if is_figuretitle and "childs" not in node:
            return node['content'].strip(), 'figuretitle', page_num, bbox
        # text without childs
        if 'childs' not in node:
            return node['content'].strip(), 'text', page_num, bbox
        # figure title and text with childs
        childs = node['childs']
        childs_contents = []
        for child in childs:
            if child['type'] == 'aline':
                childs_contents.append(child['content'])
        
        content_str = ' '.join(childs_contents)
        if is_figuretitle:
            return content_str.strip(), 'figuretitle', page_num, bbox
        return content_str.strip(), 'text', page_num, bbox
    if type_ == 'afigure' or type_ == 'atable':
        bbox = node['refer']['locs'][0]['bbox']
        page = node['refer']['locs'][0]['page']
        page_image = convert_pdf_to_image(pdf_path, page)
        figure = crop_image(page_image, bbox)
        save_dir = os.path.join(image_root_dir, file_id)
        os.makedirs(save_dir, exist_ok=True)
        if type_ == 'afigure':
            save_path = os.path.join(save_dir, f"figure_{page}_{bbox[0]}_{bbox[1]}.png")
        elif type_ == 'atable':
            save_path = os.path.join(save_dir, f"table_{page}_{bbox[0]}_{bbox[1]}.png")
        else:
            raise ValueError(f"Unsupported node type: {type_}")
        if not os.path.exists(save_path):
            figure.save(save_path)
        if type_ == 'afigure':
            image_info = {
                'bbox': bbox,
                'page': page,
                'file_id': file_id,
                'image_path': save_path
            }
            return image_info, 'image', "", []
        elif type_ == 'atable':
            table_info = {
                'bbox': bbox,
                'page': page,
                'file_id': file_id,
                'image_path': save_path,
                'node_idx': idx,
                'md_content': node.get('content', '')
            }
            return table_info, 'table', "", []
        else:
            raise ValueError(f"Unsupported node type: {type_}")
        
    return "", None, "", []

def sort_image(image_infos: list):
    """Sort images by page number and bounding box."""
    # group image by page number
    grouped_images = defaultdict(list)
    for image_info in image_infos:
        page = image_info['page']
        grouped_images[page].append(image_info)
    # sort images by bounding box within each page
    sorted_images = []
    for page, images in grouped_images.items():
        if len(images) == 1:
            sorted_images.append(images[0])
            continue
        # multiple images on the same page
        left_images = [img for img in images if img['bbox'][2] < 0.5]
        right_images = [img for img in images if img['bbox'][0] >= 0.5]
        # consider whether double column
        if left_images and right_images:
            left_images.sort(key=lambda x: x['bbox'][1])  # sort by upper
            right_images.sort(key=lambda x: x['bbox'][1])  
            sorted_images.extend(left_images)
            sorted_images.extend(right_images)
        else:
            images.sort(key=lambda x: (x['bbox'][1]))  # sort by upper
            sorted_images.extend(images)
    return sorted_images
    
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


def get_model_output(
    template_role: str,
    system_prompt: Optional[str],
    text_inputs: List[List[Any]],
    images: List[Any],
    tokenizer: Any,
    model: Any,
    prompt_suffix: str,
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
    output_instructions = templates.get("output_format_instructions", "")
    output_instructions_extract = templates.get("output_format_instructions_extract", "")
   
    if template_role != 'extract':
        formatted_queries = [template.format(*text_inputs) + output_instructions]
    else:
        formatted_queries = [template.format(*text_inputs) + output_instructions_extract]
    
    prompts = []
    for query in formatted_queries:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ] + 
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(path)}"}}
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

def extract_text_between_tags(text: str, start_tag: str, end_tag: str):
    last_block_start = text.rfind(start_tag)
    if last_block_start == -1:
        print("Warning: Start tag not found in text, returning empty string.")
        return ""
    block_end = text.rfind(end_tag)
    start_pos = last_block_start + len(start_tag)
    return text[start_pos:block_end if block_end != -1 else None].strip()

def extract_keypoints(text: str):
    split_str = 'Key points'
    result = re.split(split_str, text, flags=re.IGNORECASE)
    after_tag = result[-1] if len(result) > 1 else result[0]
    print(after_tag)
    after_tag = after_tag.strip().strip("*").strip(":").strip("\n")
    return after_tag
        

class ImageAnchor:
    """an image set as anchor to build synthesized queries"""
    def __init__(self, file_id: str, name: str, pdf_path: str, image: Image.Image, image_path: str, subtitle: str, related_text: str, related_pages: list, hints2bbox: dict):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path  
        self.subtitle = subtitle
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox
        self.tokenizer = lm_tokenizer
        self.model = lm_model
        self.keypoints = []
        self.results = []
        
    def show(self):
        print(f"Image Anchor: {self.name}")
        print(f"Subtitle: {self.subtitle}")
        print(f"Related Text: {self.related_text}")
        print(f"Image Path: {self.image_path}")
            
    def extract_key_point(self, extract_role: str, num_samples: int = 1, points_num: int = 3):
        """
        Extract key points from the image and related text.
        This method should be implemented to extract meaningful information.
        """
        rollouts_obj, _ = get_model_output(
            template_role='extract', 
            system_prompt=default_sys, 
            text_inputs=[self.subtitle + '\n' + self.related_text],
            images=[self.image],  
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix="",
            use_fewshot=False,
            num_rollouts=num_samples, 
            is_greedy=False
        )
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        total_key_points = []
        for i, rollout_text in enumerate(rollout_texts):
            match = re.search("```json(.*?)```", rollout_text, re.DOTALL)
            if match:
                result = match.group(1).strip()
                try:
                    result = json.loads(result)
                    key_points = result.get('keypoints', "")
                except:
                    print(f"Error parsing JSON from response: {rollout_text}")
                    key_points = []
                    pass
            else:
                print("No JSON block found in the response.")
                key_points = ""
                continue
            total_key_points.extend(key_points)
        # # return the most frequent key points
        # if total_key_points:
        #     key_points_counter = Counter(total_key_points)
        #     most_common_key_points = key_points_counter.most_common(points_num)
        #     if self.keypoints == []:
        #         self.keypoints = [key_point for key_point, _ in most_common_key_points]
        # else:
        #     print("No key points extracted.")
        self.keypoints = total_key_points
            
        return

    def synthesize_question_for_keypoint(self, keypoints: list, difficulty: str, num_samples: int = 1, version: str = 'v0'):
        assert difficulty in ['easy', 'hard'], "Difficulty must be either 'easy' or 'hard'."
        # template_role = f'synthesize_{difficulty}_{version}'
        if version == 'v0':
            if difficulty == "easy":
                template_role = 'synthesize_easy'
                prompt_args = [self.subtitle]
            else:
                template_role = 'synthesize_hard_v0'
                prompt_args = [self.subtitle, self.related_text]
        elif version == 'v1':
            if difficulty == "easy":
                template_role = 'synthesize_easy'
                prompt_args = [self.subtitle]
            else:
                template_role = 'synthesize_hard_v1'
                prompt_args = [self.subtitle, self.related_text, keypoints]
        elif version == 'v2':
            raise NotImplementedError("Version v2 is not implemented yet.")
        else:
            raise NotImplementedError(f"Version {version} is not implemented.")
        rollouts_obj, _ = get_model_output(
            template_role=template_role, 
            system_prompt=default_sys, 
            text_inputs=prompt_args,
            images=[self.image],  
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix="",
            use_fewshot=False,
            num_rollouts=num_samples, 
            is_greedy=False
        )
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        synthesized_questions = []
        for i, rollout_text in enumerate(rollout_texts):
            match = re.search("```json(.*?)```", rollout_text, re.DOTALL)
            if match:
                result = match.group(1).strip()
                try:
                    result = json.loads(result)
                    question_text = result['question']
                    answer_text = result['answer']
                except:
                    repaired_result = re.sub(r'\\([^\\])', r'\\\\\\1', result)
                    try:
                        result = json.loads(repaired_result)
                        question_text = result['question']
                        answer_text = result['answer']
                    except:
                        print(f"Error parsing JSON from response: {rollout_text}")
                        question_text = ""
                        answer_text = ""
                    pass
            else:
                print("No JSON block found in the response.")
                question_text = ""
                answer_text = ""
            if question_text and answer_text:
                pass
            else:
                print(f"No {difficulty} question generated for keypoints: '{keypoints}'.")

            synthesized_questions.append({
                "pdf_path": self.pdf_path,
                "figure": self.name,
                "target_image_path": self.image_path,
                "related_pages": self.related_pages,
                "keypoints": keypoints,
                "difficulty": difficulty,
                "question": question_text,
                "answer": answer_text,
                "hints2bbox": self.hints2bbox
            })
        return synthesized_questions

    def synthesize_question(self, synthesize_role: str):
        '''
        Aborted function, use synthesize_question_v0 or synthesize_question_v1 instead.
        '''
        # synthesize role is either easy or hard
        if self.keypoints == []:
            print("current no key points, will call extract_key_point first")
            self.extract_key_point(extract_role='extract')
        for keypoint in self.keypoints:
            if synthesize_role == 'easy':
                result = self.synthesize_question_for_keypoint(keypoint, 'easy', num_samples=1)
            elif synthesize_role == 'hard':
                result = self.synthesize_question_for_keypoint(keypoint, 'hard', num_samples=1)
            else:
                raise ValueError("synthesize_role must be either 'easy' or 'hard'.")
            self.results.extend(result)
    
    def synthesize_question_v0(self, synthesize_role: str):
        '''
        this version we don't use key points, just use the subtitle and related text to synthesize questions
        '''
        result = self.synthesize_question_for_keypoint(keypoints=[], difficulty=synthesize_role, num_samples=1, version='v0')
        self.results.extend(result)
    
    def synthesize_question_v1(self, synthesize_role: str):
        '''
        this version we use key points
        '''
        if self.keypoints == []:
            print("current no key points, will call extract_key_point first")
            self.extract_key_point(extract_role='extract')
        result = self.synthesize_question_for_keypoint(keypoints=self.keypoints, difficulty=synthesize_role, num_samples=1, version='v1')
        self.results.extend(result)
        
        
    def get_results(self):
        """Get the results of synthesized questions."""
        return self.results
    
    def get_keypoints(self):
        """Get the extracted key points."""
        return self.keypoints

class TableAnchor:
    """a table set as anchor to build synthesized queries"""
    def __init__(self, file_id: str, name: str, pdf_path: str, image: Image.Image, image_path: str, markdown_content: str, subtitle: str, related_text: str, related_pages: list, hints2bbox: dict):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path  
        self.markdown_content = markdown_content
        self.subtitle = subtitle
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox
        self.tokenizer = lm_tokenizer
        self.model = lm_model
        self.results = []
        
    def show(self):
        print(f"Table Anchor: {self.name}")
        print(f"Related Text: {self.related_text}")
        print(f"Image Path: {self.image_path}")
            
    def synthesize_question(self, ability: str):
        assert ability in ['extraction', 'calculation', 'comparison', 'analysis']
        template_role = 'synthesize_table'
        prompt_args = [self.markdown_content, self.subtitle, self.related_text, ability]
        rollouts_obj, _ = get_model_output(
            template_role=template_role, 
            system_prompt=default_sys, 
            text_inputs=prompt_args,
            images=[self.image],  
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix="",
            use_fewshot=False,
            num_rollouts=1, 
            is_greedy=False
        )
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        synthesized_questions = []
        for i, rollout_text in enumerate(rollout_texts):
            match = re.search("```json(.*?)```", rollout_text, re.DOTALL)
            if match:
                result = match.group(1).strip()
                try:
                    result = json.loads(result)
                    question_text = result['question']
                    answer_text = result['answer']
                except:
                    repaired_result = result.replace('\\', '\\\\')
                    try:
                        result = json.loads(repaired_result)
                        question_text = result['question']
                        answer_text = result['answer']
                    except:
                        print(f"Error parsing JSON from response: {rollout_text}")
                        question_text = ""
                        answer_text = ""
                    pass
            else:
                print("No JSON block found in the response.")
                question_text = ""
                answer_text = ""
            if question_text and answer_text:
                pass
            else:
                print(f"No {ability} question generated for table: '{self.name}'.")
                return 
            
            synthesized_questions.append({
                "pdf_path": self.pdf_path,
                "table": self.name,
                "target_image_path": self.image_path,
                "related_pages": self.related_pages,
                "ability": ability,
                "question": question_text,
                "answer": answer_text,
                "hints2bbox": self.hints2bbox
            })
        self.results.extend(synthesized_questions)
    
    def get_results(self):
        """Get the results of synthesized questions."""
        return self.results

class TextAnchor:
    '''synthesize pure-text questions'''
    def __init__(self, file_id: str, name: str, pdf_path: str, image: Image.Image, image_path: str, related_text: str, related_pages: list, hints2bbox: dict):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox
        self.tokenizer = lm_tokenizer
        self.model = lm_model
        self.results = []
        
    def synthesize_question(self): # TODO: This function may call extract_key_point first
        """
        Synthesize questions based on the related text.
        This method should be implemented to extract meaningful information.
        """
        template_role = 'synthesize_text'
        rollouts_obj, _ = get_model_output(
            template_role=template_role, 
            system_prompt=default_sys, 
            text_inputs=[self.related_text],
            images=[self.image],  
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix="",
            use_fewshot=False,
            num_rollouts=1, 
            is_greedy=False
        )
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        synthesized_questions = []
        for i, rollout_text in enumerate(rollout_texts):
            match = re.search("```json(.*?)```", rollout_text, re.DOTALL)
            if match:
                result = match.group(1).strip()
                try:
                    result = json.loads(result)
                    question_text = result['question']
                    answer_text = result['answer']
                except:
                    repaired_result = result.replace('\\', '\\\\')
                    try:
                        result = json.loads(repaired_result)
                        question_text = result['question']
                        answer_text = result['answer']
                    except:
                        print(f"Error parsing JSON from response: {rollout_text}")
                        question_text = ""
                        answer_text = ""
                    pass
            else:
                print("No JSON block found in the response.")
                question_text = ""
                answer_text = ""
            if question_text and answer_text:
                pass
            else:
                print(f"No text question generated for text: '{self.name}'.")
                return
            
            synthesized_questions.append({
                "pdf_path": self.pdf_path,
                "text": self.name,
                "target_image_path": self.image_path,
                "related_pages": self.related_pages,
                "question": question_text,
                "answer": answer_text,
                "hints2bbox": self.hints2bbox
            })
        self.results.extend(synthesized_questions)
    
    def get_results(self):
        """Get the results of synthesized questions."""
        return self.results
    
               
@ray.remote
def process_anchor(anchor: ImageAnchor | TableAnchor | TextAnchor, output_dir: str, debug: bool = False):
    anchor_name = pretty_name(anchor.name)
    file_id = pretty_name(anchor.file_id)
    output_file = os.path.join(output_dir, f"{file_id}_{anchor_name}_questions.jsonl")
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping processing for {anchor.name}.")
        return
    
    if isinstance(anchor, ImageAnchor):
        anchor.extract_key_point(extract_role='extract')
        if debug:
            print(f"Extracted key points for anchor {anchor.name}: {anchor.get_keypoints()}")
        
        if anchor.subtitle:
            anchor.synthesize_question_v0(synthesize_role='easy')
        if anchor.related_text:
            anchor.synthesize_question_v1(synthesize_role='hard')
        
        synthesized_questions = anchor.get_results()
        if not synthesized_questions:
            print(f"No synthesized questions for anchor {anchor.name}.")
            return
        with jsonlines.open(output_file, mode='a') as writer:
            for question in synthesized_questions:
                writer.write(question)
    elif isinstance(anchor, TableAnchor):
        if anchor.related_text:
            abiities = ['extraction', 'calculation', 'comparison', 'analysis']
        else:
            abiities = ['extraction', 'calculation', 'comparison']
        for ability in abiities:
            anchor.synthesize_question(ability)
        synthesized_questions = anchor.get_results()
        if not synthesized_questions:
            print(f"No synthesized questions for anchor {anchor.name}.")
            return
        with jsonlines.open(output_file, mode='a') as writer:
            for question in synthesized_questions:
                writer.write(question)
    elif isinstance(anchor, TextAnchor):
        anchor.synthesize_question()
        synthesized_questions = anchor.get_results()
        if not synthesized_questions:
            print(f"No synthesized questions for anchor {anchor.name}.")
            return
        with jsonlines.open(output_file, mode='a') as writer:
            for question in synthesized_questions:
                writer.write(question)
    else:
        raise TypeError(f"Unsupported anchor type: {type(anchor)}. Expected ImageAnchor, TableAnchor, or TextAnchor.")

def main(args: argparse.Namespace):
    config = load_config(args.config_file)
    output_dir = config['output']['file_prefix']
    output_root_dir = os.path.dirname(output_dir)
    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # load json file paths
    print("Loading JSON file paths...")
    json_file_paths = []
    raw_data_path = config['input']['raw_data_path']
    max_num_files = config['input']['max_num_files']
    # for root, dirs, files in os.walk(raw_data_path):
    #     has_pdf = False
    #     has_json = False
    #     json_file_path = None
        
    #     for file in files:
    #         if file.lower().endswith(".pdf"):
    #             has_pdf = True
    #         elif file.lower().endswith(".json"):
    #             has_json = True
    #             json_file_path = os.path.join(root, file)
        
    #     if json_file_path and has_pdf and has_json:
    #         json_file_paths.append(json_file_path)
    with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/arxiv_jsons_with_pdf.txt', 'r') as f:
        for line in f:
            json_file_path = line.strip()
            if os.path.exists(json_file_path):
                json_file_paths.append(json_file_path)
            else:
                print(f"Warning: JSON file {json_file_path} does not exist, skipping.")
    
    if len(json_file_paths) > max_num_files:
        print(f"Found {len(json_file_paths)} JSON files, but only processing the first {max_num_files}.")
        json_file_paths = json_file_paths[:max_num_files]
    print(f"Total {len(json_file_paths)} JSON files to process.")
    
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
        num_rollouts=1,
        tokenizer=lm_tokenizer,
        **config["model"]["model_args"],
    )
    
    anchors = []
    
    # # for test
    # json_file_paths = [
    #     '/home/ma-user/work/data_mllm/datasets/inf-crawl-arxiv/raw/0801.1009/0801.1009.json'
    # ]
    for file_path in tqdm(json_file_paths):
        pdf_path = file_path.replace('.json', '.pdf')
        # load document data
        nodes, nodes_num, file_id = load_json(file_path)
        # print(f"Loaded {nodes_num} nodes from {file_path}")

        figures = []
        texts = []
        figure_titles = []
        
        tables = []

        for idx, node in enumerate(nodes):
            content, type_, page_num, bbox = aggregate_childs(node, pdf_path, file_id, idx)
            if type_ == 'image':
                figures.append(content)
            elif type_ == 'table':
                tables.append(content)
            elif type_ == 'text':
                texts.append((content, page_num, bbox))
            elif type_ == 'figuretitle':
                figure_titles.append((content, page_num, bbox))
            else:
                pass
        
        if tables and args.table:
            table_anchor_prefixes = ['Table ', 'table ']
            table_anchor_suffixes = [' ', ',', '.', ':', '?']
            existing_table_numbers = set()
            for table in tables:
                idx = table['node_idx']
                page_num = table['page']
                table_bbox = table['bbox']
                image_path = table['image_path']
                file_id = table['file_id']
                markdown_content = table['md_content']
                
                hints2bbox = defaultdict(list)
                
                # find subtitle and add bbox info
                subtitle = ""
                if nodes[idx-1].get('tabletitle', False):
                    subtitle = nodes[idx-1].get('content', "").strip()
                    subtitle_bbox = nodes[idx-1].get('locs', [{}])[0].get('bbox', None)
                elif (idx+1 < len(nodes)) and nodes[idx+1].get('tabletitle', False):
                    subtitle = nodes[idx+1].get('content', "").strip()
                    subtitle_bbox = nodes[idx+1].get('locs', [{}])[0].get('bbox', None)
                else:
                    continue
                
                if not subtitle:
                    continue
                
                hints2bbox[subtitle] = subtitle_bbox
                
                # find table name
                pattern = r"Table\s+(\d+)"
                match = re.search(pattern, subtitle, re.IGNORECASE)
                if match:
                    table_number = match.group(1)
                else:
                    continue
                if table_number in existing_table_numbers:
                    continue
                
                # find related text
                related_texts_tuple = []
                for prefix in table_anchor_prefixes:
                    for suffix in table_anchor_suffixes:
                        name = f'{prefix}{table_number}{suffix}'
                        related_texts_tuple.extend([text for text in texts if name in text[0]])
                related_texts = [text[0] for text in related_texts_tuple]
                related_texts_pages = list(set([text[1] for text in related_texts_tuple] + [page_num]))
                related_texts_bboxes = [text[2] for text in related_texts_tuple]
                
                # add bbox info for related texts
                for text, bbox in zip(related_texts, related_texts_bboxes):
                    hints2bbox[text] = bbox
                
                # # test
                # print(f"table number: {table_number}\nsubtitle: {subtitle}\nrelated texts: {related_texts}\nrelated pages: {related_texts_pages}\nimage path: {image_path}")
                
                existing_table_numbers.add(table_number)
                
                name = f"Table {table_number}"
                hints2bbox[name] = table_bbox
                
                anchor = TableAnchor(
                    file_id=file_id,
                    name=name, 
                    pdf_path=pdf_path,
                    image=Image.open(image_path), 
                    image_path=image_path,
                    markdown_content=markdown_content,
                    subtitle=subtitle,
                    related_text='\n\n'.join(related_texts),
                    related_pages=related_texts_pages,
                    hints2bbox=hints2bbox
                )
                
                anchors.append(anchor)
            
        if texts and args.pure_text:
            page2texts = defaultdict(list)
            page2texts2bbox = defaultdict(defaultdict)
            image_dir = os.path.join(image_root_dir, file_id)
            os.makedirs(image_dir, exist_ok=True)
            for idx, (content, page_num, bbox) in enumerate(texts):
                page2texts[page_num].append(content)
                page2texts2bbox[page_num][content] = bbox 
            
            if len(page2texts) > 10:
                # randomly sample 10 pages if there are more than 10 pages, avoid unhealthy data distribution
                sampled_page_nums = random.sample(list(page2texts.keys()), 10)
            else:
                sampled_page_nums = list(page2texts.keys())
            for page_num in sampled_page_nums:
                # use page number as anchor name
                name = f"Page {page_num}"
                file_id = pretty_name(file_id)
                image_path = os.path.join(image_dir, f"page_{page_num}.png")
                if not os.path.exists(image_path):
                    image_obj = convert_pdf_to_image(pdf_path, page_num)
                    image_obj.save(image_path, format='PNG')
                try:
                    image = Image.open(image_path)
                except:
                    print(f"Error opening image {image_path}, skipping this page.")
                    continue
                
                # related text is the concatenation of all texts on this page
                contents = page2texts[page_num]
                related_texts = "\n\n".join(contents).strip()
                if not related_texts:
                    continue
                
                # add bbox info for related texts
                hintsbbox = page2texts2bbox[page_num]
                
                anchor = TextAnchor(
                    file_id=file_id,
                    name=name, 
                    pdf_path=pdf_path,
                    image=image, 
                    image_path=image_path,
                    related_text=related_texts,
                    related_pages=[page_num],
                    hints2bbox=hintsbbox
                )
                
                anchors.append(anchor)

        if figures and args.figure:
            image_anchor_prefixes = ['Fig. ', 'Figure ', 'figure ', 'Figure', 'figure']
            image_anchor_suffixes = [' ', ',', '.', ':', '?']
            # check if there are figure titles that have the same bbox, if so , merge them
            figure_titles_dict = defaultdict(list)
            for title, page, bbox in figure_titles:
                figure_titles_dict[(page, tuple(bbox))].append(title)
            figure_titles_ = []
            for (page, bbox), titles in figure_titles_dict.items():
                merged_title = " ".join(titles).strip()
                figure_titles_.append((merged_title, page, bbox))
                
            # check if figure nums equals to figure titles
            if len(figures) > len(figure_titles_):
                print(f"Warning: Number of figures ({len(figures)}) is larger than number of figure titles ({len(figure_titles_)}) in {file_id}.")
                continue  # skip this file if they don't match
            
            hints2bbox = defaultdict(list)
            sort_images = sort_image(figures)
            for idx, info in enumerate(sort_images):
                anchor_subtitle = []
                anchor_related_texts = []
                anchor_related_pages = []
                for prefix in image_anchor_prefixes:
                    for suffix in image_anchor_suffixes:
                        name = f'{prefix}{idx+1}{suffix}'  # e.g., 'Fig. 1 ' or 'Figure 1,' or 'figure 1'
                        image_path = info['image_path'] 
                        
                        # find subtitle
                        subtitle_tuple = [title for title in figure_titles_ if name in title[0]]
                        subtitle_texts = [title[0] for title in subtitle_tuple]
                        subtitle_pages = list(set(title[1] for title in subtitle_tuple))
                        subtitle_bboxes = [title[2] for title in subtitle_tuple]
                        
                        # add bbox info for subtitle
                        for text, bbox in zip(subtitle_texts, subtitle_bboxes):
                            hints2bbox[text] = bbox
                            
                        anchor_subtitle.extend(subtitle_texts)
                        anchor_related_pages.extend(subtitle_pages)
                        
                        # find related text
                        related_texts_tuple = [text for text in texts if name in text[0]]
                        related_texts = [text[0] for text in related_texts_tuple]
                        related_texts_pages = list(set(text[1] for text in related_texts_tuple))
                        related_texts_bboxes = [text[2] for text in related_texts_tuple]
                        
                        # add bbox info for related texts
                        for text, bbox in zip(related_texts, related_texts_bboxes):
                            hints2bbox[text] = bbox
                            
                        anchor_related_texts.extend(related_texts)
                        anchor_related_pages.extend(related_texts_pages)

                subtitle = " ".join(anchor_subtitle).strip()
                related_texts = "\n\n".join([text.strip() for text in anchor_related_texts if text.strip()])
                related_pages = list(set(anchor_related_pages)) 
                name = f"Figure {idx+1}"
                
                # add bbox info for figure name
                hints2bbox[name] = info['bbox']
                
                if not subtitle and not related_texts:
                    continue
                anchor = ImageAnchor(
                    file_id=file_id,
                    name=name, 
                    pdf_path=pdf_path,
                    image=Image.open(image_path), 
                    image_path=image_path,
                    subtitle=subtitle, 
                    related_text='\n\n'.join(related_texts),
                    related_pages=related_pages,
                    hints2bbox=hints2bbox
                )
                    
                anchors.append(anchor)
    
    print(f"Total {len(anchors)} anchors found.")
    start = time.time()
    # initialize Ray
    ray.init(num_cpus=args.num_cpus)
    tasks = [
        process_anchor.remote(
                anchor=anchor,
                output_dir=output_dir,
                debug=args.debug
            )
        for anchor in anchors
    ]
    
    with tqdm(total=len(tasks), desc="synthesizing queries...") as pbar:
        while len(tasks) > 0:
            ready, not_ready = ray.wait(tasks)
            pbar.update(len(ready))
            tasks = not_ready
    
    print(f"All tasks completed in {time.time() - start:.2f} seconds. Results saved to {output_dir}.")
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run language model generation and refinement pipeline.")
    parser.add_argument("--config-file", type=str, help="Path to yaml configuration file.")
    parser.add_argument("--port", type=int, required=True, help="Port.")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process.")
    parser.add_argument("--total-ranks", type=int, required=True, help="Total number of parallel processes.")
    parser.add_argument("--num-cpus", type=int, default=32, help="Number of CPUs to allocate for Ray.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--figure", action="store_true", help="Use figure as anchors.")
    parser.add_argument("--table", action="store_true", help="Use table as anchors.")
    parser.add_argument("--pure-text", action="store_true", help="Use pure text as anchors.")
    
    args = parser.parse_args()
    main(args)  