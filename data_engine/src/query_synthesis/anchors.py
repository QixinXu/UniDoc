"""
Anchor classes for document elements.

Anchors represent document elements (figures, tables, text) that serve as
the basis for generating question-answer pairs.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import ray
from PIL import Image

from ..core.utils import image_to_base64
from .prompts import TEMPLATES, DEFAULT_SYSTEM_PROMPT, format_query_prompt


# Global references to tokenizer and model (set during initialization)
_tokenizer = None
_model = None


def set_model_references(tokenizer, model):
    """Set global references to the tokenizer and model."""
    global _tokenizer, _model
    _tokenizer = tokenizer
    _model = model


@ray.remote
def _generate_remote(inputs, model, num_rollouts=None, is_vl=True, is_greedy=True, **kwargs):
    """Remote function for Ray-based parallel generation."""
    if is_vl:
        assert len(inputs) == 1
        completions, temperature = model.generate_vl_vllm_server(
            inputs[0], num_rollouts, is_greedy, **kwargs
        )
        return completions
    
    if len(inputs) == 1:
        completions, temperature = model.generate(inputs, num_rollouts, is_greedy, **kwargs)
        return completions
    else:
        results = []
        for inp in inputs:
            completions, temperature = model.generate(inp, num_rollouts, is_greedy, **kwargs)
            results.append(completions)
        return results


def _get_model_output(
    template_role: str,
    system_prompt: Optional[str],
    text_inputs: List[Any],
    images: List[Any],
    tokenizer: Any,
    model: Any,
    num_rollouts: int = 1,
    is_greedy: bool = False,
    **kwargs
):
    """
    Construct prompts and submit to the model for generation.
    
    Returns:
        A tuple of (ray_object, prompts).
    """
    template = TEMPLATES[template_role]
    output_instructions = TEMPLATES.get("output_format_instructions", "")
    output_instructions_extract = TEMPLATES.get("output_format_instructions_extract", "")
    
    # Format the query
    if template_role != 'extract':
        formatted_queries = [template.format(*text_inputs) + output_instructions]
    else:
        formatted_queries = [template.format(*text_inputs) + output_instructions_extract]
    
    prompts = []
    for query in formatted_queries:
        # Build message with images
        content = [{"type": "text", "text": query}]
        for img in images:
            if isinstance(img, str):
                b64 = image_to_base64(img)
            else:
                b64 = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        prompts.append(messages)
    
    # Launch remote generation
    generation_task = _generate_remote.remote(
        prompts, model, num_rollouts, is_vl=True, isgreedy=is_greedy, **kwargs
    )
    return generation_task, prompts


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from model response, handling common formatting issues."""
    match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try escaping backslashes
            try:
                repaired = re.sub(r'\\([^\\])', r'\\\\\\1', json_str)
                return json.loads(repaired)
            except:
                pass
    return {}


class ImageAnchor:
    """
    An anchor representing a figure/image in a document.
    
    This class handles the extraction of key points from figures and
    the synthesis of question-answer pairs based on the figure content.
    
    Args:
        file_id: Unique identifier for the source file.
        name: Name of the figure (e.g., "Figure 1").
        pdf_path: Path to the source PDF file.
        image: PIL Image object of the figure.
        image_path: Path to the saved figure image.
        subtitle: Caption/subtitle of the figure.
        related_text: Text from the document that references this figure.
        related_pages: List of page numbers where this figure is referenced.
        hints2bbox: Optional mapping of text hints to bounding boxes.
    
    Example:
        >>> anchor = ImageAnchor(
        ...     file_id="paper001",
        ...     name="Figure 1",
        ...     pdf_path="/path/to/paper.pdf",
        ...     image=pil_image,
        ...     image_path="/path/to/figure.png",
        ...     subtitle="Performance comparison",
        ...     related_text="As shown in Figure 1...",
        ...     related_pages=[3, 5]
        ... )
        >>> anchor.extract_key_points()
        >>> anchor.synthesize_question_v1("hard")
    """
    
    def __init__(
        self,
        file_id: str,
        name: str,
        pdf_path: str,
        image: Image.Image,
        image_path: str,
        subtitle: str,
        related_text: str,
        related_pages: List[int],
        hints2bbox: Optional[Dict] = None
    ):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path
        self.subtitle = subtitle
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox or {}
        
        self.keypoints: List[str] = []
        self.results: List[Dict] = []
    
    def show(self):
        """Display anchor information."""
        print(f"Image Anchor: {self.name}")
        print(f"File ID: {self.file_id}")
        print(f"Subtitle: {self.subtitle}")
        print(f"Related Text: {self.related_text[:200]}..." if len(self.related_text) > 200 else f"Related Text: {self.related_text}")
        print(f"Image Path: {self.image_path}")
        print(f"Related Pages: {self.related_pages}")
    
    def extract_key_points(self, num_samples: int = 1) -> List[str]:
        """
        Extract key points from the figure and related text.
        
        Args:
            num_samples: Number of extraction attempts.
        
        Returns:
            List of extracted key points.
        """
        global _tokenizer, _model
        
        rollouts_obj, _ = _get_model_output(
            template_role='extract',
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            text_inputs=[self.subtitle + '\n' + self.related_text],
            images=[self.image],
            tokenizer=_tokenizer,
            model=_model,
            num_rollouts=num_samples,
            is_greedy=False
        )
        
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj
        
        total_key_points = []
        for rollout_text in rollout_texts:
            result = _parse_json_response(rollout_text)
            key_points = result.get('keypoints', [])
            if isinstance(key_points, list):
                total_key_points.extend(key_points)
        
        self.keypoints = total_key_points
        return self.keypoints
    
    def synthesize_question_for_keypoint(
        self,
        keypoints: List[str],
        difficulty: str,
        num_samples: int = 1,
        version: str = 'v0'
    ) -> List[Dict]:
        """
        Synthesize questions based on the figure and extracted keypoints.
        
        Args:
            keypoints: List of key points to base questions on.
            difficulty: Either 'easy' or 'hard'.
            num_samples: Number of questions to generate.
            version: Template version ('v0' or 'v1').
        
        Returns:
            List of synthesized question-answer dictionaries.
        """
        global _tokenizer, _model
        
        assert difficulty in ['easy', 'hard'], "Difficulty must be 'easy' or 'hard'."
        
        # Determine template and arguments based on version and difficulty
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
        else:
            raise NotImplementedError(f"Version {version} is not implemented.")
        
        rollouts_obj, _ = _get_model_output(
            template_role=template_role,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            text_inputs=prompt_args,
            images=[self.image],
            tokenizer=_tokenizer,
            model=_model,
            num_rollouts=num_samples,
            is_greedy=False
        )
        
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj
        
        synthesized_questions = []
        for rollout_text in rollout_texts:
            result = _parse_json_response(rollout_text)
            question_text = result.get('question', '')
            answer_text = result.get('answer', '')
            
            if not question_text or not answer_text:
                print(f"[ImageAnchor] No {difficulty} question generated for {self.name}")
                continue
            
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
    
    def synthesize_question_v0(self, synthesize_role: str):
        """Synthesize questions without using key points."""
        result = self.synthesize_question_for_keypoint(
            keypoints=[],
            difficulty=synthesize_role,
            num_samples=1,
            version='v0'
        )
        self.results.extend(result)
    
    def synthesize_question_v1(self, synthesize_role: str):
        """Synthesize questions using extracted key points."""
        if not self.keypoints:
            print("[ImageAnchor] No key points found, extracting first...")
            self.extract_key_points()
        
        result = self.synthesize_question_for_keypoint(
            keypoints=self.keypoints,
            difficulty=synthesize_role,
            num_samples=1,
            version='v1'
        )
        self.results.extend(result)
    
    def get_results(self) -> List[Dict]:
        """Get all synthesized questions."""
        return self.results
    
    def get_keypoints(self) -> List[str]:
        """Get extracted key points."""
        return self.keypoints


class TableAnchor:
    """
    An anchor representing a table in a document.
    
    This class handles the synthesis of question-answer pairs based on
    table content, supporting different question types (extraction,
    calculation, comparison, analysis).
    
    Args:
        file_id: Unique identifier for the source file.
        name: Name of the table (e.g., "Table 1").
        pdf_path: Path to the source PDF file.
        image: PIL Image object of the table.
        image_path: Path to the saved table image.
        markdown_content: Markdown representation of the table from OCR.
        subtitle: Caption/subtitle of the table.
        related_text: Text from the document that references this table.
        related_pages: List of page numbers where this table is referenced.
        hints2bbox: Optional mapping of text hints to bounding boxes.
    """
    
    def __init__(
        self,
        file_id: str,
        name: str,
        pdf_path: str,
        image: Image.Image,
        image_path: str,
        markdown_content: str,
        subtitle: str,
        related_text: str,
        related_pages: List[int],
        hints2bbox: Optional[Dict] = None
    ):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path
        self.markdown_content = markdown_content
        self.subtitle = subtitle
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox or {}
        
        self.results: List[Dict] = []
    
    def show(self):
        """Display anchor information."""
        print(f"Table Anchor: {self.name}")
        print(f"Subtitle: {self.subtitle}")
        print(f"Related Text: {self.related_text[:200]}..." if len(self.related_text) > 200 else f"Related Text: {self.related_text}")
        print(f"Image Path: {self.image_path}")
    
    def synthesize_question(self, ability: str):
        """
        Synthesize a question for a specific ability type.
        
        Args:
            ability: One of 'extraction', 'calculation', 'comparison', 'analysis'.
        """
        global _tokenizer, _model
        
        assert ability in ['extraction', 'calculation', 'comparison', 'analysis']
        
        prompt_args = [self.markdown_content, self.subtitle, self.related_text, ability]
        
        rollouts_obj, _ = _get_model_output(
            template_role='synthesize_table',
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            text_inputs=prompt_args,
            images=[self.image],
            tokenizer=_tokenizer,
            model=_model,
            num_rollouts=1,
            is_greedy=False
        )
        
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj
        
        for rollout_text in rollout_texts:
            result = _parse_json_response(rollout_text)
            question_text = result.get('question', '')
            answer_text = result.get('answer', '')
            
            if not question_text or not answer_text:
                print(f"[TableAnchor] No {ability} question generated for {self.name}")
                return
            
            self.results.append({
                "pdf_path": self.pdf_path,
                "table": self.name,
                "target_image_path": self.image_path,
                "related_pages": self.related_pages,
                "ability": ability,
                "question": question_text,
                "answer": answer_text,
                "hints2bbox": self.hints2bbox
            })
    
    def get_results(self) -> List[Dict]:
        """Get all synthesized questions."""
        return self.results


class TextAnchor:
    """
    An anchor representing pure text content in a document.
    
    This class handles the synthesis of question-answer pairs based on
    text content without associated figures or tables.
    
    Args:
        file_id: Unique identifier for the source file.
        name: Name/identifier for the text (e.g., "Page 5").
        pdf_path: Path to the source PDF file.
        image: PIL Image of the page containing the text.
        image_path: Path to the saved page image.
        related_text: The text content.
        related_pages: List of page numbers.
        hints2bbox: Optional mapping of text hints to bounding boxes.
    """
    
    def __init__(
        self,
        file_id: str,
        name: str,
        pdf_path: str,
        image: Image.Image,
        image_path: str,
        related_text: str,
        related_pages: List[int],
        hints2bbox: Optional[Dict] = None
    ):
        self.file_id = file_id
        self.name = name
        self.pdf_path = pdf_path
        self.image = image
        self.image_path = image_path
        self.related_text = related_text
        self.related_pages = related_pages
        self.hints2bbox = hints2bbox or {}
        
        self.results: List[Dict] = []
    
    def synthesize_question(self):
        """Synthesize a question based on the text content."""
        global _tokenizer, _model
        
        rollouts_obj, _ = _get_model_output(
            template_role='synthesize_text',
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            text_inputs=[self.related_text],
            images=[self.image],
            tokenizer=_tokenizer,
            model=_model,
            num_rollouts=1,
            is_greedy=False
        )
        
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj
        
        for rollout_text in rollout_texts:
            result = _parse_json_response(rollout_text)
            question_text = result.get('question', '')
            answer_text = result.get('answer', '')
            
            if not question_text or not answer_text:
                print(f"[TextAnchor] No question generated for {self.name}")
                return
            
            self.results.append({
                "pdf_path": self.pdf_path,
                "text": self.name,
                "target_image_path": self.image_path,
                "related_pages": self.related_pages,
                "question": question_text,
                "answer": answer_text,
                "hints2bbox": self.hints2bbox
            })
    
    def get_results(self) -> List[Dict]:
        """Get all synthesized questions."""
        return self.results
