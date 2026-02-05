"""
Processors for thinking pattern refinement.

This module provides classes for managing the iterative refinement
of thinking patterns through multiple generations and judge feedback.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from PIL import Image

from ..core.utils import image_to_base64
from .prompts import THINKING_TEMPLATES, VL_SYSTEM_PROMPT


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from model response, handling common formatting issues.
    
    Args:
        response_text: Raw response text from the model.
    
    Returns:
        Parsed JSON as a dictionary, or empty dict on failure.
    """
    match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try escaping problematic backslashes
            try:
                repaired = re.sub(r'\\([^\\])', r'\\\\\\1', json_str)
                return json.loads(repaired)
            except:
                pass
    
    # Try parsing the entire response as JSON
    try:
        return json.loads(response_text)
    except:
        pass
    
    return {}


class RefinementProcessor:
    """
    Processor for iterative refinement of thinking patterns.
    
    This class manages the process of generating initial thinking patterns
    and refining them through multiple iterations with reference answers.
    
    Args:
        model: The language model client.
        tokenizer: The tokenizer for the model.
        max_iterations: Maximum number of refinement iterations.
        use_judge: Whether to use a judge model for feedback.
    
    Example:
        >>> processor = RefinementProcessor(model, tokenizer)
        >>> result = processor.refine(
        ...     question="What is shown in Figure 1?",
        ...     images=[page_image],
        ...     ref_answer="A performance comparison chart"
        ... )
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_iterations: int = 3,
        use_judge: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.use_judge = use_judge
    
    def _build_messages(
        self,
        template_name: str,
        images: List[Union[str, Image.Image]],
        **kwargs
    ) -> List[Dict]:
        """Build message list for model input."""
        template = THINKING_TEMPLATES[template_name]
        query = template.format(**kwargs)
        
        content = [{"type": "text", "text": query}]
        for img in images:
            b64 = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        
        messages = [
            {"role": "system", "content": VL_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        return messages
    
    def generate_initial(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        context: Optional[str] = None,
        num_samples: int = 1
    ) -> List[Dict[str, str]]:
        """
        Generate initial thinking pattern without reference.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            context: Optional context text.
            num_samples: Number of samples to generate.
        
        Returns:
            List of dictionaries with 'analysis' and 'answer' keys.
        """
        if context:
            template_name = "student_generation_with_context"
            kwargs = {"question": question, "context": context}
        else:
            template_name = "student_generation"
            kwargs = {"question": question}
        
        messages = self._build_messages(template_name, images, **kwargs)
        
        completions, _, _ = self.model.generate_vl_vllm_server(
            messages, num_samples, is_greedy=False
        )
        
        results = []
        for completion in completions:
            parsed = parse_json_response(completion)
            results.append({
                "analysis": parsed.get("analysis", ""),
                "answer": parsed.get("answer", ""),
                "raw": completion
            })
        
        return results
    
    def refine_with_reference(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        ref_answer: str,
        previous_analysis: str
    ) -> Dict[str, str]:
        """
        Refine thinking pattern using reference answer.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            ref_answer: The reference/ground truth answer.
            previous_analysis: The previous analysis to refine.
        
        Returns:
            Dictionary with refined 'analysis' and 'answer'.
        """
        messages = self._build_messages(
            "student_elaboration_reference",
            images,
            question=question,
            ref_answer=ref_answer,
            previous_analysis=previous_analysis
        )
        
        completions, _, _ = self.model.generate_vl_vllm_server(
            messages, num_rollouts=1, is_greedy=True
        )
        
        if completions:
            parsed = parse_json_response(completions[0])
            return {
                "analysis": parsed.get("analysis", ""),
                "answer": parsed.get("answer", ""),
                "raw": completions[0]
            }
        
        return {"analysis": "", "answer": "", "raw": ""}
    
    def refine_with_judge(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        ref_answer: str,
        previous_analysis: str,
        judge_feedback: str
    ) -> Dict[str, str]:
        """
        Refine thinking pattern using judge feedback.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            ref_answer: The reference/ground truth answer.
            previous_analysis: The previous analysis to refine.
            judge_feedback: Feedback from the judge model.
        
        Returns:
            Dictionary with refined 'analysis' and 'answer'.
        """
        messages = self._build_messages(
            "posterior_with_judge",
            images,
            question=question,
            ref_answer=ref_answer,
            previous_analysis=previous_analysis,
            judge_feedback=judge_feedback
        )
        
        completions, _, _ = self.model.generate_vl_vllm_server(
            messages, num_rollouts=1, is_greedy=True
        )
        
        if completions:
            parsed = parse_json_response(completions[0])
            return {
                "analysis": parsed.get("analysis", ""),
                "answer": parsed.get("answer", ""),
                "raw": completions[0]
            }
        
        return {"analysis": "", "answer": "", "raw": ""}
    
    def judge(
        self,
        question: str,
        ref_answer: str,
        ai_analysis: str,
        ai_answer: str
    ) -> Dict[str, Any]:
        """
        Get judge feedback on an analysis.
        
        Args:
            question: The question.
            ref_answer: The reference answer.
            ai_analysis: The AI's analysis.
            ai_answer: The AI's answer.
        
        Returns:
            Dictionary with 'is_correct', 'feedback', and 'score'.
        """
        template = THINKING_TEMPLATES["judge_response"]
        query = template.format(
            question=question,
            ref_answer=ref_answer,
            ai_analysis=ai_analysis,
            ai_answer=ai_answer
        )
        
        messages = [
            {"role": "system", "content": VL_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        completions, _, _ = self.model.generate([messages], num_rollouts=1, is_greedy=True)
        
        if completions:
            parsed = parse_json_response(completions[0])
            return {
                "is_correct": parsed.get("is_correct", False),
                "feedback": parsed.get("feedback", ""),
                "score": parsed.get("score", 0)
            }
        
        return {"is_correct": False, "feedback": "", "score": 0}
    
    def refine(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        ref_answer: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform full refinement process.
        
        This method generates an initial thinking pattern and iteratively
        refines it using the reference answer and optionally judge feedback.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            ref_answer: The reference/ground truth answer.
            context: Optional context text.
        
        Returns:
            Dictionary with final 'analysis', 'answer', and 'iterations' history.
        """
        # Generate initial
        initial_results = self.generate_initial(question, images, context)
        
        if not initial_results:
            return {"analysis": "", "answer": "", "iterations": []}
        
        current = initial_results[0]
        iterations = [current]
        
        for i in range(self.max_iterations):
            # Check with judge if enabled
            if self.use_judge:
                judge_result = self.judge(
                    question,
                    ref_answer,
                    current["analysis"],
                    current["answer"]
                )
                
                if judge_result["is_correct"]:
                    break
                
                # Refine with judge feedback
                current = self.refine_with_judge(
                    question,
                    images,
                    ref_answer,
                    current["analysis"],
                    judge_result["feedback"]
                )
            else:
                # Refine with reference only
                current = self.refine_with_reference(
                    question,
                    images,
                    ref_answer,
                    current["analysis"]
                )
            
            iterations.append(current)
        
        return {
            "analysis": current["analysis"],
            "answer": current["answer"],
            "iterations": iterations
        }


class PosteriorManager:
    """
    Manager for posterior thinking distribution.
    
    This class handles sampling from the posterior distribution of thinking
    patterns, managing multiple candidates and selecting the best ones.
    
    Args:
        model: The language model client.
        tokenizer: The tokenizer for the model.
        num_candidates: Number of candidates to generate per question.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        num_candidates: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.processor = RefinementProcessor(model, tokenizer)
    
    def sample_candidates(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        ref_answer: str,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample multiple thinking pattern candidates.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            ref_answer: The reference answer.
            context: Optional context text.
        
        Returns:
            List of candidate thinking patterns.
        """
        # Generate initial candidates
        initial_candidates = self.processor.generate_initial(
            question, images, context, num_samples=self.num_candidates
        )
        
        # Refine each candidate
        refined_candidates = []
        for candidate in initial_candidates:
            refined = self.processor.refine_with_reference(
                question,
                images,
                ref_answer,
                candidate["analysis"]
            )
            
            # Judge the refined candidate
            judge_result = self.processor.judge(
                question,
                ref_answer,
                refined["analysis"],
                refined["answer"]
            )
            
            refined["score"] = judge_result["score"]
            refined["is_correct"] = judge_result["is_correct"]
            refined_candidates.append(refined)
        
        # Sort by score
        refined_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return refined_candidates
    
    def select_best(
        self,
        question: str,
        images: List[Union[str, Image.Image]],
        ref_answer: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select the best thinking pattern from candidates.
        
        Args:
            question: The question to answer.
            images: List of document page images.
            ref_answer: The reference answer.
            context: Optional context text.
        
        Returns:
            The best candidate thinking pattern.
        """
        candidates = self.sample_candidates(question, images, ref_answer, context)
        
        if not candidates:
            return {"analysis": "", "answer": "", "score": 0}
        
        return candidates[0]
