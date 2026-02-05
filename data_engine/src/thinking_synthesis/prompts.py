"""
Prompt templates for thinking pattern synthesis.

This module contains all the prompt templates used for generating
thinking patterns (chain-of-thought reasoning) from question-answer pairs.
"""

# Default system prompt for VL models
VL_SYSTEM_PROMPT = "You are a helpful assistant."


THINKING_TEMPLATES = {
    # =========================================================================
    # Initial Thinking Generation - Student Generation
    # =========================================================================
    "student_generation": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations. 
- Please read all the images first carefully, and then answer the questions.
- When answering questions, examine the images first to find relevant information, and conduct reasoning based on it if necessary.
- The analysis is your process of analyzing the images. This field should contain your observations and reasoning based on the images. An analysis should start with the description of each image and what info in the image could be useful for answering the question, then it should contain the reasoning steps based on the information observed.
- An answer should be within a few words or a short sentence, but NOT single number or single word. It should directly provide the answer without additional explanation or context.
- Both analysis and answer must use ENGLISH!

## Inputs
Question: {question}

## Output Format
Your output must be a single, valid JSON object containing two keys: "analysis" and "answer". Do not add any text or explanation outside of the JSON structure. Your analysis should be detailed and include specific information from the images relevant to the question.
```json
{{
  "analysis": "your analysis",
  "answer": "your answer here"
}}
```
''',

    "student_generation_with_context": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations. 
- Please read all the images first carefully, and then answer the questions.
- When answering questions, examine the images first to find relevant information, and conduct reasoning based on it if necessary.
- The analysis is your process of analyzing the images. This field should contain your observations and reasoning based on the images. An analysis should start with the description of each image and what info in the image could be useful for answering the question, then it should contain the reasoning steps based on the information observed.
- An answer should be within a few words or a short sentence, but NOT single number or single word. It should directly provide the answer without additional explanation or context.
- Both analysis and answer must use ENGLISH!
- Note: context is a piece of text that is related to the question, you can use it as a hint to help you answer the question.

## Inputs
Question: {question}
Context: {context}

## Output Format
Your output must be a single, valid JSON object containing two keys: "analysis" and "answer". Do not add any text or explanation outside of the JSON structure. Your analysis should be detailed and include specific information from the images relevant to the question.
```json
{{
  "analysis": "your analysis",
  "answer": "your answer here"
}}
```
''',

    # =========================================================================
    # Student Elaboration (Refining thinking with ground truth)
    # =========================================================================
    "student_elaboration_reference": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations. 
- Please read all the images first carefully, and then answer the questions.
- When answering questions, examine the images first to find relevant information, and conduct reasoning based on it if necessary.
- The analysis is your process of analyzing the images. This field should contain your observations and reasoning based on the images. An analysis should start with the description of each image and what info in the image could be useful for answering the question, then it should contain the reasoning steps based on the information observed.
- An answer should be within a few words or a short sentence, but NOT single number or single word. It should directly provide the answer without additional explanation or context.
- Both analysis and answer must use ENGLISH!

## Additional Information
Besides the question and images, you will also be provided with:
- reference answer: a reference answer that is derived from the document, which you should use as a guide for your analysis and answer. Don't copy the reference answer directly, but use it to help you understand the question better.
- Your previous analysis: a previous response that you might generate based on the question and images. Some of them is incorrect, so you need to refine your analysis based on the reference answer.

## Inputs
Question: {question}
Reference answer: {ref_answer}
Your previous analysis: {previous_analysis}

## Output Format
Your output must be a single, valid JSON object containing two keys: "analysis" and "answer". Do not add any text or explanation outside of the JSON structure. Your analysis should be detailed and include specific information from the images relevant to the question.
```json
{{
  "analysis": "your refined analysis",
  "answer": "your answer here"
}}
```
''',

    # =========================================================================
    # Posterior Thinking with Judge Feedback
    # =========================================================================
    "posterior_with_judge": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations. 
- Please read all the images first carefully, and then answer the questions.
- When answering questions, examine the images first to find relevant information, and conduct reasoning based on it if necessary.
- The analysis is your process of analyzing the images. This field should contain your observations and reasoning based on the images. An analysis should start with the description of each image and what info in the image could be useful for answering the question, then it should contain the reasoning steps based on the information observed.
- An answer should be within a few words or a short sentence, but NOT single number or single word. It should directly provide the answer without additional explanation or context.
- Both analysis and answer must use ENGLISH!

## Additional Information
Besides the question and images, you will also be provided with:
- reference answer: a reference answer that is derived from the document, which you should use as a guide for your analysis and answer.
- Your previous analysis: a previous response that you might generate based on the question and images.
- Judge feedback: feedback from a judge about your previous analysis, indicating what was wrong and how to improve.

## Inputs
Question: {question}
Reference answer: {ref_answer}
Your previous analysis: {previous_analysis}
Judge feedback: {judge_feedback}

## Output Format
Your output must be a single, valid JSON object containing two keys: "analysis" and "answer". Do not add any text or explanation outside of the JSON structure.
```json
{{
  "analysis": "your refined analysis based on judge feedback",
  "answer": "your answer here"
}}
```
''',

    # =========================================================================
    # Page Retrieval Thinking
    # =========================================================================
    "retrieval_generation": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations.

## Task
Given a question about a multi-page document, you need to:
1. First identify which page(s) contain information relevant to answering the question
2. Then analyze the relevant page(s) to find the answer
3. Provide a step-by-step reasoning process

## Inputs
Question: {question}
Total number of pages: {num_pages}

## Output Format
Your output must be a single, valid JSON object with the following structure:
```json
{{
  "relevant_pages": [1, 3],  // List of 1-indexed page numbers that are relevant
  "page_analysis": {{
    "1": "Description of what's on page 1 relevant to the question...",
    "3": "Description of what's on page 3 relevant to the question..."
  }},
  "reasoning": "Step-by-step reasoning connecting the observations to the answer...",
  "answer": "Your final answer"
}}
```
''',

    "retrieval_generation_with_target": '''
# Your role
You are an advanced AI designed to learn and understand visual documents. Your task is to analyze provided images of document pages and respond to questions based on your observations.

## Task
Given a question about a multi-page document, you need to analyze the target page(s) and reason about the answer.

## Additional Information
- Target pages: the specific page(s) you should focus on to answer the question.
- Reference answer: the correct answer for guidance.

## Inputs
Question: {question}
Target pages: {target_pages}
Reference answer: {ref_answer}

## Output Format
Your output must be a single, valid JSON object:
```json
{{
  "page_analysis": "Detailed analysis of the target page(s)...",
  "reasoning": "Step-by-step reasoning connecting observations to the answer...",
  "answer": "Your answer"
}}
```
''',

    # =========================================================================
    # Judge/Evaluator Templates
    # =========================================================================
    "judge_response": '''
# Your role
You are a strict evaluator assessing the quality of an AI's response to a document understanding task.

## Task
Evaluate whether the AI's analysis and answer are correct based on:
1. The question
2. The reference answer (ground truth)
3. The AI's analysis and answer

## Inputs
Question: {question}
Reference Answer: {ref_answer}
AI's Analysis: {ai_analysis}
AI's Answer: {ai_answer}

## Output Format
Your output must be a single, valid JSON object:
```json
{{
  "is_correct": true/false,
  "feedback": "Detailed feedback explaining what was correct/incorrect and how to improve...",
  "score": 0-10
}}
```
''',

    # =========================================================================
    # Document Description for Retrieval
    # =========================================================================
    "document_description": '''
# Your role
You are an advanced AI designed to understand and describe visual documents.

## Task
Provide a brief description of each page in the document. Focus on:
- The type of content (text, figures, tables, charts)
- Key topics or sections
- Any notable elements

## Inputs
Number of pages: {num_pages}

## Output Format
Your output must be a single, valid JSON object:
```json
{{
  "page_descriptions": {{
    "1": "Brief description of page 1...",
    "2": "Brief description of page 2...",
    ...
  }}
}}
```
''',
}


def get_thinking_template(template_name: str) -> str:
    """
    Get a thinking template by name.
    
    Args:
        template_name: Name of the template to retrieve.
    
    Returns:
        The template string.
    
    Raises:
        KeyError: If the template name is not found.
    """
    if template_name not in THINKING_TEMPLATES:
        raise KeyError(
            f"Template '{template_name}' not found. "
            f"Available templates: {list(THINKING_TEMPLATES.keys())}"
        )
    return THINKING_TEMPLATES[template_name]


def format_thinking_prompt(template_name: str, **kwargs) -> str:
    """
    Format a thinking prompt with keyword arguments.
    
    Args:
        template_name: Name of the template to use.
        **kwargs: Keyword arguments to format into the template.
    
    Returns:
        Formatted prompt string.
    """
    template = get_thinking_template(template_name)
    return template.format(**kwargs)
