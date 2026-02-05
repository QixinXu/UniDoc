"""
Prompt templates for query synthesis.

This module contains all the prompt templates used for generating
question-answer pairs from document elements (figures, tables, text).
"""

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


TEMPLATES = {
    # =========================================================================
    # Key Point Extraction
    # =========================================================================
    "extract": '''
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

    # =========================================================================
    # Figure/Image Question Synthesis
    # =========================================================================
    "synthesize_easy": '''
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

    "synthesize_hard_v0": '''
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

    "synthesize_hard_v1": '''
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

    # =========================================================================
    # Table Question Synthesis
    # =========================================================================
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
    - For **comparison questions**, the answer should clearly indicate the result of the comparison, such as which value is higher or lower, or the nature of the difference between categories. The answer should as concise as possible.
    - For **analysis questions**, the answer should be a concise and short statement based on the data trends, patterns, or implications presented in the table and texts.
    
''',

    # =========================================================================
    # Pure Text Question Synthesis
    # =========================================================================
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

    # =========================================================================
    # Output Format Instructions
    # =========================================================================
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
  "keypoints": ["key point A", "key point B", "key point C"]
}
```
''',
}


def get_template(template_name: str) -> str:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template to retrieve.
    
    Returns:
        The template string.
    
    Raises:
        KeyError: If the template name is not found.
    """
    if template_name not in TEMPLATES:
        raise KeyError(
            f"Template '{template_name}' not found. "
            f"Available templates: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[template_name]


def format_query_prompt(
    template_name: str,
    *args,
    include_output_format: bool = True
) -> str:
    """
    Format a query synthesis prompt with the given arguments.
    
    Args:
        template_name: Name of the template to use.
        *args: Arguments to format into the template.
        include_output_format: Whether to append output format instructions.
    
    Returns:
        Formatted prompt string.
    """
    template = get_template(template_name)
    formatted = template.format(*args)
    
    if include_output_format:
        if template_name == "extract":
            formatted += TEMPLATES["output_format_instructions_extract"]
        else:
            formatted += TEMPLATES["output_format_instructions"]
    
    return formatted
