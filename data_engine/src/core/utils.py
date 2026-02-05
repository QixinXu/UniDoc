"""
Utility functions for the CogDoc Data Engine.
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.
    
    Example:
        >>> config = load_config("configs/query_synthesis.yaml")
        >>> print(config["model"]["model_name"])
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def image_to_base64(image_input, format: str = "PNG") -> str:
    """
    Convert an image to a base64-encoded string.
    
    Args:
        image_input: Either a file path (str) or a PIL Image object.
        format: Image format for encoding (default: "PNG").
    
    Returns:
        Base64-encoded string of the image.
    
    Example:
        >>> b64_str = image_to_base64("/path/to/image.png")
        >>> b64_str = image_to_base64(pil_image, format="JPEG")
    """
    if isinstance(image_input, str):
        # Input is a file path
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        # Input is a PIL Image
        buffered = io.BytesIO()
        try:
            image_input.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[Utils] Error converting PIL image to base64: {e}")
            return ""
    else:
        raise TypeError(
            f"Expected str or PIL.Image, got {type(image_input)}"
        )


def crop_image(image: Image.Image, bbox: List[float]) -> Image.Image:
    """
    Crop an image using a normalized bounding box.
    
    Args:
        image: PIL Image to crop.
        bbox: Bounding box as [left, upper, right, lower] with values in [0, 1].
    
    Returns:
        Cropped PIL Image.
    
    Example:
        >>> cropped = crop_image(image, [0.1, 0.2, 0.5, 0.8])
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if len(bbox) != 4:
        raise ValueError("Bounding box must have four elements: [left, upper, right, lower].")
    
    width, height = image.size
    coordinates = (
        int(width * bbox[0]),
        int(height * bbox[1]),
        int(width * bbox[2]),
        int(height * bbox[3])
    )
    return image.crop(coordinates)


def convert_pdf_to_image(pdf_path: str, page_number: int) -> Image.Image:
    """
    Convert a specific page of a PDF to an image.
    
    Args:
        pdf_path: Path to the PDF file.
        page_number: Page number to convert (1-indexed).
    
    Returns:
        PIL Image of the specified page.
    
    Requires:
        pdf2image package and poppler-utils installed.
    """
    from pdf2image import convert_from_path
    
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number
    )
    
    if images:
        return images[0]
    else:
        raise ValueError(f"No images found for page {page_number} in {pdf_path}")


def pretty_name(name: str) -> str:
    """
    Convert a name to a filesystem-safe format.
    
    Args:
        name: Original name string.
    
    Returns:
        Sanitized name with spaces and dots replaced by hyphens.
    """
    return name.replace(" ", "-").replace(".", "-")


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy types.
    
    Converts NumPy integers, floats, and arrays into standard Python types
    for JSON serialization.
    
    Example:
        >>> import json
        >>> data = {"value": np.float64(3.14), "array": np.array([1, 2, 3])}
        >>> json.dumps(data, cls=NumpyEncoder)
    """
    
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Handle NumPy floats
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class handle other types
        return super(NumpyEncoder, self).default(obj)


class Node:
    """
    A data structure for managing question-answer pairs and their metadata.
    
    This class is used to track the state of a Q&A item through the
    synthesis pipeline, including the question, reference answer,
    and any associated images or thinking patterns.
    
    Args:
        ref: The reference (ground truth) answer.
        raw_q: The raw question text.
        rational: Optional rationale for the answer.
        target_images: Optional list of target image identifiers.
        gold: Optional gold standard data.
        info: Optional dictionary with additional metadata.
    
    Example:
        >>> node = Node(
        ...     ref="42",
        ...     raw_q="What is the answer to life?",
        ...     info={"qid": "q001", "image": ["page1.png"]}
        ... )
        >>> node.memory['q']
        'What is the answer to life?'
    """
    
    def __init__(
        self,
        ref: str,
        raw_q: str,
        rational: Optional[str] = None,
        target_images: Optional[List[str]] = None,
        gold: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None
    ):
        self.memory = info if info is not None else {}
        self.memory.update({
            'q': raw_q,
            'ref': ref,
            'history': [],
            'rational': rational,
            'target_images': target_images
        })
        self.ref = ref
        self.raw_q = raw_q
        self.rational = rational
        self.target_images = target_images
    
    def show(self):
        """Print all stored information."""
        for k, v in self.memory.items():
            print(f"{k}: {v}")


def segment_response(text: str, sep: str = '\n\n') -> List[str]:
    """
    Segment a response into logical parts, preserving LaTeX blocks.
    
    This function splits text by the specified separator while ensuring
    that LaTeX math blocks (\\[ ... \\]) are kept intact.
    
    Args:
        text: The text to segment.
        sep: The separator to split by (default: double newline).
    
    Returns:
        A list of text segments.
    """
    segments = []
    temp = ''
    i = 0
    n = len(text)
    
    while i < n:
        # Check for LaTeX block start
        if text[i:i+2] == '\\[':
            j = i + 2
            while j < n and text[j:j+2] != '\\]':
                j += 1
            temp += text[i:j+2]
            i = j + 2
        # Check for separator
        elif text[i:i+len(sep)] == sep:
            if temp.strip():
                segments.append(temp)
            temp = ''
            # Skip consecutive whitespace
            j = i + len(sep)
            while j < n and text[j].strip() == '':
                j += 1
            i = j
        else:
            temp += text[i]
            i += 1
    
    if temp.strip():
        segments.append(temp)
    
    # Merge short segments with adjacent ones
    final = []
    buffer = []
    for seg in segments:
        if not seg.strip():
            buffer.append(seg)
        elif len(seg) < 100:
            buffer.append(seg)
        else:
            if buffer:
                prefix = "".join(buffer)
                buffer = []
                seg = prefix + seg
            final.append(seg)
    
    if buffer:
        prefix = "".join(buffer)
        if final:
            final[-1] += prefix
        else:
            final = [prefix]
    
    return final


def breakdown_steps(answer: str) -> Tuple[str, List[str], List[str]]:
    """
    Break down an answer into numbered steps.
    
    Args:
        answer: The answer text to break down.
    
    Returns:
        A tuple of (formatted_text, step_list, raw_segments).
    """
    steps = segment_response(answer)
    if len(steps) <= 1:
        steps = segment_response(answer, '\n')
    
    final = ""
    all_steps = []
    for idx, step in enumerate(steps):
        step_text = f"<Step{idx+1}> {step}\n"
        all_steps.append(step_text)
        final += step_text
    
    return final, all_steps, steps


def make_prompt(tokenizer, messages: List[Dict]) -> str:
    """
    Create a prompt string from messages using the tokenizer's chat template.
    
    Args:
        tokenizer: The tokenizer with apply_chat_template method.
        messages: List of message dictionaries.
    
    Returns:
        Formatted prompt string.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
