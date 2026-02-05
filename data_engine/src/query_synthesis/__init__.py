"""
Query Synthesis Module

This module provides functionality to synthesize question-answer pairs
from document data (PDFs with figures, tables, and text).
"""

from .anchors import ImageAnchor, TableAnchor, TextAnchor
from .document_parser import DocumentParser
from .synthesizer import QuerySynthesizer

__all__ = [
    "ImageAnchor",
    "TableAnchor", 
    "TextAnchor",
    "DocumentParser",
    "QuerySynthesizer",
]
