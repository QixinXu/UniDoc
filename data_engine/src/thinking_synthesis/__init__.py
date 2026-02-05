"""
Thinking Pattern Synthesis Module

This module provides functionality to synthesize thinking patterns
(chain-of-thought reasoning) from question-answer pairs.
"""

from .processor import RefinementProcessor, PosteriorManager
from .synthesizer import ThinkingSynthesizer

__all__ = [
    "RefinementProcessor",
    "PosteriorManager",
    "ThinkingSynthesizer",
]
