"""
Structured Language Modeling (SLM) Package

A research framework for implementing language models with progressive linguistic priors,
from simple baselines to complex regex and context-free grammar structures.
"""

__version__ = "0.1.0"
__author__ = "Jacob Valdez"

from . import models, data, training, evaluation, utils

__all__ = ["models", "data", "training", "evaluation", "utils"] 