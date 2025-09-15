"""
Evaluation module for tax law reasoning.

Provides comprehensive evaluation of LLM reasoning quality including
scoring and batch evaluation capabilities.
"""

# Main exports - avoiding circular imports
from .schema import LLMCaseOutput, ReasoningStep, LawCitation, Evidence, FinalAnswer
from .scorer import score_case, CaseScore, ScoreWeights

__all__ = [
    "LLMCaseOutput",
    "ReasoningStep", 
    "LawCitation",
    "Evidence",
    "FinalAnswer",
    "score_case",
    "CaseScore",
    "ScoreWeights"
]
