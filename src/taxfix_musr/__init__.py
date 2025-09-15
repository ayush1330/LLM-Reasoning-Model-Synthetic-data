"""
Taxfix MuSR - MuSR-style Synthetic Tax Law Reasoning Module

A Python module that generates synthetic tax law cases on demand and evaluates
an LLM's reasoning on them. Inspired by MuSR methodology: create structured
reasoning artifacts (logic trees), render natural-language narratives, then
evaluate reasoning quality.
"""

__version__ = "0.1.0"
__author__ = "Taxfix Case Study"

# Main exports - avoiding circular imports
from .models import Case, Fact, Rule, Node, NodeType, RuleKind
from .logic_tree import LogicTree
from .schema_generator import random_case, manual_case
from .llm_client import LLMClient, MockLLMClient
from .retriever import LawRetriever
from .agentic import run_with_retry, RetryResult
from .cache import LLMCache, get_cache, make_prompt_hash
from .manifest import ManifestManager
from .reports import ReportGenerator, BatchSummary, CaseRecord
from .narrative_generator import AdvancedNarrativeGenerator, ComplexityLevel, ScenarioType

__all__ = [
    # Core models
    "Case", "Fact", "Rule", "Node", "NodeType", "RuleKind",
    # Core functionality
    "LogicTree", "random_case", "manual_case",
    # LLM integration
    "LLMClient", "MockLLMClient",
    # Supporting modules
    "LawRetriever", "run_with_retry", "RetryResult",
    "LLMCache", "get_cache", "make_prompt_hash",
    "ManifestManager", "ReportGenerator", "BatchSummary", "CaseRecord",
    "AdvancedNarrativeGenerator", "ComplexityLevel", "ScenarioType"
]

