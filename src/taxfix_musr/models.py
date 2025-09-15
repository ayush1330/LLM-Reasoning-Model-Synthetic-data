"""
Data models for tax law reasoning cases.

Defines the core data structures: Fact, Rule, Node, and Case dataclasses
that represent the structured reasoning artifacts for tax law evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class NodeType(Enum):
    """Types of nodes in the logic tree."""
    FACT = "fact"
    DERIVED = "derived"
    DECISION = "decision"
    OUTPUT = "output"


class RuleKind(Enum):
    """Types of rules that can be applied."""
    CAP = "cap"
    THRESHOLD = "threshold"
    PHASEOUT = "phaseout"
    FORMULA = "formula"


@dataclass
class Fact:
    """Atomic input fact for a tax case."""
    name: str
    value: Union[int, float, str, bool]
    description: Optional[str] = None



@dataclass
class Rule:
    """Tax rule with formula and law references."""
    rule_id: str
    kind: RuleKind
    formula: str
    law_refs: List[str]
    description: Optional[str] = None



@dataclass
class Node:
    """Node in the logic tree representing a computation step."""
    node_id: str
    node_type: NodeType
    depends_on: List[str]
    rule_id: Optional[str] = None
    formula: Optional[str] = None
    value: Optional[Union[int, float, str]] = None



@dataclass
class Case:
    """Complete tax law reasoning case."""
    case_id: str
    facts: Dict[str, Fact]
    rules: Dict[str, Rule]
    nodes: Dict[str, Node]
    expected: Optional[Dict[str, Any]] = None
    target_question: str = ""

