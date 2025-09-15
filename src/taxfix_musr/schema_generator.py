"""
Schema generator for creating tax law cases.

Generates both random and manual tax law cases with configurable parameters
and rule sets for testing and evaluation purposes.
"""

import random
from typing import Any, Dict, Optional

from .models import Case, Fact, Node, NodeType, Rule, RuleKind


def random_case(
    case_id: Optional[str] = None,
    seed: Optional[int] = None,
    **overrides: Any
) -> Case:
    """
    Generate a random tax law case.

    Args:
        case_id: Optional case identifier
        seed: Random seed for reproducibility
        **overrides: Override default parameters

    Returns:
        Generated tax case
    """
    if seed is not None:
        random.seed(seed)

    # Generate random facts
    facts = {
        "salary": random.randint(30000, 100000),
        "freelance": random.randint(0, 20000),
        "donation": random.randint(1000, 15000),
        "retirement_contribution": random.randint(1000, 8000),
    }

    # Apply overrides
    facts.update(overrides.get("facts", {}))

    # Extract target_question from overrides if present
    target_question = overrides.pop("target_question", "What is the taxable income?")

    return manual_case(
        case_id=case_id or f"random_case_{random.randint(1000, 9999)}",
        facts=facts,
        target_question=target_question,
        **overrides
    )


def manual_case(
    case_id: str,
    facts: Dict[str, Any],
    target_question: str = "What is the taxable income?",
    **overrides: Any
) -> Case:
    """
    Generate a manual tax law case with specified facts.

    Args:
        case_id: Case identifier
        facts: Dictionary of fact name -> value
        target_question: Question to answer
        **overrides: Additional overrides

    Returns:
        Generated tax case
    """
    # Convert facts dict to Fact objects
    fact_objects = {}
    for name, value in facts.items():
        fact_objects[name] = Fact(name=name, value=value)

    # Generate seed rules
    rules = _generate_seed_rules()

    # Generate nodes
    nodes = _generate_standard_nodes()

    return Case(
        case_id=case_id,
        facts=fact_objects,
        rules=rules,
        nodes=nodes,
        target_question=target_question
    )


def _generate_seed_rules() -> Dict[str, Rule]:
    """
    Generate the standard seed rules for MVP.

    Returns:
        Dictionary of rule_id -> Rule
    """
    return {
        "donation_cap": Rule(
            rule_id="donation_cap",
            kind=RuleKind.CAP,
            formula="min(donation, gross_income * 0.10)",
            law_refs=["§DON-10pct"],
            description="Donation cap: 10% of gross income"
        ),
        "retirement_cap": Rule(
            rule_id="retirement_cap",
            kind=RuleKind.CAP,
            formula="min(retirement_contribution, 6000)",
            law_refs=["§RET-6000"],
            description="Retirement cap: €6,000 per tax year"
        ),
        "child_credit_phaseout": Rule(
            rule_id="child_credit_phaseout",
            kind=RuleKind.PHASEOUT,
            formula="2000 if AGI <= 90000 else max(0, 2000 - 0.05 * (AGI - 90000))",
            law_refs=["§CHILD-CR-Phaseout"],
            description="Child credit phase-out: €2,000 base, phases out at 5% for AGI > €90,000"
        ),
        "standard_deduction": Rule(
            rule_id="standard_deduction",
            kind=RuleKind.THRESHOLD,
            formula="10000",
            law_refs=["§STD-DED"],
            description="Standard deduction: €10,000 per tax year"
        )
    }


def _generate_standard_nodes() -> Dict[str, Node]:
    """
    Generate standard nodes for tax cases.

    Returns:
        Dictionary of node_id -> Node
    """
    return {
        # Fact nodes
        "salary": Node(
            node_id="salary",
            node_type=NodeType.FACT,
            depends_on=[]
        ),
        "freelance": Node(
            node_id="freelance",
            node_type=NodeType.FACT,
            depends_on=[]
        ),
        "donation": Node(
            node_id="donation",
            node_type=NodeType.FACT,
            depends_on=[]
        ),
        "retirement_contribution": Node(
            node_id="retirement_contribution",
            node_type=NodeType.FACT,
            depends_on=[]
        ),

        # Derived nodes
        "gross_income": Node(
            node_id="gross_income",
            node_type=NodeType.DERIVED,
            depends_on=["salary", "freelance"],
            formula="salary + freelance"
        ),
        "allowable_donation": Node(
            node_id="allowable_donation",
            node_type=NodeType.DERIVED,
            depends_on=["donation", "gross_income"],
            rule_id="donation_cap"
        ),
        "allowable_retirement": Node(
            node_id="allowable_retirement",
            node_type=NodeType.DERIVED,
            depends_on=["retirement_contribution"],
            rule_id="retirement_cap"
        ),
        "AGI": Node(
            node_id="AGI",
            node_type=NodeType.DERIVED,
            depends_on=["gross_income", "allowable_donation", "allowable_retirement"],
            formula="gross_income - allowable_donation - allowable_retirement"
        ),
        "child_credit": Node(
            node_id="child_credit",
            node_type=NodeType.DERIVED,
            depends_on=["AGI"],
            rule_id="child_credit_phaseout"
        ),
        "itemized_deduction": Node(
            node_id="itemized_deduction",
            node_type=NodeType.DERIVED,
            depends_on=["allowable_donation", "allowable_retirement"],
            formula="allowable_donation + allowable_retirement"
        ),
        "standard_deduction_amount": Node(
            node_id="standard_deduction_amount",
            node_type=NodeType.DERIVED,
            depends_on=[],
            rule_id="standard_deduction"
        ),
        "deduction_choice": Node(
            node_id="deduction_choice",
            node_type=NodeType.DERIVED,
            depends_on=["standard_deduction_amount", "itemized_deduction"],
            formula="max(standard_deduction_amount, itemized_deduction)"
        ),

        # Output nodes
        "taxable_income": Node(
            node_id="taxable_income",
            node_type=NodeType.OUTPUT,
            depends_on=["gross_income", "deduction_choice"],
            formula="gross_income - deduction_choice"
        )
    }
