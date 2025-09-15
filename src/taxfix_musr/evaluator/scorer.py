"""
Weighted score aggregation for tax law reasoning evaluation.

Combines atomic checks into a single score with configurable weights
and provides detailed scoring breakdown.
"""

from dataclasses import dataclass
from typing import Any, Dict, List



@dataclass
class ScoreWeights:
    """Configuration for scoring weights."""
    amount_accuracy: float = 0.4
    reasoning_quality: float = 0.3
    law_citations: float = 0.2
    evidence_format: float = 0.1


@dataclass
class CaseScore:
    """Detailed score for a single case."""
    case_id: str
    total_score: float
    passed: bool
    check_results: Dict[str, bool]
    details: Dict[str, Any]


def score_case(
    case_id: str,
    llm_response: Dict[str, Any],
    ground_truth: Dict[str, Any],
    required_steps: List[str],
    allowed_law_refs: List[str],
    facts_for_math: Dict[str, Any]
) -> CaseScore:
    """
    Score a single case response.

    Args:
        case_id: Case identifier
        llm_response: LLM response dictionary
        ground_truth: Ground truth values
        required_steps: List of required reasoning steps
        allowed_law_refs: List of allowed law references
        facts_for_math: Raw facts for mathematical validation

    Returns:
        Case score with detailed results
    """
    check_results = {}
    details = {}

    # Check 1: Amount accuracy
    amount_check = _check_amount_accuracy(llm_response, ground_truth)
    check_results["amount_accuracy"] = amount_check["passed"]
    details["amount_accuracy"] = amount_check

    # Check 2: Reasoning quality (required steps)
    reasoning_check = _check_reasoning_quality(llm_response, required_steps)
    check_results["reasoning_quality"] = reasoning_check["passed"]
    details["reasoning_quality"] = reasoning_check

    # Check 3: Law citations
    citations_check = _check_law_citations(llm_response, allowed_law_refs)
    check_results["law_citations"] = citations_check["passed"]
    details["law_citations"] = citations_check

    # Check 4: Evidence format
    evidence_check = _check_evidence_format(llm_response)
    check_results["evidence_format"] = evidence_check["passed"]
    details["evidence_format"] = evidence_check

    # Calculate total score
    weights = ScoreWeights()
    total_score = (
        check_results["amount_accuracy"] * weights.amount_accuracy +
        check_results["reasoning_quality"] * weights.reasoning_quality +
        check_results["law_citations"] * weights.law_citations +
        check_results["evidence_format"] * weights.evidence_format
    )

    return CaseScore(
        case_id=case_id,
        total_score=total_score,
        passed=total_score > 0.5,
        check_results=check_results,
        details=details
    )


def _check_amount_accuracy(llm_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the final amount matches ground truth."""
    final_answer = llm_response.get("final_answer", {})
    llm_amount = final_answer.get("amount")

    if llm_amount is None:
        return {"passed": False, "reason": "No amount in final answer"}

    # Find expected amount in ground truth (prefer taxable_income)
    expected_amount = None
    if "taxable_income" in ground_truth:
        expected_amount = ground_truth["taxable_income"]
    else:
        # Fallback to first numeric value
        for key, value in ground_truth.items():
            if isinstance(value, (int, float)):
                expected_amount = value
                break

    if expected_amount is None:
        return {"passed": False, "reason": "No expected amount in ground truth"}

    # Check within tolerance
    tolerance = 1e-2
    if abs(llm_amount - expected_amount) <= tolerance:
        return {
            "passed": True,
            "llm_amount": llm_amount,
            "expected_amount": expected_amount,
            "difference": abs(llm_amount - expected_amount)
        }
    else:
        return {
            "passed": False,
            "llm_amount": llm_amount,
            "expected_amount": expected_amount,
            "difference": abs(llm_amount - expected_amount),
            "reason": f"Amount difference {abs(llm_amount - expected_amount)} exceeds tolerance {tolerance}"
        }


def _check_reasoning_quality(llm_response: Dict[str, Any], required_steps: List[str]) -> Dict[str, Any]:
    """Check if required reasoning steps are present."""
    reasoning_steps = llm_response.get("reasoning_steps", [])

    if not reasoning_steps:
        return {"passed": False, "reason": "No reasoning steps provided"}

    # Check for required steps
    step_descriptions = [step.get("step", "").lower() for step in reasoning_steps]
    missing_steps = []
    found_steps = []

    for required_step in required_steps:
        step_found = any(required_step.lower() in desc for desc in step_descriptions)
        if step_found:
            found_steps.append(required_step)
        else:
            missing_steps.append(required_step)

    # More flexible evaluation: require at least 2 out of 3 core steps
    core_steps = ["gross income", "donation cap", "retirement cap"]
    found_core_steps = [step for step in found_steps if step in core_steps]
    
    # Pass if we have at least 2 core steps OR if we have all required steps
    if len(found_core_steps) >= 2 or len(missing_steps) == 0:
        return {
            "passed": True,
            "found_steps": step_descriptions,
            "required_steps": required_steps,
            "found_core_steps": found_core_steps,
            "missing_steps": missing_steps
        }
    else:
        return {
            "passed": False,
            "reason": f"Missing required steps: {missing_steps} (found {len(found_core_steps)}/3 core steps)",
            "found_steps": step_descriptions,
            "missing_steps": missing_steps,
            "found_core_steps": found_core_steps
        }


def _check_law_citations(llm_response: Dict[str, Any], allowed_law_refs: List[str]) -> Dict[str, Any]:
    """Check if law citations are valid and from allowed references."""
    law_citations = llm_response.get("law_citations", [])

    if not law_citations:
        return {"passed": False, "reason": "No law citations provided"}

    # Check if citations are from allowed references
    cited_refs = [citation.get("ref") for citation in law_citations]
    invalid_refs = [ref for ref in cited_refs if ref not in allowed_law_refs]

    if invalid_refs:
        return {
            "passed": False,
            "reason": f"Invalid law references: {invalid_refs}",
            "cited_refs": cited_refs,
            "allowed_refs": allowed_law_refs,
            "invalid_refs": invalid_refs
        }
    else:
        return {
            "passed": True,
            "cited_refs": cited_refs,
            "allowed_refs": allowed_law_refs
        }


def _check_evidence_format(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """Check if evidence sources follow the required format."""
    reasoning_steps = llm_response.get("reasoning_steps", [])

    if not reasoning_steps:
        return {"passed": False, "reason": "No reasoning steps to check"}

    invalid_evidence = []
    for i, step in enumerate(reasoning_steps):
        evidence_list = step.get("evidence", [])
        for j, evidence in enumerate(evidence_list):
            source = evidence.get("source", "")
            if not (source.startswith("fact:") or source.startswith("law:")):
                invalid_evidence.append(f"Step {i}, Evidence {j}: '{source}'")

    if invalid_evidence:
        return {
            "passed": False,
            "reason": f"Invalid evidence sources: {invalid_evidence}",
            "invalid_evidence": invalid_evidence
        }
    else:
        return {
            "passed": True,
            "total_evidence_items": sum(len(step.get("evidence", [])) for step in reasoning_steps)
        }
