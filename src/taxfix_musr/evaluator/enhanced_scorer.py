"""
Enhanced scoring system with granular evaluation metrics.

Provides detailed breakdown of reasoning quality across multiple dimensions
for comprehensive assessment of LLM performance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re


@dataclass
class EnhancedScoreWeights:
    """Enhanced configuration for scoring weights."""
    # Mathematical accuracy
    final_amount_accuracy: float = 0.20
    intermediate_calculations: float = 0.15
    
    # Legal reasoning
    law_citation_completeness: float = 0.15
    law_application_correctness: float = 0.10
    
    # Reasoning process
    step_completeness: float = 0.10
    logical_flow: float = 0.10
    
    # Edge case handling
    cap_application: float = 0.08
    phase_out_handling: float = 0.07
    
    # Response quality
    explanation_clarity: float = 0.05


@dataclass
class DetailedCaseScore:
    """Detailed score breakdown for a single case."""
    case_id: str
    total_score: float
    passed: bool
    
    # Detailed scores
    final_amount_accuracy: float
    intermediate_calculations: float
    law_citation_completeness: float
    law_application_correctness: float
    step_completeness: float
    logical_flow: float
    cap_application: float
    phase_out_handling: float
    explanation_clarity: float
    
    # Detailed results
    check_results: Dict[str, bool]
    details: Dict[str, Any]
    recommendations: List[str]


def enhanced_score_case(
    case_id: str,
    llm_response: Dict[str, Any],
    ground_truth: Dict[str, Any],
    required_steps: List[str],
    allowed_law_refs: List[str],
    facts_for_math: Dict[str, Any]
) -> DetailedCaseScore:
    """
    Score a case with enhanced granular metrics.
    
    Args:
        case_id: Case identifier
        llm_response: LLM response dictionary
        ground_truth: Ground truth values
        required_steps: List of required reasoning steps
        allowed_law_refs: List of allowed law references
        facts_for_math: Raw facts for mathematical validation
    
    Returns:
        Detailed case score with granular breakdown
    """
    recommendations = []
    
    # 1. Final Amount Accuracy
    final_amount_score = _check_final_amount_accuracy(llm_response, ground_truth)
    if final_amount_score < 1.0:
        recommendations.append("Improve final amount calculation precision")
    
    # 2. Intermediate Calculations
    intermediate_score = _check_intermediate_calculations(llm_response, ground_truth, facts_for_math)
    if intermediate_score < 1.0:
        recommendations.append("Show more detailed intermediate calculation steps")
    
    # 3. Law Citation Completeness
    citation_completeness = _check_law_citation_completeness(llm_response, allowed_law_refs)
    if citation_completeness < 1.0:
        recommendations.append("Include all provided law references in citations")
    
    # 4. Law Application Correctness
    law_application = _check_law_application_correctness(llm_response, ground_truth)
    if law_application < 1.0:
        recommendations.append("Apply tax laws more accurately in reasoning")
    
    # 5. Step Completeness
    step_completeness = _check_step_completeness(llm_response, required_steps)
    if step_completeness < 1.0:
        recommendations.append("Include all required reasoning steps")
    
    # 6. Logical Flow
    logical_flow = _check_logical_flow(llm_response)
    if logical_flow < 1.0:
        recommendations.append("Improve logical flow between reasoning steps")
    
    # 7. Cap Application
    cap_application = _check_cap_application(llm_response, ground_truth, facts_for_math)
    if cap_application < 1.0:
        recommendations.append("Correctly apply donation and retirement caps")
    
    # 8. Phase-out Handling
    phase_out_handling = _check_phase_out_handling(llm_response, ground_truth, facts_for_math)
    if phase_out_handling < 1.0:
        recommendations.append("Handle child credit phase-out calculations correctly")
    
    # 9. Explanation Clarity
    explanation_clarity = _check_explanation_clarity(llm_response)
    if explanation_clarity < 1.0:
        recommendations.append("Provide clearer explanations in final answer")
    
    # Calculate total score
    weights = EnhancedScoreWeights()
    total_score = (
        final_amount_score * weights.final_amount_accuracy +
        intermediate_score * weights.intermediate_calculations +
        citation_completeness * weights.law_citation_completeness +
        law_application * weights.law_application_correctness +
        step_completeness * weights.step_completeness +
        logical_flow * weights.logical_flow +
        cap_application * weights.cap_application +
        phase_out_handling * weights.phase_out_handling +
        explanation_clarity * weights.explanation_clarity
    )
    
    # Create check results for compatibility
    check_results = {
        "final_amount_accuracy": final_amount_score >= 0.95,
        "intermediate_calculations": intermediate_score >= 0.8,
        "law_citation_completeness": citation_completeness >= 0.9,
        "law_application_correctness": law_application >= 0.8,
        "step_completeness": step_completeness >= 0.8,
        "logical_flow": logical_flow >= 0.7,
        "cap_application": cap_application >= 0.9,
        "phase_out_handling": phase_out_handling >= 0.8,
        "explanation_clarity": explanation_clarity >= 0.7
    }
    
    # Detailed breakdown
    details = {
        "scores": {
            "final_amount_accuracy": final_amount_score,
            "intermediate_calculations": intermediate_score,
            "law_citation_completeness": citation_completeness,
            "law_application_correctness": law_application,
            "step_completeness": step_completeness,
            "logical_flow": logical_flow,
            "cap_application": cap_application,
            "phase_out_handling": phase_out_handling,
            "explanation_clarity": explanation_clarity
        },
        "weights": {
            "final_amount_accuracy": weights.final_amount_accuracy,
            "intermediate_calculations": weights.intermediate_calculations,
            "law_citation_completeness": weights.law_citation_completeness,
            "law_application_correctness": weights.law_application_correctness,
            "step_completeness": weights.step_completeness,
            "logical_flow": weights.logical_flow,
            "cap_application": weights.cap_application,
            "phase_out_handling": weights.phase_out_handling,
            "explanation_clarity": weights.explanation_clarity
        }
    }
    
    return DetailedCaseScore(
        case_id=case_id,
        total_score=total_score,
        passed=total_score >= 0.75,  # Higher threshold for enhanced scoring
        final_amount_accuracy=final_amount_score,
        intermediate_calculations=intermediate_score,
        law_citation_completeness=citation_completeness,
        law_application_correctness=law_application,
        step_completeness=step_completeness,
        logical_flow=logical_flow,
        cap_application=cap_application,
        phase_out_handling=phase_out_handling,
        explanation_clarity=explanation_clarity,
        check_results=check_results,
        details=details,
        recommendations=recommendations
    )


def _check_final_amount_accuracy(llm_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Check accuracy of final amount calculation."""
    final_answer = llm_response.get("final_answer", {})
    llm_amount = final_answer.get("amount")
    expected_amount = ground_truth.get("taxable_income")
    
    if llm_amount is None or expected_amount is None:
        return 0.0
    
    # Allow small tolerance for floating point differences
    tolerance = max(0.01, abs(expected_amount) * 0.001)
    if abs(llm_amount - expected_amount) <= tolerance:
        return 1.0
    
    # Partial credit based on relative error
    relative_error = abs(llm_amount - expected_amount) / max(abs(expected_amount), 1)
    if relative_error <= 0.01:  # 1% error
        return 0.9
    elif relative_error <= 0.05:  # 5% error
        return 0.7
    elif relative_error <= 0.10:  # 10% error
        return 0.5
    else:
        return 0.0


def _check_intermediate_calculations(llm_response: Dict[str, Any], ground_truth: Dict[str, Any], facts: Dict[str, Any]) -> float:
    """Check accuracy of intermediate calculations mentioned in reasoning."""
    reasoning_steps = llm_response.get("reasoning_steps", [])
    score = 1.0
    
    # Look for specific calculations in reasoning steps
    calculations_found = {
        "gross_income": False,
        "donation_cap": False,
        "retirement_cap": False,
        "deduction_choice": False
    }
    
    for step in reasoning_steps:
        step_text = step.get("claim", "").lower()
        
        # Check for gross income calculation
        if "gross income" in step_text or "total income" in step_text:
            calculations_found["gross_income"] = True
            # Look for correct gross income value
            expected_gross = ground_truth.get("gross_income", 0)
            if str(expected_gross) in step_text:
                pass  # Correct
            else:
                score -= 0.1
        
        # Check for donation cap calculation
        if "donation" in step_text and ("cap" in step_text or "10%" in step_text or "limit" in step_text):
            calculations_found["donation_cap"] = True
            expected_donation = ground_truth.get("allowable_donation", 0)
            if str(expected_donation) in step_text:
                pass  # Correct
            else:
                score -= 0.1
        
        # Check for retirement cap calculation
        if "retirement" in step_text and ("cap" in step_text or "6000" in step_text or "limit" in step_text):
            calculations_found["retirement_cap"] = True
        
        # Check for deduction choice reasoning
        if "standard" in step_text and "itemized" in step_text:
            calculations_found["deduction_choice"] = True
    
    # Penalize for missing key calculations
    missing_calculations = sum(1 for found in calculations_found.values() if not found)
    score -= missing_calculations * 0.15
    
    return max(0.0, score)


def _check_law_citation_completeness(llm_response: Dict[str, Any], allowed_law_refs: List[str]) -> float:
    """Check completeness of law citations."""
    law_citations = llm_response.get("law_citations", [])
    
    if not law_citations:
        return 0.0
    
    cited_refs = [citation.get("ref") for citation in law_citations if citation.get("ref")]
    
    # Calculate coverage
    covered_refs = sum(1 for ref in allowed_law_refs if ref in cited_refs)
    total_refs = len(allowed_law_refs)
    
    if total_refs == 0:
        return 1.0
    
    coverage_ratio = covered_refs / total_refs
    
    # Bonus for citing all references
    if coverage_ratio == 1.0:
        return 1.0
    elif coverage_ratio >= 0.8:
        return 0.9
    elif coverage_ratio >= 0.6:
        return 0.7
    elif coverage_ratio >= 0.4:
        return 0.5
    else:
        return 0.2


def _check_law_application_correctness(llm_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Check if laws are applied correctly."""
    reasoning_steps = llm_response.get("reasoning_steps", [])
    score = 1.0
    
    # Check for correct application of key laws
    for step in reasoning_steps:
        evidence_items = step.get("evidence", [])
        
        for evidence in evidence_items:
            source = evidence.get("source", "")
            content = evidence.get("content", "")
            
            # Check donation cap law application
            if "§DON-10pct" in source:
                if "10%" not in content and "10 percent" not in content.lower():
                    score -= 0.1
            
            # Check retirement cap law application
            if "§RET-6000" in source:
                if "6000" not in content and "6,000" not in content:
                    score -= 0.1
            
            # Check child credit law application
            if "§CHILD-CR-Phaseout" in source:
                if "phase" not in content.lower() and "reduce" not in content.lower():
                    score -= 0.1
    
    return max(0.0, score)


def _check_step_completeness(llm_response: Dict[str, Any], required_steps: List[str]) -> float:
    """Check if all required reasoning steps are present."""
    reasoning_steps = llm_response.get("reasoning_steps", [])
    
    if not reasoning_steps:
        return 0.0
    
    step_texts = [step.get("step", "").lower() for step in reasoning_steps]
    all_text = " ".join(step_texts)
    
    covered_steps = 0
    for required_step in required_steps:
        step_keywords = required_step.lower().split()
        if all(keyword in all_text for keyword in step_keywords):
            covered_steps += 1
    
    return covered_steps / len(required_steps) if required_steps else 1.0


def _check_logical_flow(llm_response: Dict[str, Any]) -> float:
    """Check logical flow and coherence of reasoning steps."""
    reasoning_steps = llm_response.get("reasoning_steps", [])
    
    if len(reasoning_steps) < 2:
        return 0.5  # Not enough steps to evaluate flow
    
    score = 1.0
    
    # Check for logical order
    step_order_score = _evaluate_step_order(reasoning_steps)
    score *= step_order_score
    
    # Check for evidence supporting claims
    evidence_score = _evaluate_evidence_support(reasoning_steps)
    score *= evidence_score
    
    return score


def _evaluate_step_order(reasoning_steps: List[Dict[str, Any]]) -> float:
    """Evaluate if steps are in logical order."""
    expected_order = [
        "gross income", "income", "total",
        "donation", "charitable",
        "retirement",
        "deduction", "standard", "itemized",
        "taxable",
        "child", "credit"
    ]
    
    step_texts = [step.get("step", "").lower() for step in reasoning_steps]
    
    # Simple heuristic: earlier steps should contain earlier keywords
    order_violations = 0
    for i, step_text in enumerate(step_texts):
        step_priority = len(expected_order)  # Default low priority
        
        for j, keyword in enumerate(expected_order):
            if keyword in step_text:
                step_priority = j
                break
        
        # Check if this step comes after a later-priority step
        for prev_step in step_texts[:i]:
            for k, keyword in enumerate(expected_order):
                if keyword in prev_step and k > step_priority:
                    order_violations += 1
                    break
    
    # Convert violations to score
    max_violations = len(reasoning_steps) * 2
    violation_ratio = min(order_violations / max_violations, 1.0) if max_violations > 0 else 0
    return 1.0 - violation_ratio


def _evaluate_evidence_support(reasoning_steps: List[Dict[str, Any]]) -> float:
    """Evaluate if evidence properly supports claims."""
    total_steps = len(reasoning_steps)
    well_supported_steps = 0
    
    for step in reasoning_steps:
        evidence_items = step.get("evidence", [])
        claim = step.get("claim", "")
        
        if not evidence_items:
            continue  # No evidence, can't evaluate
        
        # Check if evidence is relevant to claim
        relevant_evidence = 0
        for evidence in evidence_items:
            content = evidence.get("content", "").lower()
            source = evidence.get("source", "").lower()
            
            # Simple relevance check
            claim_words = set(claim.lower().split())
            evidence_words = set(content.split()) | set(source.split())
            
            overlap = len(claim_words & evidence_words)
            if overlap >= 2:  # At least 2 words in common
                relevant_evidence += 1
        
        if relevant_evidence > 0:
            well_supported_steps += 1
    
    return well_supported_steps / total_steps if total_steps > 0 else 1.0


def _check_cap_application(llm_response: Dict[str, Any], ground_truth: Dict[str, Any], facts: Dict[str, Any]) -> float:
    """Check if donation and retirement caps are correctly applied."""
    reasoning_steps = llm_response.get("reasoning_steps", [])
    score = 1.0
    
    # Check donation cap application
    donation_amount = facts.get("donation", 0)
    gross_income = facts.get("salary", 0) + facts.get("freelance", 0)
    expected_donation_cap = min(donation_amount, gross_income * 0.1)
    
    donation_mentioned = False
    for step in reasoning_steps:
        step_text = step.get("claim", "").lower()
        if "donation" in step_text and ("cap" in step_text or "10%" in step_text):
            donation_mentioned = True
            # Check if correct amount is mentioned
            if str(int(expected_donation_cap)) in step_text or str(expected_donation_cap) in step_text:
                pass  # Correct
            else:
                score -= 0.2
            break
    
    if donation_amount > 0 and not donation_mentioned:
        score -= 0.3
    
    # Check retirement cap application
    retirement_amount = facts.get("retirement_contribution", 0)
    expected_retirement_cap = min(retirement_amount, 6000)
    
    retirement_mentioned = False
    for step in reasoning_steps:
        step_text = step.get("claim", "").lower()
        if "retirement" in step_text and ("cap" in step_text or "6000" in step_text):
            retirement_mentioned = True
            break
    
    if retirement_amount > 6000 and not retirement_mentioned:
        score -= 0.3
    
    return max(0.0, score)


def _check_phase_out_handling(llm_response: Dict[str, Any], ground_truth: Dict[str, Any], facts: Dict[str, Any]) -> float:
    """Check if child credit phase-out is handled correctly."""
    children = facts.get("children", 0)
    
    if children == 0:
        return 1.0  # No phase-out to handle
    
    reasoning_steps = llm_response.get("reasoning_steps", [])
    agi = ground_truth.get("AGI", 0)
    
    # If AGI is above phase-out threshold
    if agi > 90000:
        phase_out_mentioned = False
        for step in reasoning_steps:
            step_text = step.get("claim", "").lower()
            if "child" in step_text and ("phase" in step_text or "reduce" in step_text or "limit" in step_text):
                phase_out_mentioned = True
                break
        
        if not phase_out_mentioned:
            return 0.3  # Should have mentioned phase-out
    
    return 1.0


def _check_explanation_clarity(llm_response: Dict[str, Any]) -> float:
    """Check clarity and completeness of final explanation."""
    final_answer = llm_response.get("final_answer", {})
    explanation = final_answer.get("explanation", "")
    
    if not explanation:
        return 0.0
    
    score = 1.0
    
    # Check for key elements in explanation
    explanation_lower = explanation.lower()
    
    # Should mention calculation method
    if not any(word in explanation_lower for word in ["calculated", "computed", "determined"]):
        score -= 0.2
    
    # Should mention deduction choice if applicable
    if not any(word in explanation_lower for word in ["deduction", "standard", "itemized"]):
        score -= 0.2
    
    # Should be reasonably detailed (more than just a number)
    if len(explanation.split()) < 10:
        score -= 0.3
    
    # Should not be excessively verbose
    if len(explanation.split()) > 100:
        score -= 0.1
    
    return max(0.0, score)
