"""
Agentic self-check retry system for tax law reasoning.

Provides automatic retry with critique-based improvement when LLM responses
fail to meet quality thresholds or accuracy requirements.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .models import Case
from .renderer import CaseRenderer
from .evaluator.scorer import score_case


@dataclass
class RetryResult:
    """Result of agentic retry process."""
    output: Dict[str, Any]
    score: float
    retries_used: int
    improved: bool
    critique: str = ""


def run_with_retry(
    case: Case,
    renderer: CaseRenderer,
    ground_truth: Dict[str, Any],
    law_snippets: Dict[str, str],
    required_steps: List[str],
    allowed_law_refs: List[str],
    facts_for_math: Dict[str, Any],
    threshold: float = 0.85
) -> RetryResult:
    """
    Run case with agentic retry if score is below threshold.

    Args:
        case: Tax case to process
        renderer: Case renderer for LLM interaction
        ground_truth: Ground truth values for comparison
        law_snippets: Law snippets for grounding
        required_steps: Required reasoning steps
        allowed_law_refs: Allowed law references
        facts_for_math: Facts for mathematical validation
        threshold: Minimum score threshold (0.0 to 1.0)

    Returns:
        RetryResult with output, score, retries used, and improvement status
    """
    # Initial attempt
    print(f"  Initial attempt...")
    initial_output = renderer.render(case, law_snippets)
    initial_score_result = score_case(
        case_id=case.case_id,
        llm_response=initial_output,
        ground_truth=ground_truth,
        required_steps=required_steps,
        allowed_law_refs=allowed_law_refs,
        facts_for_math=facts_for_math
    )

    initial_score = initial_score_result.total_score
    print(f"  Initial score: {initial_score:.2f}")

    # Check if retry is needed
    needs_retry = (
        initial_score < threshold or 
        not initial_score_result.check_results.get("amount_accuracy", False)
    )

    if not needs_retry:
        return RetryResult(
            output=initial_output,
            score=initial_score,
            retries_used=0,
            improved=False
        )

    # Generate critique from failed checks
    critique = _generate_critique(initial_score_result, ground_truth)
    print(f"  Critique: {critique}")

    # Retry with critique
    print(f"  Retrying with critique...")
    retry_output = _render_with_critique(
        case, renderer, law_snippets, critique, ground_truth
    )

    retry_score_result = score_case(
        case_id=case.case_id,
        llm_response=retry_output,
        ground_truth=ground_truth,
        required_steps=required_steps,
        allowed_law_refs=allowed_law_refs,
        facts_for_math=facts_for_math
    )

    retry_score = retry_score_result.total_score
    print(f"  Retry score: {retry_score:.2f}")

    # Determine if retry improved the result
    improved = retry_score > initial_score

    # Return the better result
    if retry_score >= initial_score:
        return RetryResult(
            output=retry_output,
            score=retry_score,
            retries_used=1,
            improved=improved,
            critique=critique
        )
    else:
        return RetryResult(
            output=initial_output,
            score=initial_score,
            retries_used=1,
            improved=False,
            critique=critique
        )


def _generate_critique(score_result, ground_truth: Dict[str, Any]) -> str:
    """Generate critique from failed evaluation checks."""
    critiques = []

    # Check amount accuracy
    if not score_result.check_results.get("amount_accuracy", False):
        amount_details = score_result.details.get("amount_accuracy", {})
        if "reason" in amount_details:
            critiques.append(f"Amount calculation error: {amount_details['reason']}")
        else:
            expected = ground_truth.get("taxable_income", "unknown")
            critiques.append(f"Final amount should be {expected}")

    # Check reasoning quality
    if not score_result.check_results.get("reasoning_quality", False):
        reasoning_details = score_result.details.get("reasoning_quality", {})
        if "missing_steps" in reasoning_details:
            missing = reasoning_details["missing_steps"]
            critiques.append(f"Missing required reasoning steps: {', '.join(missing)}")
        else:
            critiques.append("Incomplete reasoning steps")

    # Check law citations
    if not score_result.check_results.get("law_citations", False):
        citation_details = score_result.details.get("law_citations", {})
        if "reason" in citation_details:
            critiques.append(f"Law citation error: {citation_details['reason']}")
        else:
            critiques.append("Missing or invalid law citations")

    # Check evidence format
    if not score_result.check_results.get("evidence_format", False):
        evidence_details = score_result.details.get("evidence_format", {})
        if "reason" in evidence_details:
            critiques.append(f"Evidence format error: {evidence_details['reason']}")
        else:
            critiques.append("Evidence sources must start with 'fact:' or 'law:'")

    if not critiques:
        critiques.append("General quality issues detected")

    return " | ".join(critiques)


def _render_with_critique(
    case: Case,
    renderer: CaseRenderer,
    law_snippets: Dict[str, str],
    critique: str,
    ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """Render case with critique-based retry prompt."""
    # Build enhanced system prompt with critique
    enhanced_system_prompt = renderer._build_system_prompt() + f"""

**CRITIQUE FROM PREVIOUS ATTEMPT:**
{critique}

**INSTRUCTIONS FOR RETRY:**
- Address the specific issues mentioned in the critique above
- Ensure your final amount calculation is correct
- Include all required reasoning steps
- Cite appropriate law references
- Use proper evidence format (fact: or law: prefixes)
- Double-check your mathematical calculations

**EXPECTED RESULT:** The final amount should be {ground_truth.get('taxable_income', 'calculated correctly')}."""

    # Build user prompt
    user_prompt = renderer._build_user_prompt(case, law_snippets)

    # Prepare messages with enhanced system prompt
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call LLM with critique
    response_text = renderer.llm_client.chat(messages)
    return renderer._parse_response(response_text, case.case_id)
