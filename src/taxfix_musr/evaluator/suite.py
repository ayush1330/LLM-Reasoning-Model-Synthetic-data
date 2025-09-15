"""
Batch evaluation suite for tax law reasoning.

Provides batch evaluation, self-consistency testing, and failure taxonomy
for comprehensive reasoning quality assessment.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..models import Case
from ..renderer import CaseRenderer
from ..agentic import run_with_retry
from .scorer import score_case


@dataclass
class BatchResults:
    """Results from batch evaluation."""
    total_cases: int
    average_score: float
    std_dev: float
    passed_cases: int
    failed_cases: int
    individual_scores: List[Dict[str, Any]]
    failure_taxonomy: Dict[str, int]


@dataclass
class SelfConsistencyResults:
    """Results from self-consistency evaluation."""
    case_id: str
    num_samples: int
    agreement_percentage: float
    majority_amount: Optional[float]
    std_dev: float
    individual_amounts: List[Optional[float]]
    individual_scores: List[float]


def run_batch(
    n: int,
    case_generator: Callable[[], Case],
    renderer: CaseRenderer,
    retriever,
    use_agentic: bool = False,
    threshold: float = 0.85,
    use_cache: bool = True
) -> BatchResults:
    """
    Run batch evaluation on n cases.

    Args:
        n: Number of cases to generate and evaluate
        case_generator: Function that generates a new case
        renderer: Case renderer for LLM responses
        retriever: Law retriever for fetching snippets
        use_agentic: Whether to use agentic retry
        threshold: Score threshold for pass/fail

    Returns:
        Batch evaluation results
    """
    from ..logic_tree import LogicTree

    individual_scores = []
    all_failed_checks = []

    for i in range(n):
        # Generate case
        case = case_generator()

        # Compute ground truth
        logic_tree = LogicTree(case)
        ground_truth = logic_tree.compute()

        # Get law snippets from rule law references
        law_refs = []
        for rule in case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))  # Remove duplicates
        law_snippets = retriever.fetch_by_refs(law_refs)

        # Prepare scoring parameters
        required_steps = ["gross income", "donation cap", "retirement cap"]
        facts_for_math = {name: fact.value for name, fact in case.facts.items()}

        try:
            # Initialize retry_result to avoid None reference errors
            retry_result = None

            if use_agentic:
                # Use agentic retry
                retry_result = run_with_retry(
                    case=case,
                    renderer=renderer,
                    ground_truth=ground_truth,
                    law_snippets=law_snippets,
                    required_steps=required_steps,
                    allowed_law_refs=law_refs,
                    facts_for_math=facts_for_math,
                    threshold=threshold
                )
                llm_response = retry_result.output
            else:
                # Standard processing
                llm_response = renderer.render(case, law_snippets, use_cache=use_cache)

            # Score the response
            score_result = score_case(
                case_id=case.case_id,
                llm_response=llm_response,
                ground_truth=ground_truth,
                required_steps=required_steps,
                allowed_law_refs=law_refs,
                facts_for_math=facts_for_math
            )

            # Collect results
            case_result = {
                "case_id": case.case_id,
                "score": score_result.total_score,
                "passed": score_result.passed,
                "check_results": score_result.check_results,
                "details": score_result.details,
                "ground_truth": ground_truth,
                "llm_response": llm_response,
                "facts": {name: fact.value for name, fact in case.facts.items()},
                "outputs": ground_truth,
                "retry_used": retry_result.retries_used > 0 if retry_result else False,
                "retry_improved": retry_result.improved if retry_result else False
            }
            individual_scores.append(case_result)

            # Collect failed checks for taxonomy
            for check_name, passed in score_result.check_results.items():
                if not passed:
                    reason = score_result.details.get(check_name, {}).get("reason", f"{check_name} failed")
                    all_failed_checks.append(reason)

        except Exception as e:
            # Handle errors
            case_result = {
                "case_id": case.case_id,
                "score": 0.0,
                "passed": False,
                "check_results": {},
                "details": {"error": str(e)},
                "ground_truth": ground_truth,
                "llm_response": None,
                "facts": {name: fact.value for name, fact in case.facts.items()},
                "outputs": ground_truth,
                "retry_used": False,
                "retry_improved": False
            }
            individual_scores.append(case_result)
            all_failed_checks.append(f"Rendering error: {str(e)}")

    # Calculate statistics
    scores = [result["score"] for result in individual_scores]
    average_score = statistics.mean(scores) if scores else 0.0
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    passed_cases = sum(1 for result in individual_scores if result["passed"])
    failed_cases = n - passed_cases

    # Generate failure taxonomy
    failure_taxonomy = dict(Counter(all_failed_checks).most_common(3))

    return BatchResults(
        total_cases=n,
        average_score=average_score,
        std_dev=std_dev,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        individual_scores=individual_scores,
        failure_taxonomy=failure_taxonomy
    )


def self_consistency(
    case: Case,
    renderer: CaseRenderer,
    retriever,
    k: int = 3,
    temperature: float = 0.1,
    use_cache: bool = True
) -> SelfConsistencyResults:
    """
    Run self-consistency evaluation on a single case.

    Args:
        case: Tax case to evaluate
        renderer: Case renderer for LLM responses
        retriever: Law retriever for fetching snippets
        k: Number of samples to generate
        temperature: Temperature for LLM sampling (low for consistency)

    Returns:
        Self-consistency results
    """
    from ..logic_tree import LogicTree

    # Compute ground truth
    logic_tree = LogicTree(case)
    ground_truth = logic_tree.compute()

    # Get law snippets from rule law references
    law_refs = []
    for rule in case.rules.values():
        law_refs.extend(rule.law_refs)
    law_refs = list(set(law_refs))  # Remove duplicates
    law_snippets = retriever.fetch_by_refs(law_refs)

    # Prepare scoring parameters
    required_steps = ["gross income", "donation cap", "retirement cap"]
    facts_for_math = {name: fact.value for name, fact in case.facts.items()}

    individual_amounts = []
    individual_scores = []

    # Generate k samples
    for i in range(k):
        try:
            # Temporarily set low temperature for consistency
            original_temp = renderer.llm_client.temperature
            renderer.llm_client.temperature = temperature

            llm_response = renderer.render(case, law_snippets, use_cache=use_cache)

            # Restore original temperature
            renderer.llm_client.temperature = original_temp

            # Extract amount (handle None values)
            amount = llm_response.get("final_answer", {}).get("amount")
            if amount is None:
                amount = 0.0  # Default to 0 if no amount provided
            individual_amounts.append(amount)

            # Score the response
            score_result = score_case(
                case_id=case.case_id,
                llm_response=llm_response,
                ground_truth=ground_truth,
                required_steps=required_steps,
                allowed_law_refs=law_refs,
                facts_for_math=facts_for_math
            )
            individual_scores.append(score_result.total_score)

        except Exception as e:
            individual_amounts.append(None)
            individual_scores.append(0.0)

    # Calculate consistency metrics
    valid_amounts = [a for a in individual_amounts if a is not None]

    if valid_amounts:
        # Round amounts to nearest integer for voting
        rounded_amounts = [round(a) for a in valid_amounts]
        amount_counts = Counter(rounded_amounts)
        majority_amount = amount_counts.most_common(1)[0][0] if amount_counts else None
        agreement_percentage = (amount_counts.most_common(1)[0][1] / len(rounded_amounts)) * 100 if amount_counts else 0.0
        std_dev = statistics.stdev(valid_amounts) if len(valid_amounts) > 1 else 0.0
    else:
        majority_amount = None
        agreement_percentage = 0.0
        std_dev = 0.0

    return SelfConsistencyResults(
        case_id=case.case_id,
        num_samples=k,
        agreement_percentage=agreement_percentage,
        majority_amount=majority_amount,
        std_dev=std_dev,
        individual_amounts=individual_amounts,
        individual_scores=individual_scores
    )


def failure_taxonomy(reports: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Analyze failure patterns across batch reports.

    Args:
        reports: List of case reports with check results

    Returns:
        Top failure reasons with counts
    """
    all_failed_checks = []

    for report in reports:
        check_results = report.get("check_results", {})
        details = report.get("details", {})

        for check_name, passed in check_results.items():
            if not passed:
                reason = details.get(check_name, {}).get("reason", f"{check_name} failed")
                all_failed_checks.append(reason)

    return dict(Counter(all_failed_checks).most_common(3))


class EvaluationSuite:
    """Comprehensive evaluation suite for tax law reasoning."""

    def __init__(self, renderer: CaseRenderer):
        """
        Initialize evaluation suite.

        Args:
            renderer: Case renderer for generating responses
        """
        self.renderer = renderer

    def evaluate_batch(
        self,
        cases: List[Case],
        ground_truths: List[Dict[str, Any]],
        law_snippets_list: List[Dict[str, str]]
    ) -> BatchResults:
        """
        Evaluate a batch of cases.

        Args:
            cases: List of tax cases
            ground_truths: List of ground truth values
            law_snippets_list: List of law snippets for each case

        Returns:
            Batch evaluation results
        """
        # Placeholder for future batch processing implementation
        pass

    def evaluate_self_consistency(
        self,
        case: Case,
        ground_truth: Dict[str, Any],
        law_snippets: Dict[str, str],
        num_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate self-consistency across multiple samples.

        Args:
            case: Tax case to evaluate
            ground_truth: Ground truth values
            law_snippets: Law snippets
            num_samples: Number of samples to generate

        Returns:
            Self-consistency analysis
        """
        # Placeholder for future self-consistency evaluation implementation
        pass

    def evaluate_perturbation_robustness(
        self,
        case: Case,
        ground_truth: Dict[str, Any],
        law_snippets: Dict[str, str],
        perturbations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate robustness to prompt perturbations.

        Args:
            case: Tax case to evaluate
            ground_truth: Ground truth values
            law_snippets: Law snippets
            perturbations: List of perturbation configurations

        Returns:
            Perturbation robustness analysis
        """
        # Placeholder for future perturbation robustness evaluation implementation
        pass

    def _generate_perturbations(
        self,
        base_prompt: str,
        perturbation_config: Dict[str, Any]
    ) -> List[str]:
        """
        Generate perturbed versions of prompts.

        Args:
            base_prompt: Original prompt
            perturbation_config: Perturbation configuration

        Returns:
            List of perturbed prompts
        """
        # Placeholder for future prompt perturbation implementation
        pass
