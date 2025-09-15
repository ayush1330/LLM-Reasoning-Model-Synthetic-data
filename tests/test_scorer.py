"""
Tests for evaluation scoring functionality.

Tests the comprehensive scorer's ability to evaluate LLM responses against
ground truth with detailed metrics and scoring.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.evaluator.scorer import score_case


class TestScorer:
    """Test cases for evaluation scorer functions."""

    def test_amount_comparison_within_tolerance(self):
        """Test amount comparison within tolerance using comprehensive scorer."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case narrative",
            "law_citations": [
                {
                    "ref": "§STD-DED",
                    "snippet": "Standard deduction is €10,000 per tax year."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Calculate taxable income",
                    "claim": "Taxable income is €84,000",
                    "evidence": [
                        {
                            "source": "fact: salary",
                            "content": "80000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 84000,
                "explanation": "Taxable income calculation"
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["taxable income"],
            allowed_law_refs=["§STD-DED"],
            facts_for_math={"salary": 80000}
        )
        
        assert score_result.passed is True
        assert score_result.check_results["amount_accuracy"] is True
        assert score_result.total_score > 0.8

    def test_amount_comparison_none_values(self):
        """Test handling of None amount values."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case",
            "law_citations": [],
            "reasoning_steps": [],
            "final_answer": {
                "explanation": "No amount provided"
                # Missing amount field
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=[],
            allowed_law_refs=[],
            facts_for_math={}
        )
        
        assert score_result.check_results["amount_accuracy"] is False
        assert score_result.total_score < 0.5

    def test_verdict_comparison(self):
        """Test verdict comparison when applicable."""
        ground_truth = {"verdict": "eligible"}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case",
            "law_citations": [
                {
                    "ref": "§STD-DED",
                    "snippet": "Standard deduction applies."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Check eligibility",
                    "claim": "Taxpayer is eligible",
                    "evidence": [
                        {
                            "source": "fact: status",
                            "content": "eligible"
                        }
                    ]
                }
            ],
            "final_answer": {
                "verdict": "eligible",
                "explanation": "Matches expected verdict"
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["eligibility"],
            allowed_law_refs=["§STD-DED"],
            facts_for_math={"status": "eligible"}
        )
        
        # Should pass with proper citations and reasoning
        assert score_result.total_score > 0.5

    def test_reasoning_steps_validation(self):
        """Test reasoning steps structure validation."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case",
            "law_citations": [],
            "reasoning_steps": [
                {
                    "step": "Calculate gross income",
                    "claim": "Gross income calculation",
                    "evidence": [
                        {
                            "source": "fact: salary",
                            "content": "80000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 84000,
                "explanation": "Complete reasoning provided"
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["gross income"],
            allowed_law_refs=[],
            facts_for_math={"salary": 80000}
        )
        
        assert score_result.check_results["reasoning_quality"] is True
        assert score_result.check_results["evidence_format"] is True

    def test_citations_validation(self):
        """Test law citations validation."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case",
            "law_citations": [
                {
                    "ref": "§STD-DED",
                    "snippet": "Standard deduction is €10,000 per tax year."
                },
                {
                    "ref": "§DON-10pct", 
                    "snippet": "Charitable contributions are deductible up to 10% of gross income."
                }
            ],
            "reasoning_steps": [],
            "final_answer": {
                "amount": 84000,
                "explanation": "With proper law citations"
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=[],
            allowed_law_refs=["§STD-DED", "§DON-10pct"],
            facts_for_math={}
        )
        
        assert score_result.check_results["law_citations"] is True

    def test_complete_evaluation_pass(self):
        """Test complete evaluation with passing case using comprehensive scorer."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case_1",
            "narrative": "A taxpayer with salary and freelance income.",
            "law_citations": [
                {
                    "ref": "§DON-10pct",
                    "snippet": "Charitable contributions are deductible up to 10% of gross income."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Compute gross income",
                    "claim": "Gross income is the sum of salary and freelance income",
                    "evidence": [
                        {
                            "source": "fact: salary",
                            "content": "Salary income of €80,000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 84000,
                "explanation": "Taxable income is gross income minus allowable deductions."
            }
        }
        
        score_result = score_case(
            case_id="test_case_1",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["gross income"],
            allowed_law_refs=["§DON-10pct"],
            facts_for_math={"salary": 80000}
        )
        
        assert score_result.passed is True
        assert score_result.check_results["amount_accuracy"] is True
        assert score_result.check_results["law_citations"] is True
        assert score_result.total_score > 0.8

    def test_complete_evaluation_fail(self):
        """Test complete evaluation with failing case using comprehensive scorer."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case_1",
            "narrative": "A taxpayer with salary and freelance income.",
            "law_citations": [],  # Missing citations
            "reasoning_steps": [],  # Missing reasoning
            "final_answer": {
                "amount": 85000,  # Wrong amount
                "explanation": "Taxable income calculation"
            }
        }
        
        score_result = score_case(
            case_id="test_case_1",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["gross income"],
            allowed_law_refs=["§DON-10pct"],
            facts_for_math={"salary": 80000}
        )
        
        assert score_result.passed is False
        assert score_result.check_results["amount_accuracy"] is False
        assert score_result.check_results["law_citations"] is False
        assert score_result.total_score < 0.5

    def test_evaluation_details(self):
        """Test that evaluation provides detailed results and feedback."""
        ground_truth = {"taxable_income": 84000}
        llm_response = {
            "case_id": "test_case",
            "narrative": "Test case",
            "law_citations": [
                {
                    "ref": "§STD-DED",
                    "snippet": "Standard deduction is €10,000 per tax year."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Calculate taxable income",
                    "claim": "Final calculation",
                    "evidence": [
                        {
                            "source": "fact: salary",
                            "content": "80000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 83000,  # Slightly off
                "explanation": "Partial reasoning"
            }
        }
        
        score_result = score_case(
            case_id="test_case",
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["taxable income"],
            allowed_law_refs=["§STD-DED"],
            facts_for_math={"salary": 80000}
        )
        
        # Should provide detailed breakdown
        assert hasattr(score_result, 'details')
        assert hasattr(score_result, 'check_results')
        assert isinstance(score_result.check_results, dict)
        assert len(score_result.check_results) == 4  # All 4 evaluation dimensions


@pytest.fixture
def passing_case():
    """Create a case that should pass evaluation."""
    return {
        "ground_truth": {"taxable_income": 84000},
        "llm_response": {
            "case_id": "passing_test",
            "narrative": "Complete tax scenario",
            "law_citations": [
                {
                    "ref": "§STD-DED",
                    "snippet": "Standard deduction is €10,000 per tax year."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Calculate taxable income",
                    "claim": "Correct calculation",
                    "evidence": [
                        {
                            "source": "fact: salary",
                            "content": "84000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 84000,
                "explanation": "Accurate taxable income calculation"
            }
        },
        "required_steps": ["taxable income"],
        "allowed_law_refs": ["§STD-DED"],
        "facts_for_math": {"salary": 84000}
    }


@pytest.fixture
def failing_case():
    """Create a case that should fail evaluation."""
    return {
        "ground_truth": {"taxable_income": 84000},
        "llm_response": {
            "case_id": "failing_test",
            "narrative": "Incomplete tax scenario",
            "law_citations": [],  # Missing citations
            "reasoning_steps": [],  # Missing reasoning
            "final_answer": {
                "amount": 90000,  # Wrong amount
                "explanation": "Incorrect calculation"
            }
        },
        "required_steps": ["taxable income"],
        "allowed_law_refs": ["§STD-DED"],
        "facts_for_math": {"salary": 84000}
    }
