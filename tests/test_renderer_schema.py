"""
Tests for renderer schema validation.

Tests that LLM responses conform to the strict JSON schema requirements
and that validation catches common errors.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.evaluator.schema import LLMCaseOutput, ReasoningStep, LawCitation, Evidence, FinalAnswer


class TestRendererSchema:
    """Test cases for renderer schema validation."""

    def test_valid_response_schema(self):
        """Test that valid responses pass schema validation."""
        valid_response = {
            "case_id": "test_case_1",
            "narrative": "A taxpayer with salary and freelance income, making charitable donations and retirement contributions.",
            "law_citations": [
                {
                    "ref": "§DON-10pct",
                    "snippet": "Charitable contributions are deductible up to 10% of gross income."
                },
                {
                    "ref": "§RET-6000",
                    "snippet": "Retirement contributions are deductible up to €6,000 per tax year."
                }
            ],
            "reasoning_steps": [
                {
                    "step": "Compute gross income",
                    "claim": "Gross income is the sum of salary and freelance income",
                    "evidence": [
                        {
                            "source": "fact: salary=80000",
                            "content": "Salary income of €80,000"
                        },
                        {
                            "source": "fact: freelance=15000",
                            "content": "Freelance income of €15,000"
                        }
                    ]
                },
                {
                    "step": "Apply donation cap",
                    "claim": "Allowable donation is limited to 10% of gross income",
                    "evidence": [
                        {
                            "source": "law: §DON-10pct",
                            "content": "Charitable contributions are deductible up to 10% of gross income."
                        },
                        {
                            "source": "fact: donation=5000",
                            "content": "Donation amount of €5,000"
                        }
                    ]
                }
            ],
            "final_answer": {
                "amount": 84000,
                "explanation": "Taxable income is gross income minus allowable deductions."
            }
        }
        
        # Test that valid response passes Pydantic validation
        llm_output = LLMCaseOutput(**valid_response)
        assert llm_output.case_id == "test_case_1"
        assert llm_output.final_answer.amount == 84000
        assert len(llm_output.reasoning_steps) == 2
        assert len(llm_output.law_citations) == 2

    def test_missing_required_fields(self):
        """Test that missing required fields are caught."""
        # Test missing case_id
        invalid_response = {
            "narrative": "Test narrative",
            "law_citations": [],
            "reasoning_steps": [],
            "final_answer": {"explanation": "Test explanation"}
        }
        
        with pytest.raises(ValueError, match="Field required"):
            LLMCaseOutput(**invalid_response)

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        valid_response = {
            "case_id": "test_case_1",
            "narrative": "Test narrative",
            "law_citations": [],
            "reasoning_steps": [],
            "final_answer": {"explanation": "Test explanation"},
            "extra_field": "This should be rejected"
        }
        
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            LLMCaseOutput(**valid_response)

    def test_reasoning_steps_validation(self):
        """Test reasoning steps structure validation."""
        # Test empty reasoning_steps array
        invalid_response = {
            "case_id": "test_case_1",
            "narrative": "Test narrative",
            "law_citations": [],
            "reasoning_steps": [],
            "final_answer": {"explanation": "Test explanation"}
        }
        
        with pytest.raises(ValueError, match="At least one reasoning step is required"):
            LLMCaseOutput(**invalid_response)
        
        # Test reasoning step without evidence
        invalid_response["reasoning_steps"] = [
            {
                "step": "Test step",
                "claim": "Test claim",
                "evidence": []
            }
        ]
        
        with pytest.raises(ValueError, match="Each reasoning step must cite at least one evidence item"):
            LLMCaseOutput(**invalid_response)

    def test_law_citations_validation(self):
        """Test law citations structure validation."""
        # TODO: Test citation without ref
        # TODO: Test citation without snippet
        # TODO: Test citation with invalid ref format
        pass

    def test_final_answer_validation(self):
        """Test final answer structure validation."""
        # TODO: Test final_answer without explanation
        # TODO: Test explanation too short/long
        # TODO: Test amount field validation
        # TODO: Test verdict field validation
        pass

    def test_evidence_format_validation(self):
        """Test evidence format validation."""
        invalid_response = {
            "case_id": "test_case_1",
            "narrative": "Test narrative",
            "law_citations": [],
            "reasoning_steps": [
                {
                    "step": "Test step",
                    "claim": "Test claim",
                    "evidence": [
                        {
                            "source": "invalid_source",
                            "content": "Test content"
                        }
                    ]
                }
            ],
            "final_answer": {"explanation": "Test explanation"}
        }
        
        with pytest.raises(ValueError, match='Evidence source must start with "fact:" or "law:"'):
            LLMCaseOutput(**invalid_response)

    def test_json_parsing_errors(self):
        """Test handling of invalid JSON responses."""
        # TODO: Test malformed JSON
        # TODO: Test incomplete JSON
        # TODO: Test JSON with wrong data types
        pass


@pytest.fixture
def valid_llm_response():
    """Create a valid LLM response for testing."""
    # TODO: Create complete valid response
    # TODO: Include all required fields
    # TODO: Include proper evidence format
    # TODO: Include valid law citations
    pass


@pytest.fixture
def invalid_llm_responses():
    """Create various invalid LLM responses for testing."""
    # TODO: Create responses with different validation errors
    # TODO: Include missing fields
    # TODO: Include extra fields
    # TODO: Include malformed evidence
    pass
