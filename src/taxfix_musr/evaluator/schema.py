"""
Pydantic models for LLM response validation.

Defines structured schemas that mirror the OUTPUT_SCHEMA.json contract
for validating LLM responses and ensuring compliance with requirements.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class LawCitation(BaseModel):
    """Law citation with reference and snippet."""
    ref: str = Field(..., description="Law reference ID")
    snippet: str = Field(..., description="Law text snippet")


class Evidence(BaseModel):
    """Evidence item cited in reasoning step."""
    source: str = Field(..., description="Source type (fact: or law:)")
    content: str = Field(..., description="Evidence content")

    @validator('source')
    def source_must_start_with_prefix(cls, v):
        """Ensure source starts with fact: or law:."""
        if not (v.startswith('fact:') or v.startswith('law:')):
            raise ValueError('Evidence source must start with "fact:" or "law:"')
        return v


class ReasoningStep(BaseModel):
    """Individual reasoning step with evidence."""
    step: str = Field(..., description="Step description")
    claim: str = Field(..., description="Claim made in this step")
    evidence: List[Evidence] = Field(..., description="Evidence supporting the claim")

    @validator('evidence')
    def evidence_not_empty(cls, v):
        """Ensure at least one evidence item per step."""
        if not v:
            raise ValueError('Each reasoning step must cite at least one evidence item')
        return v


class FinalAnswer(BaseModel):
    """Final answer with amount, verdict, and explanation."""
    amount: Optional[float] = Field(None, description="Numeric result")
    verdict: Optional[str] = Field(None, description="Categorical verdict")
    explanation: str = Field(..., description="Brief explanation of the answer")


class LLMCaseOutput(BaseModel):
    """Complete LLM response following OUTPUT_SCHEMA.json."""
    case_id: str = Field(..., description="Case identifier")
    narrative: str = Field(..., description="Concise case story")
    law_citations: List[LawCitation] = Field(..., description="Law citations used")
    reasoning_steps: List[ReasoningStep] = Field(..., description="Reasoning steps")
    final_answer: FinalAnswer = Field(..., description="Final computed answer")
    child_credit: Optional[float] = Field(None, description="Child tax credit amount (if applicable)")
    deduction_choice: Optional[float] = Field(None, description="Deduction amount chosen (standard or itemized)")
    deduction_path: Optional[str] = Field(None, description="Deduction path chosen ('standard' or 'itemized')")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # No extra keys allowed

    @validator('reasoning_steps')
    def reasoning_steps_not_empty(cls, v):
        """Ensure at least one reasoning step."""
        if not v:
            raise ValueError('At least one reasoning step is required')
        return v
