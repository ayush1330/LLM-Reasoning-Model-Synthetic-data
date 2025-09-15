"""
Few-shot examples for tax law reasoning prompts.

Provides high-quality example cases with perfect responses to improve
LLM performance through demonstration-based learning.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class FewShotExample:
    """A single few-shot example with input case and expected response."""
    case_id: str
    facts: Dict[str, Any]
    law_snippets: Dict[str, str]
    question: str
    expected_response: Dict[str, Any]
    description: str


class FewShotExampleBank:
    """Bank of high-quality few-shot examples for tax law reasoning."""
    
    def __init__(self):
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[FewShotExample]:
        """Create a bank of high-quality few-shot examples."""
        return [
            # Example 1: Basic donation cap case
            FewShotExample(
                case_id="example_donation_cap",
                facts={
                    "salary": 75000,
                    "donation": 10000,
                    "retirement_contribution": 4000
                },
                law_snippets={
                    "§DON-10pct": "Charitable contributions are deductible up to 10% of gross income.",
                    "§RET-6000": "Retirement contributions are deductible up to €6,000 per tax year.",
                    "§STD-DED": "Standard deduction is €10,000 per tax year. Taxpayers may choose between standard deduction or itemized deductions, whichever is higher."
                },
                question="What is the taxable income?",
                expected_response={
                    "case_id": "example_donation_cap",
                    "narrative": "A taxpayer with €75,000 salary made €10,000 charitable donation and €4,000 retirement contribution. Need to apply donation cap and determine optimal deduction strategy.",
                    "law_citations": [
                        {
                            "ref": "§DON-10pct",
                            "snippet": "Charitable contributions are deductible up to 10% of gross income."
                        },
                        {
                            "ref": "§RET-6000", 
                            "snippet": "Retirement contributions are deductible up to €6,000 per tax year."
                        },
                        {
                            "ref": "§STD-DED",
                            "snippet": "Standard deduction is €10,000 per tax year. Taxpayers may choose between standard deduction or itemized deductions, whichever is higher."
                        }
                    ],
                    "reasoning_steps": [
                        {
                            "step": "Calculate gross income",
                            "claim": "Total income from all sources is €75,000",
                            "evidence": [
                                {
                                    "source": "fact: salary",
                                    "content": "75000"
                                }
                            ]
                        },
                        {
                            "step": "Apply donation cap",
                            "claim": "Charitable deduction is limited to €7,500 (10% of €75,000)",
                            "evidence": [
                                {
                                    "source": "law: §DON-10pct",
                                    "content": "Charitable contributions are deductible up to 10% of gross income."
                                },
                                {
                                    "source": "fact: donation",
                                    "content": "10000"
                                }
                            ]
                        },
                        {
                            "step": "Apply retirement cap",
                            "claim": "Retirement deduction is €4,000 (within €6,000 limit)",
                            "evidence": [
                                {
                                    "source": "law: §RET-6000",
                                    "content": "Retirement contributions are deductible up to €6,000 per tax year."
                                },
                                {
                                    "source": "fact: retirement_contribution",
                                    "content": "4000"
                                }
                            ]
                        },
                        {
                            "step": "Calculate total deductions",
                            "claim": "Total itemized deductions are €11,500 (€7,500 + €4,000)",
                            "evidence": [
                                {
                                    "source": "fact: donation_capped",
                                    "content": "7500"
                                },
                                {
                                    "source": "fact: retirement_contribution",
                                    "content": "4000"
                                }
                            ]
                        },
                        {
                            "step": "Choose deduction method",
                            "claim": "Itemized deductions (€11,500) exceed standard deduction (€10,000)",
                            "evidence": [
                                {
                                    "source": "law: §STD-DED",
                                    "content": "Standard deduction is €10,000 per tax year."
                                }
                            ]
                        },
                        {
                            "step": "Calculate taxable income",
                            "claim": "Taxable income is €63,500 (€75,000 - €11,500)",
                            "evidence": [
                                {
                                    "source": "fact: gross_income",
                                    "content": "75000"
                                },
                                {
                                    "source": "fact: total_deductions",
                                    "content": "11500"
                                }
                            ]
                        }
                    ],
                    "final_answer": {
                        "amount": 63500.0,
                        "explanation": "Taxable income calculated as gross income (€75,000) minus itemized deductions (€11,500), where donation is capped at 10% of gross income (€7,500) and retirement contribution (€4,000) is within limits."
                    }
                },
                description="Basic case demonstrating donation cap application and deduction choice"
            ),
            
            # Example 2: Child credit phase-out case
            FewShotExample(
                case_id="example_child_phaseout",
                facts={
                    "salary": 95000,
                    "children": 1,
                    "donation": 3000,
                    "retirement_contribution": 2000
                },
                law_snippets={
                    "§DON-10pct": "Charitable contributions are deductible up to 10% of gross income.",
                    "§RET-6000": "Retirement contributions are deductible up to €6,000 per tax year.",
                    "§STD-DED": "Standard deduction is €10,000 per tax year. Taxpayers may choose between standard deduction or itemized deductions, whichever is higher.",
                    "§CHILD-CR-Phaseout": "Child tax credit is €2,000 per qualifying child. The credit phases out at 5% for each euro of AGI above €90,000, reducing to zero when AGI reaches €130,000."
                },
                question="What is the taxable income and child tax credit?",
                expected_response={
                    "case_id": "example_child_phaseout",
                    "narrative": "High-income taxpayer with €95,000 salary and 1 child faces child tax credit phase-out. Need to calculate AGI and determine reduced credit amount.",
                    "law_citations": [
                        {
                            "ref": "§DON-10pct",
                            "snippet": "Charitable contributions are deductible up to 10% of gross income."
                        },
                        {
                            "ref": "§RET-6000",
                            "snippet": "Retirement contributions are deductible up to €6,000 per tax year."
                        },
                        {
                            "ref": "§STD-DED", 
                            "snippet": "Standard deduction is €10,000 per tax year. Taxpayers may choose between standard deduction or itemized deductions, whichever is higher."
                        },
                        {
                            "ref": "§CHILD-CR-Phaseout",
                            "snippet": "Child tax credit is €2,000 per qualifying child. The credit phases out at 5% for each euro of AGI above €90,000, reducing to zero when AGI reaches €130,000."
                        }
                    ],
                    "reasoning_steps": [
                        {
                            "step": "Calculate gross income",
                            "claim": "Gross income is €95,000 from salary",
                            "evidence": [
                                {
                                    "source": "fact: salary",
                                    "content": "95000"
                                }
                            ]
                        },
                        {
                            "step": "Apply donation cap",
                            "claim": "Donation deduction is €3,000 (within 10% limit of €9,500)",
                            "evidence": [
                                {
                                    "source": "law: §DON-10pct",
                                    "content": "Charitable contributions are deductible up to 10% of gross income."
                                },
                                {
                                    "source": "fact: donation",
                                    "content": "3000"
                                }
                            ]
                        },
                        {
                            "step": "Apply retirement cap",
                            "claim": "Retirement deduction is €2,000 (within €6,000 limit)",
                            "evidence": [
                                {
                                    "source": "law: §RET-6000",
                                    "content": "Retirement contributions are deductible up to €6,000 per tax year."
                                },
                                {
                                    "source": "fact: retirement_contribution",
                                    "content": "2000"
                                }
                            ]
                        },
                        {
                            "step": "Choose deduction method",
                            "claim": "Standard deduction (€10,000) exceeds itemized deductions (€5,000)",
                            "evidence": [
                                {
                                    "source": "law: §STD-DED",
                                    "content": "Standard deduction is €10,000 per tax year."
                                }
                            ]
                        },
                        {
                            "step": "Calculate taxable income",
                            "claim": "Taxable income is €85,000 (€95,000 - €10,000)",
                            "evidence": [
                                {
                                    "source": "fact: gross_income",
                                    "content": "95000"
                                },
                                {
                                    "source": "fact: standard_deduction",
                                    "content": "10000"
                                }
                            ]
                        },
                        {
                            "step": "Apply child tax credit",
                            "claim": "Child credit phases out: €1,750 (€2,000 - 5% × €5,000 excess over €90,000)",
                            "evidence": [
                                {
                                    "source": "law: §CHILD-CR-Phaseout",
                                    "content": "The credit phases out at 5% for each euro of AGI above €90,000"
                                },
                                {
                                    "source": "fact: agi",
                                    "content": "95000"
                                }
                            ]
                        }
                    ],
                    "final_answer": {
                        "amount": 85000.0,
                        "explanation": "Taxable income is €85,000 with reduced child tax credit of €1,750 due to AGI phase-out calculation (€2,000 - 5% × €5,000 excess)."
                    }
                },
                description="Complex case showing child tax credit phase-out calculations"
            )
        ]
    
    def get_examples_for_prompt(self, max_examples: int = 2) -> List[FewShotExample]:
        """Get examples to include in prompts."""
        return self.examples[:max_examples]
    
    def format_examples_for_prompt(self, max_examples: int = 2) -> str:
        """Format examples as text for inclusion in prompts."""
        examples = self.get_examples_for_prompt(max_examples)
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            law_snippets_text = "\n".join([f"- {ref}: {snippet}" for ref, snippet in example.law_snippets.items()])
            facts_text = "\n".join([f"- {key}: {value}" for key, value in example.facts.items()])
            
            example_text = f"""
**EXAMPLE {i}: {example.description}**

**CASE_ID:** {example.case_id}

**CASE_FACTS:**
{facts_text}

**AVAILABLE TAX LAW REFERENCES:**
{law_snippets_text}

**QUESTION:** {example.question}

**EXPECTED RESPONSE:**
```json
{self._format_json_response(example.expected_response)}
```
"""
            formatted_examples.append(example_text)
        
        return "\n".join(formatted_examples)
    
    def _format_json_response(self, response: Dict[str, Any]) -> str:
        """Format JSON response with proper indentation."""
        import json
        return json.dumps(response, indent=2, ensure_ascii=False)
