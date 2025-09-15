"""
Advanced narrative generator for complex tax law reasoning scenarios.

Creates diverse, challenging narratives that test multi-step reasoning,
edge cases, and complex tax law interactions beyond simple arithmetic.
"""

import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    """Complexity levels for narrative generation."""
    BASIC = "basic"           # Simple arithmetic
    INTERMEDIATE = "intermediate"  # Multi-step with edge cases
    ADVANCED = "advanced"     # Complex scenarios with exceptions
    EXPERT = "expert"         # Ambiguous cases requiring deep reasoning


class ScenarioType(Enum):
    """Types of tax scenarios to generate."""
    PHASE_OUT = "phase_out"           # Credit/deduction phase-outs
    THRESHOLD = "threshold"           # Income thresholds
    EXCEPTION = "exception"           # Special circumstances
    MULTI_YEAR = "multi_year"         # Cross-year considerations
    BUSINESS_MIX = "business_mix"     # Mixed business/personal
    FAMILY = "family"                 # Family tax situations
    RETIREMENT = "retirement"         # Complex retirement scenarios
    CHARITABLE = "charitable"         # Complex charitable giving


@dataclass
class NarrativeTemplate:
    """Template for generating complex narratives."""
    scenario_type: ScenarioType
    complexity: ComplexityLevel
    template: str
    variables: List[str]
    reasoning_challenges: List[str]


class AdvancedNarrativeGenerator:
    """Generates complex, diverse tax law narratives for reasoning testing."""

    def __init__(self):
        """Initialize with narrative templates."""
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> List[NarrativeTemplate]:
        """Initialize complex narrative templates."""
        return [
            # Phase-out scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.PHASE_OUT,
                complexity=ComplexityLevel.INTERMEDIATE,
                template="""Sarah, a software engineer, earned ${salary} in salary and ${freelance} from freelance consulting. She has ${children} qualifying children and made charitable donations of ${donation}. Her retirement contributions totaled ${retirement}. 

The challenge: Sarah's income is near the child tax credit phase-out threshold. She needs to determine if her AGI will trigger the phase-out and calculate the reduced credit amount. Additionally, she's considering whether to increase her retirement contributions to reduce her AGI and potentially preserve more of her child tax credit.

Key reasoning steps: Calculate AGI, determine phase-out impact, evaluate optimization strategies.""",
                variables=["salary", "freelance", "children", "donation", "retirement"],
                reasoning_challenges=["phase_out_calculation", "agi_optimization", "credit_preservation"]
            ),

            # Threshold scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.THRESHOLD,
                complexity=ComplexityLevel.ADVANCED,
                template="""Michael, a consultant, had a fluctuating income year. He earned ${salary} in base salary, ${bonus} in year-end bonus, and ${freelance} from side projects. He made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The complexity: Michael's total income is just above the standard deduction threshold where itemizing becomes beneficial. However, his charitable contributions were made in different months, and some were to organizations that may not qualify for full deduction. He also needs to consider whether his freelance income qualifies for the QBI deduction.

Key reasoning steps: Determine itemization vs standard deduction, verify charitable organization status, evaluate QBI eligibility.""",
                variables=["salary", "bonus", "freelance", "donation", "retirement"],
                reasoning_challenges=["threshold_analysis", "itemization_decision", "qbi_evaluation"]
            ),

            # Exception scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.EXCEPTION,
                complexity=ComplexityLevel.EXPERT,
                template="""Dr. Johnson, a physician, earned ${salary} from her medical practice and ${investment} from investment income. She made ${donation} in charitable contributions, including ${appreciated_stock} in appreciated stock donations. Her retirement contributions were ${retirement}.

The dilemma: Dr. Johnson's income exceeds the phase-out thresholds for most credits and deductions. However, she has significant charitable giving that might qualify for special treatment. She's also considering whether her medical practice expenses qualify for the QBI deduction, and whether her investment income affects her eligibility for certain credits.

Key reasoning steps: Analyze high-income limitations, evaluate special charitable rules, determine QBI eligibility with investment income.""",
                variables=["salary", "investment", "donation", "appreciated_stock", "retirement"],
                reasoning_challenges=["high_income_limitations", "special_charitable_rules", "qbi_with_investments"]
            ),

            # Business mix scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.BUSINESS_MIX,
                complexity=ComplexityLevel.ADVANCED,
                template="""Alex runs a small business and also works as a consultant. Business income was ${business_income}, consulting fees were ${consulting}, and they made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The complexity: Alex needs to determine how to allocate expenses between business and personal use, whether their consulting income qualifies for the QBI deduction, and how to optimize their retirement contributions across different account types. They're also considering whether to increase charitable giving to reduce their tax burden.

Key reasoning steps: Expense allocation, QBI calculation, retirement optimization, charitable giving strategy.""",
                variables=["business_income", "consulting", "donation", "retirement"],
                reasoning_challenges=["expense_allocation", "qbi_calculation", "retirement_optimization"]
            ),

            # Family scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.FAMILY,
                complexity=ComplexityLevel.INTERMEDIATE,
                template="""The Martinez family has a combined income of ${salary} from Mr. Martinez's job and ${freelance} from Mrs. Martinez's freelance work. They have ${children} children and made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The situation: The family is trying to maximize their child tax credit while also taking advantage of other deductions. They're considering whether to file jointly or separately, and whether Mrs. Martinez should increase her retirement contributions to reduce their overall tax burden.

Key reasoning steps: Filing status optimization, child credit maximization, retirement contribution strategy.""",
                variables=["salary", "freelance", "children", "donation", "retirement"],
                reasoning_challenges=["filing_status_optimization", "child_credit_maximization", "family_tax_planning"]
            ),

            # Retirement scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.RETIREMENT,
                complexity=ComplexityLevel.ADVANCED,
                template="""Robert, age 45, earned ${salary} in salary and ${freelance} from freelance work. He made ${donation} in charitable contributions and is considering retirement contributions of ${retirement}.

The challenge: Robert is trying to balance current tax savings with future retirement planning. He needs to determine the optimal mix of traditional and Roth contributions, considering his current tax bracket and expected future income. He's also evaluating whether to make additional catch-up contributions.

Key reasoning steps: Traditional vs Roth analysis, catch-up contribution eligibility, long-term tax planning.""",
                variables=["salary", "freelance", "donation", "retirement"],
                reasoning_challenges=["traditional_vs_roth", "catch_up_contributions", "long_term_planning"]
            ),

            # Multi-variable optimization scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.EXCEPTION,
                complexity=ComplexityLevel.EXPERT,
                template="""Jennifer, a high-earning executive, faces a complex tax optimization challenge. She earned ${salary} in base salary, ${bonus} in performance bonuses, and ${investment} from investment income. She made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The complexity: Jennifer's income places her in the highest tax bracket, triggering multiple phase-outs and limitations. She must strategically analyze the interaction between her child tax credit phase-out, charitable contribution limitations, and retirement contribution caps. Additionally, she's considering whether to accelerate charitable giving to offset the alternative minimum tax (AMT) impact.

Key reasoning steps: Analyze AMT implications, evaluate phase-out interactions, optimize charitable giving timing, determine optimal retirement contribution strategy, assess investment income impact on credits.""",
                variables=["salary", "bonus", "investment", "donation", "retirement"],
                reasoning_challenges=["amt_analysis", "phase_out_interactions", "charitable_timing", "retirement_optimization", "investment_impact"]
            ),

            # Cross-year planning scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.BUSINESS_MIX,
                complexity=ComplexityLevel.EXPERT,
                template="""David, a business owner, is implementing a sophisticated tax strategy across multiple years. His business generated ${business_income} in revenue, he earned ${consulting} from consulting work, and he made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The strategic challenge: David needs to optimize his tax position by considering the interplay between business income recognition, QBI deduction calculations, and charitable giving strategies. He's evaluating whether to defer income recognition, accelerate deductions, and optimize his business structure for maximum tax efficiency.

Key reasoning steps: Analyze QBI deduction optimization, evaluate income deferral strategies, assess charitable giving timing, determine optimal business structure, calculate multi-year tax impact.""",
                variables=["business_income", "consulting", "donation", "retirement"],
                reasoning_challenges=["qbi_optimization", "income_deferral", "charitable_timing", "business_structure", "multi_year_planning"]
            ),

            # Complex family scenarios
            NarrativeTemplate(
                scenario_type=ScenarioType.FAMILY,
                complexity=ComplexityLevel.EXPERT,
                template="""The Rodriguez family faces a multi-faceted tax optimization challenge. Mr. Rodriguez earned ${salary} from his corporate job, while Mrs. Rodriguez earned ${freelance} from her freelance consulting. They have ${children} children and made ${donation} in charitable contributions and ${retirement} in retirement contributions.

The strategic dilemma: The family must optimize their filing status, maximize child tax credits, and coordinate their retirement contributions to minimize their overall tax burden. They're considering whether to file jointly or separately, how to optimize their retirement contributions across different account types, and whether to adjust their charitable giving to maximize deductions.

Key reasoning steps: Analyze filing status optimization, evaluate child credit maximization, coordinate retirement contributions, assess charitable giving optimization, determine optimal tax strategy.""",
                variables=["salary", "freelance", "children", "donation", "retirement"],
                reasoning_challenges=["filing_optimization", "child_credit_max", "retirement_coordination", "charitable_optimization", "tax_strategy"]
            )
        ]

    def generate_narrative(
        self,
        case_facts: Dict[str, Any],
        complexity: ComplexityLevel = ComplexityLevel.INTERMEDIATE,
        scenario_type: Optional[ScenarioType] = None
    ) -> str:
        """
        Generate a complex narrative based on case facts.

        Args:
            case_facts: Dictionary of case facts (salary, freelance, donation, retirement, etc.)
            complexity: Desired complexity level
            scenario_type: Specific scenario type (optional)

        Returns:
            Generated narrative string
        """
        # Filter templates by complexity and scenario type
        available_templates = [
            t for t in self.templates
            if t.complexity == complexity and (scenario_type is None or t.scenario_type == scenario_type)
        ]

        if not available_templates:
            # Fallback to any template of the desired complexity
            available_templates = [t for t in self.templates if t.complexity == complexity]

        if not available_templates:
            # Ultimate fallback
            available_templates = self.templates

        # Select random template
        template = random.choice(available_templates)

        # Generate narrative with case facts
        narrative = template.template

        # Replace variables with case facts
        for variable in template.variables:
            # Map template variables to actual case fact keys
            fact_key = variable
            if variable == "retirement":
                fact_key = "retirement_contribution"
            elif variable == "children":
                fact_key = "children"
            
            value = case_facts.get(fact_key, 0)
            if isinstance(value, (int, float)):
                value = f"{value:,.0f}"
            narrative = narrative.replace(f"${{{variable}}}", str(value))

        return narrative

    def generate_reasoning_challenges(self, narrative: str) -> List[str]:
        """
        Extract reasoning challenges from a narrative.

        Args:
            narrative: Generated narrative

        Returns:
            List of reasoning challenges
        """
        challenges = []

        # Extract challenges based on narrative content (enhanced detection)
        if "phase-out" in narrative.lower():
            challenges.append("phase_out_calculation")
        if "threshold" in narrative.lower():
            challenges.append("threshold_analysis")
        if "optimization" in narrative.lower():
            challenges.append("optimization_strategy")
        if "qbi" in narrative.lower():
            challenges.append("qbi_evaluation")
        if "charitable" in narrative.lower():
            challenges.append("charitable_deduction_rules")
        if "retirement" in narrative.lower():
            challenges.append("retirement_planning")
        if "filing status" in narrative.lower():
            challenges.append("filing_status_optimization")

        # Enhanced challenge detection
        if "alternative minimum tax" in narrative.lower() or "amt" in narrative.lower():
            challenges.append("amt_analysis")
        if "interaction" in narrative.lower():
            challenges.append("multi_variable_interaction")
        if "strategic" in narrative.lower():
            challenges.append("strategic_planning")
        if "coordinate" in narrative.lower():
            challenges.append("coordination_analysis")
        if "maximize" in narrative.lower() or "minimize" in narrative.lower():
            challenges.append("optimization_analysis")
        if "cross-year" in narrative.lower() or "multi-year" in narrative.lower():
            challenges.append("multi_year_planning")
        if "timing" in narrative.lower():
            challenges.append("timing_optimization")
        if "structure" in narrative.lower():
            challenges.append("structural_analysis")
        if "sophisticated" in narrative.lower():
            challenges.append("sophisticated_reasoning")
        if "multi-faceted" in narrative.lower():
            challenges.append("multi_faceted_analysis")

        return challenges

    def get_complexity_metrics(self, narrative: str) -> Dict[str, Any]:
        """
        Analyze narrative complexity for evaluation.

        Args:
            narrative: Generated narrative

        Returns:
            Complexity metrics
        """
        return {
            "word_count": len(narrative.split()),
            "has_phase_out": "phase-out" in narrative.lower(),
            "has_threshold": "threshold" in narrative.lower(),
            "has_optimization": "optimization" in narrative.lower(),
            "has_exceptions": "exception" in narrative.lower() or "special" in narrative.lower(),
            "reasoning_steps_required": len(self.generate_reasoning_challenges(narrative)),
            "complexity_score": self._calculate_complexity_score(narrative)
        }

    def _calculate_complexity_score(self, narrative: str) -> float:
        """Calculate a complexity score for the narrative."""
        score = 0.0

        # Base score
        score += 1.0

        # Add points for complexity indicators (enhanced list)
        complexity_indicators = [
            "phase-out", "threshold", "optimization", "exception", "special",
            "considering", "evaluate", "determine", "analyze", "challenge",
            "dilemma", "situation", "complexity", "balance", "strategy",
            "strategic", "sophisticated", "multi-faceted", "interaction",
            "coordinate", "maximize", "minimize", "accelerate", "defer",
            "alternative minimum tax", "amt", "qbi", "filing status",
            "cross-year", "multi-year", "timing", "structure", "impact"
        ]

        for indicator in complexity_indicators:
            if indicator in narrative.lower():
                score += 0.4  # Slightly reduced per indicator

        # Add points for multiple reasoning steps
        if "Key reasoning steps:" in narrative:
            steps_text = narrative.split("Key reasoning steps:")[1].split(".")[0]
            step_count = len([s for s in steps_text.split(",") if s.strip()])
            score += step_count * 0.4  # Increased weight for reasoning steps

        # Add points for advanced reasoning patterns
        advanced_patterns = [
            "interaction between", "strategically analyze", "optimize",
            "coordinate", "maximize", "minimize", "assess", "calculate",
            "implement", "sophisticated", "multi-faceted", "cross-year",
            "alternative minimum tax", "phase-out interactions"
        ]

        for pattern in advanced_patterns:
            if pattern in narrative.lower():
                score += 0.6  # Higher weight for advanced patterns

        # Add points for multiple variables/considerations
        variable_indicators = ["income", "deduction", "credit", "contribution", "charitable"]
        variable_count = sum(1 for indicator in variable_indicators if indicator in narrative.lower())
        score += min(variable_count * 0.2, 2.0)  # Cap at 2.0 for variables

        return min(score, 10.0)  # Cap at 10.0


def create_diverse_narrative_dataset(
    num_cases: int = 20,
    complexity_distribution: Dict[ComplexityLevel, float] = None
) -> List[Dict[str, Any]]:
    """
    Create a diverse dataset of complex tax narratives.

    Args:
        num_cases: Number of cases to generate
        complexity_distribution: Distribution of complexity levels

    Returns:
        List of case dictionaries with narratives
    """
    if complexity_distribution is None:
        complexity_distribution = {
            ComplexityLevel.BASIC: 0.1,
            ComplexityLevel.INTERMEDIATE: 0.4,
            ComplexityLevel.ADVANCED: 0.4,
            ComplexityLevel.EXPERT: 0.1
        }

    generator = AdvancedNarrativeGenerator()
    cases = []

    for i in range(num_cases):
        # Generate random case facts
        case_facts = {
            "salary": random.randint(30000, 150000),
            "freelance": random.randint(5000, 50000),
            "donation": random.randint(1000, 20000),
            "retirement": random.randint(3000, 15000),
            "children": random.randint(0, 3),
            "investment": random.randint(0, 30000),
            "appreciated_stock": random.randint(0, 10000),
            "business_income": random.randint(20000, 100000),
            "consulting": random.randint(10000, 40000),
            "bonus": random.randint(0, 25000)
        }

        # Select complexity level based on distribution
        rand = random.random()
        cumulative = 0.0
        selected_complexity = ComplexityLevel.BASIC

        for complexity, probability in complexity_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                selected_complexity = complexity
                break

        # Generate narrative
        narrative = generator.generate_narrative(case_facts, selected_complexity)
        challenges = generator.generate_reasoning_challenges(narrative)
        metrics = generator.get_complexity_metrics(narrative)

        cases.append({
            "case_id": f"complex_case_{i+1:03d}",
            "narrative": narrative,
            "case_facts": case_facts,
            "complexity_level": selected_complexity.value,
            "reasoning_challenges": challenges,
            "complexity_metrics": metrics
        })

    return cases
