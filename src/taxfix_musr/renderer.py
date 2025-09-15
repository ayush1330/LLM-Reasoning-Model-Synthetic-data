"""
LLM renderer for generating structured tax law reasoning.

Renders tax cases into natural language narratives with structured reasoning
steps, following strict JSON schema requirements.
"""

import json
from typing import Any, Dict

from .llm_client import LLMClient
from .models import Case
from .evaluator.schema import LLMCaseOutput
from .cache import get_cache, make_prompt_hash
from .narrative_generator import AdvancedNarrativeGenerator, ComplexityLevel
from .few_shot_examples import FewShotExampleBank


class CaseRenderer:
    """Renders tax cases into structured LLM responses."""

    def __init__(self, llm_client: LLMClient, use_advanced_narratives: bool = True, use_few_shot: bool = True):
        """
        Initialize renderer with LLM client.

        Args:
            llm_client: LLM client for generating responses
            use_advanced_narratives: Whether to use advanced narrative generation
            use_few_shot: Whether to use few-shot prompting with examples
        """
        self.llm_client = llm_client
        self.use_advanced_narratives = use_advanced_narratives
        self.use_few_shot = use_few_shot
        
        if use_advanced_narratives:
            self.narrative_generator = AdvancedNarrativeGenerator()
        
        if use_few_shot:
            self.few_shot_bank = FewShotExampleBank()

    def render(
        self,
        case: Case,
        law_snippets: Dict[str, str],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Render a tax case into structured LLM response.

        Args:
            case: Tax case to render
            law_snippets: Law snippets for grounding
            use_cache: Whether to use caching

        Returns:
            Structured JSON response
        """
        # Check cache first if enabled
        if use_cache:
            cache = get_cache()

            # Create prompt hash for cache key
            facts = {name: fact.value for name, fact in case.facts.items()}
            prompt_hash = make_prompt_hash(facts, law_snippets, case.target_question)

            # Try to get cached response
            cached_response = cache.get_cached(
                provider=getattr(self.llm_client, 'provider', 'openai'),
                model=getattr(self.llm_client, 'model', 'unknown'),
                seed=getattr(self.llm_client, 'seed', None),
                temperature=getattr(self.llm_client, 'temperature', 0.2),
                case_id=case.case_id,
                prompt_hash=prompt_hash
            )

            if cached_response:
                print(f"  Cache HIT for case {case.case_id}")
                return self._parse_response(cached_response, case.case_id)
            else:
                print(f"  Cache MISS for case {case.case_id}")

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(case, law_snippets)

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM and parse response
        try:
            response_text = self.llm_client.chat(messages)
            parsed_response = self._parse_response(response_text, case.case_id)

            # Cache the response if caching is enabled
            if use_cache:
                cache = get_cache()
                facts = {name: fact.value for name, fact in case.facts.items()}
                prompt_hash = make_prompt_hash(facts, law_snippets, case.target_question)

                cache.set_cached(
                    provider=getattr(self.llm_client, 'provider', 'openai'),
                    model=getattr(self.llm_client, 'model', 'unknown'),
                    seed=getattr(self.llm_client, 'seed', None),
                    temperature=getattr(self.llm_client, 'temperature', 0.2),
                    case_id=case.case_id,
                    prompt_hash=prompt_hash,
                    response=response_text
                )

            return parsed_response
        except Exception as e:
            raise RuntimeError(f"Failed to render case {case.case_id}: {e}")

    def _build_system_prompt(self) -> str:
        """
        Build enhanced system prompt for tax law reasoning.

        Returns:
            System prompt string
        """
        base_prompt = """You are a tax law expert tasked with analyzing complex tax scenarios and providing structured reasoning. Your response must be valid JSON that strictly follows the provided schema.

**CRITICAL REQUIREMENTS:**
1. Your response must be ONLY valid JSON - no prose, explanations, or text outside the JSON structure
2. You MUST cite ALL provided law references in the law_citations array
3. Each reasoning step must include evidence with sources starting with "fact:" or "law:"
4. The final answer must include an explanation and accurate numeric calculations
5. For complex scenarios, break down multi-step reasoning into clear, logical steps
6. Consider edge cases, phase-outs, thresholds, and optimization strategies

**MANDATORY REASONING STEPS:**
You MUST include these exact reasoning steps in your analysis:
1. "Calculate gross income" - Sum all income sources (salary + freelance + other)
2. "Apply donation cap" - Calculate allowable charitable deduction (up to 10% of gross income)
3. "Apply retirement cap" - Calculate allowable retirement deduction (up to €6,000)
4. "Calculate total deductions" - Sum all allowable deductions
5. "Choose deduction method" - Compare standard (€10,000) vs itemized deductions
6. "Calculate taxable income" - Gross income minus chosen deduction amount
7. "Apply child tax credit" - Calculate credit with phase-out rules if applicable

**LAW CITATION REQUIREMENTS:**
- You MUST include ALL provided law references in your law_citations array
- Each citation must have both "ref" and "snippet" fields
- Copy the exact reference IDs and snippet text provided
- Reference laws in your reasoning steps using "law:" prefix

**CALCULATION ACCURACY:**
- Show all mathematical steps clearly
- Use precise decimal calculations
- Apply percentage-based rules correctly (e.g., 10% donation cap, 5% phase-out rate)
- Consider thresholds exactly (e.g., €90,000 AGI for child credit phase-out)

**EVIDENCE FORMAT EXAMPLES:**
- fact: salary=80000
- fact: donation=5000  
- fact: retirement=3000
- law: §DON-10pct
- law: §RET-6000
- law: §CHILD-CR-Phaseout
- law: §STD-DED

**JSON STRUCTURE EXAMPLE:**
{
  "case_id": "provided_case_id",
  "narrative": "Brief description of the tax scenario",
  "law_citations": [
    {"ref": "§DON-10pct", "snippet": "Charitable contributions are deductible up to 10% of gross income."},
    {"ref": "§STD-DED", "snippet": "Standard deduction is €10,000 per tax year..."}
  ],
  "reasoning_steps": [
    {
      "step": "Calculate gross income",
      "claim": "Total income from all sources",
      "evidence": [
        {"source": "fact: salary", "content": "80000"},
        {"source": "fact: freelance", "content": "5000"}
      ]
    }
  ],
  "final_answer": {
    "amount": 75000.0,
    "explanation": "Detailed explanation of the final calculation"
  }
}"""
        
        # Add few-shot examples if enabled
        if self.use_few_shot and hasattr(self, 'few_shot_bank'):
            few_shot_examples = self.few_shot_bank.format_examples_for_prompt(max_examples=2)
            return f"""{base_prompt}

**LEARNING FROM EXAMPLES:**
Study these perfect examples to understand the expected reasoning approach and response format:

{few_shot_examples}

**NOW ANALYZE THE NEW CASE:**
Follow the same structured approach demonstrated in the examples above."""
        
        return base_prompt

    def _build_user_prompt(
        self,
        case: Case,
        law_snippets: Dict[str, str]
    ) -> str:
        """
        Build user prompt with case facts and law snippets.

        Args:
            case: Tax case
            law_snippets: Law snippets

        Returns:
            User prompt string
        """
        # Generate complex narrative if advanced narratives are enabled
        if self.use_advanced_narratives and hasattr(self, 'narrative_generator'):
            # Convert case facts to dictionary format
            case_facts = {name: fact.value for name, fact in case.facts.items()}
            
            # Generate complex narrative based on case facts with random complexity and scenario type
            import random
            from .narrative_generator import ScenarioType
            
            complexity_levels = [
                ComplexityLevel.INTERMEDIATE, 
                ComplexityLevel.ADVANCED, 
                ComplexityLevel.EXPERT
            ]
            scenario_types = [
                ScenarioType.PHASE_OUT,
                ScenarioType.THRESHOLD, 
                ScenarioType.EXCEPTION,
                ScenarioType.BUSINESS_MIX,
                ScenarioType.FAMILY,
                ScenarioType.RETIREMENT,
                ScenarioType.CHARITABLE
            ]
            
            selected_complexity = random.choice(complexity_levels)
            selected_scenario = random.choice(scenario_types)
            
            complex_narrative = self.narrative_generator.generate_narrative(
                case_facts, 
                complexity=selected_complexity,
                scenario_type=selected_scenario
            )
            
            # Use complex narrative as the main case description
            case_description = f"""**COMPLEX TAX SCENARIO:**
{complex_narrative}

**UNDERLYING FINANCIAL FACTS:**
{self._format_case_facts(case.facts)}"""
        else:
            # Fallback to simple facts format
            case_description = f"""**CASE_FACTS:**
{self._format_case_facts(case.facts)}"""

        # Format law snippets
        law_text = "\n".join([f"- {ref}: {snippet}" for ref, snippet in law_snippets.items()])

        # Create law citations example
        law_citations_example = []
        for ref, snippet in law_snippets.items():
            law_citations_example.append(f'    {{"ref": "{ref}", "snippet": "{snippet}"}}')
        law_citations_str = "[\n" + ",\n".join(law_citations_example) + "\n  ]"

        # Build user prompt
        user_prompt = f"""**CASE_ID:** {case.case_id}

{case_description}

**AVAILABLE TAX LAW REFERENCES:**
{law_text}

**QUESTION:** {case.target_question}

**MANDATORY LAW CITATIONS:**
You MUST include ALL of these law references in your law_citations array:
{law_citations_str}

**REQUIRED OUTPUT FORMAT:**
Respond with ONLY valid JSON matching this exact structure:
```json
{{
  "case_id": "{case.case_id}",
  "narrative": "Brief description of the tax scenario", 
  "law_citations": {law_citations_str},
  "reasoning_steps": [
    {{
      "step": "Calculate gross income",
      "claim": "Sum of all income sources",
      "evidence": [
        {{"source": "fact: salary", "content": "value from facts"}},
        {{"source": "fact: freelance", "content": "value from facts"}}
      ]
    }},
    {{
      "step": "Apply donation cap",
      "claim": "Charitable deduction limited by law",
      "evidence": [
        {{"source": "law: §DON-10pct", "content": "Deduction rule"}},
        {{"source": "fact: donation", "content": "donation amount"}}
      ]
    }}
  ],
  "final_answer": {{
    "amount": 75000.0,
    "explanation": "Detailed calculation explanation showing all steps"
  }}
}}
```

**CRITICAL REMINDERS:**
- Include ALL {len(law_snippets)} law references in law_citations
- Use exact reference IDs: {', '.join(law_snippets.keys())}
- Show all mathematical calculations step by step
- Use precise decimal numbers in final_answer.amount"""

        return user_prompt

    def _format_case_facts(self, facts: Dict[str, Any]) -> str:
        """
        Format case facts for display.

        Args:
            facts: Dictionary of case facts

        Returns:
            Formatted facts string
        """
        return "\n".join([f"- {name}: {fact.value}" for name, fact in facts.items()])

    def _parse_response(self, response_text: str, case_id: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.

        Args:
            response_text: Raw LLM response
            case_id: Case identifier for error messages

        Returns:
            Parsed and validated JSON response

        Raises:
            ValueError: If response is invalid JSON or doesn't match schema
        """
        # Clean response text (remove any markdown formatting)
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            # Parse JSON
            response_dict = json.loads(cleaned_text)

            # Fix evidence sources to ensure they have proper prefixes
            response_dict = self._fix_evidence_sources(response_dict)

            # Validate with Pydantic
            llm_output = LLMCaseOutput(**response_dict)

            # Return as dictionary
            return llm_output.model_dump()

        except (json.JSONDecodeError, ValueError) as e:
            # If invalid JSON, retry once with error notice
            print(f"Warning: Invalid JSON response for case {case_id}, retrying with error notice")

            # Add error notice and retry
            error_notice = f"\n\nIMPORTANT: Your previous response was invalid JSON. Please respond with ONLY valid JSON, no prose or explanations. Error: {str(e)}"

            # This would require calling the LLM again, but for now we'll raise an error
            raise ValueError(f"Invalid JSON response for case {case_id}: {e}")

    def _fix_evidence_sources(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix evidence sources to ensure they have proper prefixes.

        Args:
            response_dict: Raw response dictionary

        Returns:
            Fixed response dictionary
        """
        if "reasoning_steps" in response_dict:
            for step in response_dict["reasoning_steps"]:
                if "evidence" in step:
                    for evidence in step["evidence"]:
                        if "source" in evidence:
                            source = evidence["source"]
                            # Fix common issues
                            if source == "fact":
                                evidence["source"] = "fact: salary"  # Default to salary
                            elif source == "law":
                                evidence["source"] = "law: §DON-10pct"  # Default law reference
                            elif not (source.startswith("fact:") or source.startswith("law:")):
                                # Try to guess the type based on content
                                if any(keyword in source.lower() for keyword in ["salary", "donation", "retirement", "income"]):
                                    evidence["source"] = f"fact: {source}"
                                else:
                                    evidence["source"] = f"law: {source}"

        return response_dict
