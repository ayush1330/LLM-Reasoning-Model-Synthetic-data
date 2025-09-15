"""
Human evaluation baseline system for tax law reasoning.

Provides tools for collecting human annotations and comparing
LLM performance against human expert reasoning.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from pathlib import Path


@dataclass
class HumanAnnotation:
    """Human annotation for a tax law case."""
    case_id: str
    annotator_id: str
    timestamp: str
    
    # Human-provided answers
    expected_amount: float
    reasoning_steps: List[str]
    key_considerations: List[str]
    difficulty_rating: int  # 1-10 scale
    confidence_level: int  # 1-10 scale
    
    # Quality ratings for LLM response (if provided)
    llm_amount_accuracy: Optional[int] = None  # 1-10 scale
    llm_reasoning_quality: Optional[int] = None  # 1-10 scale
    llm_law_usage: Optional[int] = None  # 1-10 scale
    llm_overall_quality: Optional[int] = None  # 1-10 scale
    
    # Comments
    comments: str = ""
    improvement_suggestions: List[str] = None


@dataclass
class HumanBaselineResults:
    """Results from human baseline evaluation."""
    total_cases: int
    annotated_cases: int
    
    # Human performance metrics
    human_accuracy_rate: float
    average_confidence: float
    average_difficulty: float
    
    # LLM vs Human comparison
    llm_human_agreement_rate: float
    llm_outperforms_human_rate: float
    human_outperforms_llm_rate: float
    
    # Detailed breakdowns
    difficulty_breakdown: Dict[str, Any]
    common_disagreements: List[Dict[str, Any]]
    improvement_areas: List[str]


class HumanEvaluationInterface:
    """Interface for collecting human evaluations."""
    
    def __init__(self, annotation_file: str = "human_annotations.jsonl"):
        """Initialize human evaluation interface."""
        self.annotation_file = Path(annotation_file)
        self.annotations = self._load_existing_annotations()
    
    def _load_existing_annotations(self) -> List[HumanAnnotation]:
        """Load existing annotations from file."""
        annotations = []
        if self.annotation_file.exists():
            with open(self.annotation_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    annotations.append(HumanAnnotation(**data))
        return annotations
    
    def _save_annotation(self, annotation: HumanAnnotation):
        """Save a single annotation to file."""
        with open(self.annotation_file, 'a') as f:
            annotation_dict = {
                'case_id': annotation.case_id,
                'annotator_id': annotation.annotator_id,
                'timestamp': annotation.timestamp,
                'expected_amount': annotation.expected_amount,
                'reasoning_steps': annotation.reasoning_steps,
                'key_considerations': annotation.key_considerations,
                'difficulty_rating': annotation.difficulty_rating,
                'confidence_level': annotation.confidence_level,
                'llm_amount_accuracy': annotation.llm_amount_accuracy,
                'llm_reasoning_quality': annotation.llm_reasoning_quality,
                'llm_law_usage': annotation.llm_law_usage,
                'llm_overall_quality': annotation.llm_overall_quality,
                'comments': annotation.comments,
                'improvement_suggestions': annotation.improvement_suggestions or []
            }
            f.write(json.dumps(annotation_dict) + '\n')
    
    def collect_human_annotation(
        self,
        case_id: str,
        case_facts: Dict[str, Any],
        law_snippets: Dict[str, str],
        question: str,
        annotator_id: str,
        llm_response: Optional[Dict[str, Any]] = None
    ) -> HumanAnnotation:
        """
        Collect human annotation for a case (interactive mode).
        
        In a real implementation, this would present an interface
        for human annotators. For now, it returns a template.
        """
        print(f"\n=== HUMAN ANNOTATION REQUEST ===")
        print(f"Case ID: {case_id}")
        print(f"Annotator: {annotator_id}")
        print(f"\nCase Facts:")
        for key, value in case_facts.items():
            print(f"  {key}: {value}")
        
        print(f"\nAvailable Laws:")
        for ref, snippet in law_snippets.items():
            print(f"  {ref}: {snippet}")
        
        print(f"\nQuestion: {question}")
        
        if llm_response:
            print(f"\nLLM Response:")
            print(f"  Amount: {llm_response.get('final_answer', {}).get('amount', 'N/A')}")
            print(f"  Steps: {len(llm_response.get('reasoning_steps', []))} reasoning steps")
        
        # In a real implementation, this would collect actual human input
        # For now, return a template annotation
        return HumanAnnotation(
            case_id=case_id,
            annotator_id=annotator_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            expected_amount=0.0,  # Would be filled by human
            reasoning_steps=[],  # Would be filled by human
            key_considerations=[],  # Would be filled by human
            difficulty_rating=5,  # Would be filled by human
            confidence_level=5,  # Would be filled by human
            comments="Template annotation - replace with actual human input"
        )
    
    def create_annotation_template(
        self,
        case_id: str,
        case_facts: Dict[str, Any],
        law_snippets: Dict[str, str],
        question: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        llm_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an annotation template for human evaluators."""
        template = {
            "case_id": case_id,
            "case_facts": case_facts,
            "law_snippets": law_snippets,
            "question": question,
            "ground_truth": ground_truth,
            "llm_response": llm_response,
            
            # Fields for human annotator to fill
            "human_annotation": {
                "annotator_id": "FILL_IN",
                "expected_amount": "CALCULATE",
                "reasoning_steps": [
                    "Step 1: FILL_IN",
                    "Step 2: FILL_IN",
                    "Step 3: FILL_IN"
                ],
                "key_considerations": [
                    "FILL_IN important tax considerations"
                ],
                "difficulty_rating": "1-10 (1=very easy, 10=very hard)",
                "confidence_level": "1-10 (1=not confident, 10=very confident)",
                "comments": "Any additional notes or observations"
            },
            
            # LLM evaluation (if LLM response provided)
            "llm_evaluation": {
                "amount_accuracy": "1-10 (how accurate is the final amount?)",
                "reasoning_quality": "1-10 (how good is the reasoning process?)",
                "law_usage": "1-10 (how well are laws cited and applied?)",
                "overall_quality": "1-10 (overall response quality)",
                "improvement_suggestions": [
                    "What could the LLM do better?"
                ]
            } if llm_response else None
        }
        
        return template


class HumanBaselineAnalyzer:
    """Analyzer for comparing LLM performance against human baselines."""
    
    def __init__(self, annotations: List[HumanAnnotation]):
        """Initialize with human annotations."""
        self.annotations = annotations
    
    def analyze_human_baseline(
        self,
        llm_results: List[Dict[str, Any]],
        ground_truth_results: List[Dict[str, Any]]
    ) -> HumanBaselineResults:
        """
        Analyze LLM performance against human baseline.
        
        Args:
            llm_results: List of LLM evaluation results
            ground_truth_results: List of ground truth computations
            
        Returns:
            Human baseline analysis results
        """
        if not self.annotations:
            return self._create_empty_results()
        
        # Create lookup dictionaries
        llm_lookup = {result["case_id"]: result for result in llm_results}
        gt_lookup = {result["case_id"]: result for result in ground_truth_results}
        
        # Analyze human performance
        human_correct = 0
        total_confidence = 0
        total_difficulty = 0
        
        # Analyze LLM vs Human agreement
        agreements = 0
        llm_better = 0
        human_better = 0
        
        common_disagreements = []
        improvement_areas = []
        
        for annotation in self.annotations:
            case_id = annotation.case_id
            
            # Human accuracy
            gt_result = gt_lookup.get(case_id)
            if gt_result:
                expected_amount = gt_result.get("taxable_income", 0)
                human_amount = annotation.expected_amount
                
                # Allow 1% tolerance for human calculations
                tolerance = max(1.0, abs(expected_amount) * 0.01)
                if abs(human_amount - expected_amount) <= tolerance:
                    human_correct += 1
            
            total_confidence += annotation.confidence_level
            total_difficulty += annotation.difficulty_rating
            
            # LLM vs Human comparison
            llm_result = llm_lookup.get(case_id)
            if llm_result and gt_result:
                llm_amount = llm_result.get("llm_response", {}).get("final_answer", {}).get("amount", 0)
                human_amount = annotation.expected_amount
                expected_amount = gt_result.get("taxable_income", 0)
                
                # Calculate errors
                llm_error = abs(llm_amount - expected_amount)
                human_error = abs(human_amount - expected_amount)
                
                tolerance = max(1.0, abs(expected_amount) * 0.01)
                
                if llm_error <= tolerance and human_error <= tolerance:
                    agreements += 1
                elif llm_error < human_error:
                    llm_better += 1
                elif human_error < llm_error:
                    human_better += 1
                else:
                    # Both wrong, check which is closer
                    if llm_error < human_error:
                        llm_better += 1
                    else:
                        human_better += 1
                
                # Track disagreements
                if abs(llm_amount - human_amount) > tolerance:
                    common_disagreements.append({
                        "case_id": case_id,
                        "llm_amount": llm_amount,
                        "human_amount": human_amount,
                        "expected_amount": expected_amount,
                        "difficulty": annotation.difficulty_rating
                    })
            
            # Collect improvement suggestions
            if annotation.improvement_suggestions:
                improvement_areas.extend(annotation.improvement_suggestions)
        
        # Calculate metrics
        total_annotations = len(self.annotations)
        human_accuracy_rate = human_correct / total_annotations if total_annotations > 0 else 0
        average_confidence = total_confidence / total_annotations if total_annotations > 0 else 0
        average_difficulty = total_difficulty / total_annotations if total_annotations > 0 else 0
        
        total_comparisons = agreements + llm_better + human_better
        if total_comparisons > 0:
            agreement_rate = agreements / total_comparisons
            llm_outperforms_rate = llm_better / total_comparisons
            human_outperforms_rate = human_better / total_comparisons
        else:
            agreement_rate = llm_outperforms_rate = human_outperforms_rate = 0
        
        # Difficulty breakdown
        difficulty_breakdown = self._analyze_difficulty_breakdown(ground_truth_results)
        
        # Common improvement areas
        improvement_counter = {}
        for suggestion in improvement_areas:
            improvement_counter[suggestion] = improvement_counter.get(suggestion, 0) + 1
        
        top_improvements = sorted(improvement_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return HumanBaselineResults(
            total_cases=len(ground_truth_results),
            annotated_cases=total_annotations,
            human_accuracy_rate=human_accuracy_rate,
            average_confidence=average_confidence,
            average_difficulty=average_difficulty,
            llm_human_agreement_rate=agreement_rate,
            llm_outperforms_human_rate=llm_outperforms_rate,
            human_outperforms_llm_rate=human_outperforms_rate,
            difficulty_breakdown=difficulty_breakdown,
            common_disagreements=common_disagreements[:10],  # Top 10
            improvement_areas=[suggestion for suggestion, count in top_improvements]
        )
    
    def _create_empty_results(self) -> HumanBaselineResults:
        """Create empty results when no annotations are available."""
        return HumanBaselineResults(
            total_cases=0,
            annotated_cases=0,
            human_accuracy_rate=0.0,
            average_confidence=0.0,
            average_difficulty=0.0,
            llm_human_agreement_rate=0.0,
            llm_outperforms_human_rate=0.0,
            human_outperforms_llm_rate=0.0,
            difficulty_breakdown={},
            common_disagreements=[],
            improvement_areas=[]
        )
    
    def _analyze_difficulty_breakdown(self, ground_truth_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by difficulty level."""
        difficulty_groups = {
            "easy": [],      # 1-3
            "medium": [],    # 4-6
            "hard": []       # 7-10
        }
        
        for annotation in self.annotations:
            difficulty = annotation.difficulty_rating
            if difficulty <= 3:
                group = "easy"
            elif difficulty <= 6:
                group = "medium"
            else:
                group = "hard"
            
            difficulty_groups[group].append(annotation)
        
        breakdown = {}
        for group, annotations in difficulty_groups.items():
            if annotations:
                avg_confidence = sum(a.confidence_level for a in annotations) / len(annotations)
                breakdown[group] = {
                    "count": len(annotations),
                    "average_confidence": avg_confidence,
                    "cases": [a.case_id for a in annotations]
                }
            else:
                breakdown[group] = {"count": 0, "average_confidence": 0, "cases": []}
        
        return breakdown


def generate_annotation_batch(
    cases: List[Dict[str, Any]],
    output_dir: str = "human_evaluation"
) -> List[str]:
    """
    Generate a batch of annotation templates for human evaluators.
    
    Args:
        cases: List of case data with facts, laws, ground truth, LLM responses
        output_dir: Directory to save annotation templates
        
    Returns:
        List of generated template file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    interface = HumanEvaluationInterface()
    template_files = []
    
    for i, case_data in enumerate(cases):
        template = interface.create_annotation_template(
            case_id=case_data.get("case_id", f"case_{i}"),
            case_facts=case_data.get("facts", {}),
            law_snippets=case_data.get("law_snippets", {}),
            question=case_data.get("question", "What is the taxable income?"),
            ground_truth=case_data.get("ground_truth"),
            llm_response=case_data.get("llm_response")
        )
        
        template_file = output_path / f"annotation_template_{i+1}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        template_files.append(str(template_file))
    
    # Create instructions file
    instructions_file = output_path / "annotation_instructions.md"
    with open(instructions_file, 'w') as f:
        f.write("""# Human Annotation Instructions

## Overview
You will be evaluating tax law reasoning cases. For each case, you need to:
1. Calculate the correct taxable income
2. Provide your reasoning steps
3. Rate the difficulty and your confidence
4. Evaluate the LLM response (if provided)

## Guidelines

### Calculation Process
1. Calculate gross income (sum all income sources)
2. Apply donation cap (10% of gross income)
3. Apply retirement cap (€6,000 maximum)
4. Choose between standard (€10,000) and itemized deductions
5. Calculate taxable income (gross income - deductions)
6. Apply child tax credit with phase-out if applicable

### Difficulty Rating (1-10)
- 1-3: Basic arithmetic, no edge cases
- 4-6: Multiple deductions, some complexity
- 7-8: Phase-outs, optimization decisions
- 9-10: Multiple complex interactions

### Confidence Level (1-10)
- 1-3: Unsure about calculation
- 4-6: Reasonably confident
- 7-8: Very confident
- 9-10: Completely certain

### LLM Evaluation
Rate the LLM response on:
- Amount accuracy: How close is the final number?
- Reasoning quality: Are the steps logical and complete?
- Law usage: Are laws properly cited and applied?
- Overall quality: General assessment of the response

## File Format
Fill in the template fields marked with "FILL_IN" or "CALCULATE".
Save the completed file with the same name.
""")
    
    print(f"Generated {len(template_files)} annotation templates in {output_dir}/")
    print(f"Instructions saved to {instructions_file}")
    
    return template_files
