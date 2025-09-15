"""
Random case demonstration script.

Generates diverse random cases and shows batch evaluation with average scores.
Demonstrates the system's ability to handle varied tax scenarios.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.logic_tree import LogicTree
from taxfix_musr.schema_generator import random_case
from taxfix_musr.retriever import LawRetriever
from taxfix_musr.renderer import CaseRenderer
from taxfix_musr.llm_client import LLMClient
from taxfix_musr.evaluator.scorer import score_case
from taxfix_musr.agentic import run_with_retry
from taxfix_musr.evaluator.suite import run_batch, self_consistency, failure_taxonomy
from taxfix_musr.cache import get_cache, set_cache, LLMCache
from taxfix_musr.manifest import get_manifest_manager
from taxfix_musr.reports import ReportGenerator


def main():
    """Run end-to-end demo with random cases."""
    parser = argparse.ArgumentParser(description="Run random tax case evaluation")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--cases", type=int, default=3, help="Number of cases to run")
    parser.add_argument("--n", type=int, default=None, help="Number of cases for batch evaluation (overrides --cases)")
    parser.add_argument("--mode", default="embedded", help="Retrieval mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agentic", action="store_true", help="Enable agentic retry with critique")
    parser.add_argument("--threshold", type=float, default=0.85, help="Score threshold for retry (0.0-1.0)")
    parser.add_argument("--self-consistency", type=int, default=None, help="Enable self-consistency evaluation with k samples")
    parser.add_argument("--cache", action="store_true", default=True, help="Enable LLM output caching (default: True)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM output caching")
    parser.add_argument("--cache-dir", default="out/cache", help="Cache directory (default: out/cache)")
    parser.add_argument("--out", default="out", help="Output directory for reports (default: out)")
    
    args = parser.parse_args()
    
    # Use --n if provided, otherwise use --cases
    num_cases = args.n if args.n is not None else args.cases
    
    # Determine cache settings
    use_cache = args.cache and not args.no_cache
    
    print("=== Tax Law Reasoning Evaluation (Random Cases) ===")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Cases: {num_cases}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Agentic retry: {args.agentic}")
    if args.agentic:
        print(f"Retry threshold: {args.threshold}")
    if args.self_consistency:
        print(f"Self-consistency: {args.self_consistency} samples")
    print(f"Cache enabled: {use_cache}")
    if use_cache:
        print(f"Cache directory: {args.cache_dir}")
    print()
    
    # Initialize components
    try:
        llm_client = LLMClient(model=args.model)
        renderer = CaseRenderer(llm_client)
        retriever = LawRetriever()
        
        # Initialize cache if enabled
        if use_cache:
            cache = LLMCache(Path(args.cache_dir))
            set_cache(cache)
        
        # Initialize manifest manager
        manifest_manager = get_manifest_manager()
        run_id = manifest_manager.start_run(
            provider=args.provider,
            model=args.model,
            temperature=getattr(llm_client, 'temperature', 0.2),
            seed=args.seed,
            cache_enabled=use_cache,
            cache_dir=args.cache_dir if use_cache else None
        )
        print(f"Run ID: {run_id}")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")
        return
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize report generator
    report_generator = ReportGenerator(output_dir)
    
    if args.self_consistency:
        # Self-consistency evaluation on a single case
        case = random_case(
            case_id="consistency_test_case",
            target_question="What is the taxable income and any applicable credits?"
        )
        
        print("=== Self-Consistency Evaluation ===")
        print(f"Case ID: {case.case_id}")
        print(f"Facts: {', '.join([f'{name}={fact.value}' for name, fact in case.facts.items()])}")
        
        # Compute ground truth
        logic_tree = LogicTree(case)
        ground_truth = logic_tree.compute()
        print(f"Expected taxable income: {ground_truth['taxable_income']}")
        if 'deduction_choice' in ground_truth:
            print(f"Expected deduction choice: {ground_truth['deduction_choice']}")
        if 'deduction_path' in ground_truth:
            print(f"Expected deduction path: {ground_truth['deduction_path']}")
        if 'child_credit' in ground_truth:
            print(f"Expected child credit: {ground_truth['child_credit']}")
        print()
        
        # Run self-consistency evaluation
        consistency_result = self_consistency(
            case=case,
            renderer=renderer,
            retriever=retriever,
            k=args.self_consistency,
            use_cache=use_cache
        )
        
        print(f"Self-consistency Agreement: {consistency_result.agreement_percentage:.1f}%")
        print(f"Majority Amount: {consistency_result.majority_amount}")
        print(f"Standard Deviation: {consistency_result.std_dev:.2f}")
        print(f"Individual Amounts: {consistency_result.individual_amounts}")
        print(f"Individual Scores: {[f'{s:.2f}' for s in consistency_result.individual_scores]}")
        
        # Save self-consistency results
        consistency_data = {
            "case_id": consistency_result.case_id,
            "num_samples": consistency_result.num_samples,
            "agreement_percentage": consistency_result.agreement_percentage,
            "majority_amount": consistency_result.majority_amount,
            "std_dev": consistency_result.std_dev,
            "individual_amounts": consistency_result.individual_amounts,
            "individual_scores": consistency_result.individual_scores,
            "ground_truth": ground_truth
        }
        
        with open(output_dir / "consistency_report.json", "w") as f:
            json.dump(consistency_data, f, indent=2)
        
        print(f"\nSelf-consistency results saved to: {output_dir / 'consistency_report.json'}")
        
    else:
        # Batch evaluation
        print("=== Batch Evaluation ===")
        
        # Define case generator function
        def case_generator():
            return random_case(
                case_id=f"batch_case_{random.randint(1000, 9999)}",
                target_question="What is the taxable income and any applicable credits?"
            )
        
        # Run batch evaluation
        batch_results = run_batch(
            n=num_cases,
            case_generator=case_generator,
            renderer=renderer,
            retriever=retriever,
            use_agentic=args.agentic,
            threshold=args.threshold,
            use_cache=use_cache
        )
        
        # Generate comprehensive reports
        config = {
            "provider": args.provider,
            "model": args.model,
            "temperature": getattr(llm_client, 'temperature', 0.2),
            "seed": args.seed,
            "threshold": args.threshold,
            "agentic": args.agentic
        }
        
        # Get cache stats if available
        cache_stats = None
        if use_cache:
            cache = get_cache()
            cache_stats = cache.get_cache_stats()
        
        # Generate all reports
        summary, case_records = report_generator.generate_all_reports(
            batch_results=batch_results,
            self_consistency_results=None,
            cache_stats=cache_stats,
            config=config
        )
        
        # Update manifest with cache stats
        if use_cache:
            manifest_manager.update_cache_stats(cache_stats)
        
        # Finish manifest
        manifest_manager.finish_run(
            total_cases=batch_results.total_cases,
            successful_cases=batch_results.passed_cases,
            failed_cases=batch_results.failed_cases,
            average_score=batch_results.average_score,
            std_dev=batch_results.std_dev,
            pass_rate=(batch_results.passed_cases / batch_results.total_cases) * 100
        )


if __name__ == "__main__":
    main()