"""
Manual case demonstration script.

Shows end-to-end pipeline with a deterministic case and passing evaluation.
Demonstrates the complete workflow from case generation to evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.logic_tree import LogicTree
from taxfix_musr.schema_generator import manual_case
from taxfix_musr.retriever import LawRetriever
from taxfix_musr.renderer import CaseRenderer
from taxfix_musr.llm_client import LLMClient
from taxfix_musr.evaluator.scorer import score_case
from taxfix_musr.agentic import run_with_retry
from taxfix_musr.cache import get_cache, set_cache, LLMCache
from taxfix_musr.manifest import get_manifest_manager


def main():
    """Run end-to-end demo with manual case."""
    parser = argparse.ArgumentParser(description="Run manual tax case evaluation")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--cases", type=int, default=1, help="Number of cases to run")
    parser.add_argument("--mode", default="embedded", help="Retrieval mode")
    parser.add_argument("--agentic", action="store_true", help="Enable agentic retry with critique")
    parser.add_argument("--threshold", type=float, default=0.85, help="Score threshold for retry (0.0-1.0)")
    parser.add_argument("--cache", action="store_true", default=True, help="Enable LLM output caching (default: True)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM output caching")
    parser.add_argument("--cache-dir", default="out/cache", help="Cache directory (default: out/cache)")
    
    args = parser.parse_args()
    
    # Determine cache settings
    use_cache = args.cache and not args.no_cache
    
    print("=== Tax Law Reasoning Evaluation ===")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Cases: {args.cases}")
    print(f"Mode: {args.mode}")
    print(f"Agentic retry: {args.agentic}")
    if args.agentic:
        print(f"Retry threshold: {args.threshold}")
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
            seed=getattr(llm_client, 'seed', None),
            cache_enabled=use_cache,
            cache_dir=args.cache_dir if use_cache else None
        )
        print(f"Run ID: {run_id}")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")
        return
    
    # Run cases
    total_score = 0.0
    passed_cases = 0
    
    for i in range(args.cases):
        print(f"--- Case {i+1} ---")
        
        # Create manual case with known facts
        case = manual_case(
            case_id=f"manual_case_{i+1}",
            facts={
                "salary": 80000,
                "freelance": 15000,
                "donation": 5000,
                "retirement_contribution": 7000,
            },
            target_question="What is the taxable income?"
        )
        
        # Compute ground truth
        logic_tree = LogicTree(case)
        ground_truth = logic_tree.compute()
        
        # Add case to manifest
        manifest_manager.add_case(case.case_id)
        
        print(f"Case ID: {case.case_id}")
        print(f"Expected taxable income: {ground_truth['taxable_income']}")
        if 'deduction_choice' in ground_truth:
            print(f"Expected deduction choice: {ground_truth['deduction_choice']}")
        if 'deduction_path' in ground_truth:
            print(f"Expected deduction path: {ground_truth['deduction_path']}")
        if 'child_credit' in ground_truth:
            print(f"Expected child credit: {ground_truth['child_credit']}")
        
        # Get law snippets from rule law references
        law_refs = []
        for rule in case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))  # Remove duplicates
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Prepare scoring parameters
        required_steps = ["gross income", "donation cap", "retirement cap"]
        facts_for_math = {name: fact.value for name, fact in case.facts.items()}
        
        # Process with or without agentic retry
        if args.agentic:
            try:
                retry_result = run_with_retry(
                    case=case,
                    renderer=renderer,
                    ground_truth=ground_truth,
                    law_snippets=law_snippets,
                    required_steps=required_steps,
                    allowed_law_refs=law_refs,
                    facts_for_math=facts_for_math,
                    threshold=args.threshold
                )
                
                llm_response = retry_result.output
                score_result = score_case(
                    case_id=case.case_id,
                    llm_response=llm_response,
                    ground_truth=ground_truth,
                    required_steps=required_steps,
                    allowed_law_refs=law_refs,
                    facts_for_math=facts_for_math
                )
                
                print(f"LLM final answer: {llm_response['final_answer'].get('amount', 'N/A')}")
                if 'deduction_choice' in ground_truth:
                    print(f"LLM deduction choice: {llm_response.get('deduction_choice', 'N/A')}")
                if 'deduction_path' in ground_truth:
                    print(f"LLM deduction path: {llm_response.get('deduction_path', 'N/A')}")
                if 'child_credit' in ground_truth:
                    print(f"LLM child credit: {llm_response.get('child_credit', 'N/A')}")
                print(f"Retries used: {retry_result.retries_used}")
                if retry_result.retries_used > 0:
                    print(f"Retry improved score: {retry_result.improved}")
                    if retry_result.critique:
                        print(f"Critique applied: {retry_result.critique}")
                        
            except Exception as e:
                print(f"Error in agentic processing: {e}")
                continue
        else:
            # Standard processing without retry
            try:
                llm_response = renderer.render(case, law_snippets, use_cache=use_cache)
                print(f"LLM final answer: {llm_response['final_answer'].get('amount', 'N/A')}")
                if 'deduction_choice' in ground_truth:
                    print(f"LLM deduction choice: {llm_response.get('deduction_choice', 'N/A')}")
                if 'deduction_path' in ground_truth:
                    print(f"LLM deduction path: {llm_response.get('deduction_path', 'N/A')}")
                if 'child_credit' in ground_truth:
                    print(f"LLM child credit: {llm_response.get('child_credit', 'N/A')}")
            except Exception as e:
                print(f"Error rendering case: {e}")
                continue
            
            score_result = score_case(
                case_id=case.case_id,
                llm_response=llm_response,
                ground_truth=ground_truth,
                required_steps=required_steps,
                allowed_law_refs=law_refs,
                facts_for_math=facts_for_math
            )
        
        # Print results
        print(f"Total Score: {score_result.total_score:.2f}")
        print(f"Passed: {score_result.passed}")
        print("Check Results:")
        for check_name, passed in score_result.check_results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        print("Details:")
        for check_name, details in score_result.details.items():
            if not score_result.check_results[check_name]:
                print(f"  {check_name}: {details.get('reason', 'Failed')}")
        
        total_score += score_result.total_score
        if score_result.passed:
            passed_cases += 1
        
        print()
    
    # Print summary
    if args.cases > 0:
        avg_score = total_score / args.cases
        pass_rate = passed_cases / args.cases
        print(f"=== Summary ===")
        print(f"Total Cases: {args.cases}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Pass Rate: {pass_rate:.2f} ({passed_cases}/{args.cases})")
        
        # Update cache stats and finish manifest
        if use_cache:
            cache = get_cache()
            cache_stats = cache.get_cache_stats()
            manifest_manager.update_cache_stats(cache_stats)
            print(f"\nCache Statistics:")
            print(f"  Total entries: {cache_stats['total_entries']}")
            print(f"  Unique models: {cache_stats['unique_models']}")
            print(f"  Unique cases: {cache_stats['unique_cases']}")
        
        # Finish manifest
        manifest_manager.finish_run(
            total_cases=args.cases,
            successful_cases=passed_cases,
            failed_cases=args.cases - passed_cases,
            average_score=avg_score,
            pass_rate=pass_rate
        )


if __name__ == "__main__":
    main()
