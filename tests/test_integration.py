"""
Integration tests for the complete tax law reasoning pipeline.

Tests the end-to-end functionality with real LLM calls to ensure
all components work together correctly in production scenarios.
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.llm_client import LLMClient
from taxfix_musr.schema_generator import manual_case, random_case
from taxfix_musr.logic_tree import LogicTree
from taxfix_musr.retriever import LawRetriever
from taxfix_musr.renderer import CaseRenderer
from taxfix_musr.evaluator.scorer import score_case
from taxfix_musr.agentic import run_with_retry
from taxfix_musr.evaluator.suite import run_batch, self_consistency


@pytest.fixture(scope="module")
def llm_client():
    """Create LLM client for integration tests."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration tests")
    
    return LLMClient(model="gpt-4o-mini", temperature=0.1, seed=42)


@pytest.fixture(scope="module")
def retriever():
    """Create law retriever."""
    return LawRetriever()


@pytest.fixture(scope="module")
def renderer(llm_client):
    """Create case renderer with LLM client."""
    return CaseRenderer(llm_client, use_few_shot=True)


@pytest.fixture
def simple_case():
    """Create a simple test case."""
    return manual_case(
        case_id="integration_test_simple",
        facts={
            "salary": 60000,
            "donation": 3000,
            "retirement_contribution": 2000
        }
    )


@pytest.fixture
def complex_case():
    """Create a complex test case with phase-outs."""
    return manual_case(
        case_id="integration_test_complex",
        facts={
            "salary": 95000,
            "freelance": 8000,
            "donation": 12000,
            "retirement_contribution": 6500,
            "children": 2
        }
    )


class TestEndToEndPipeline:
    """Test complete pipeline functionality."""
    
    @pytest.mark.integration
    def test_simple_case_pipeline(self, simple_case, renderer, retriever):
        """Test complete pipeline with a simple case."""
        # Compute ground truth
        logic_tree = LogicTree(simple_case)
        ground_truth = logic_tree.compute()
        
        # Get law snippets
        law_refs = []
        for rule in simple_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Render with LLM
        llm_response = renderer.render(simple_case, law_snippets, use_cache=False)
        
        # Validate response structure
        assert "case_id" in llm_response
        assert "law_citations" in llm_response
        assert "reasoning_steps" in llm_response
        assert "final_answer" in llm_response
        
        # Validate law citations
        assert len(llm_response["law_citations"]) > 0
        cited_refs = [citation["ref"] for citation in llm_response["law_citations"]]
        for ref in law_refs:
            assert ref in cited_refs, f"Missing law reference: {ref}"
        
        # Score the response
        score_result = score_case(
            case_id=simple_case.case_id,
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["gross income", "donation cap", "retirement cap"],
            allowed_law_refs=law_refs,
            facts_for_math={name: fact.value for name, fact in simple_case.facts.items()}
        )
        
        # Validate high score
        assert score_result.total_score >= 0.8, f"Score too low: {score_result.total_score}"
        assert score_result.check_results["law_citations"], "Law citations check failed"
        assert score_result.check_results["amount_accuracy"], "Amount accuracy check failed"
    
    @pytest.mark.integration
    def test_complex_case_pipeline(self, complex_case, renderer, retriever):
        """Test pipeline with complex case including phase-outs."""
        # Compute ground truth
        logic_tree = LogicTree(complex_case)
        ground_truth = logic_tree.compute()
        
        # Get law snippets
        law_refs = []
        for rule in complex_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Render with LLM
        llm_response = renderer.render(complex_case, law_snippets, use_cache=False)
        
        # Score the response
        score_result = score_case(
            case_id=complex_case.case_id,
            llm_response=llm_response,
            ground_truth=ground_truth,
            required_steps=["gross income", "donation cap", "retirement cap"],
            allowed_law_refs=law_refs,
            facts_for_math={name: fact.value for name, fact in complex_case.facts.items()}
        )
        
        # Complex cases should still achieve reasonable scores
        assert score_result.total_score >= 0.7, f"Complex case score too low: {score_result.total_score}"
        
        # Validate child credit calculation if applicable
        if "child_credit" in ground_truth:
            final_answer = llm_response.get("final_answer", {})
            # Should mention phase-out in explanation for high-income cases
            if ground_truth.get("gross_income", 0) > 90000:
                explanation = final_answer.get("explanation", "").lower()
                assert any(word in explanation for word in ["phase", "reduce", "limit"]), \
                    "Should mention phase-out for high-income cases"
    
    @pytest.mark.integration
    def test_few_shot_vs_regular_prompting(self, simple_case, llm_client, retriever):
        """Test that few-shot prompting improves performance."""
        # Get law snippets
        law_refs = []
        for rule in simple_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Ground truth
        logic_tree = LogicTree(simple_case)
        ground_truth = logic_tree.compute()
        
        # Test without few-shot
        renderer_no_few_shot = CaseRenderer(llm_client, use_few_shot=False)
        response_no_few_shot = renderer_no_few_shot.render(simple_case, law_snippets, use_cache=False)
        
        score_no_few_shot = score_case(
            case_id=simple_case.case_id,
            llm_response=response_no_few_shot,
            ground_truth=ground_truth,
            required_steps=["gross income", "donation cap"],
            allowed_law_refs=law_refs,
            facts_for_math={name: fact.value for name, fact in simple_case.facts.items()}
        )
        
        # Test with few-shot
        renderer_few_shot = CaseRenderer(llm_client, use_few_shot=True)
        response_few_shot = renderer_few_shot.render(simple_case, law_snippets, use_cache=False)
        
        score_few_shot = score_case(
            case_id=simple_case.case_id,
            llm_response=response_few_shot,
            ground_truth=ground_truth,
            required_steps=["gross income", "donation cap"],
            allowed_law_refs=law_refs,
            facts_for_math={name: fact.value for name, fact in simple_case.facts.items()}
        )
        
        # Few-shot should perform at least as well (allowing for some variance)
        assert score_few_shot.total_score >= score_no_few_shot.total_score - 0.1, \
            f"Few-shot ({score_few_shot.total_score}) should not be significantly worse than regular ({score_no_few_shot.total_score})"


class TestAgenticRetry:
    """Test agentic retry functionality."""
    
    @pytest.mark.integration
    def test_agentic_retry_improvement(self, simple_case, renderer, retriever):
        """Test that agentic retry can improve low scores."""
        # Get law snippets
        law_refs = []
        for rule in simple_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Ground truth
        logic_tree = LogicTree(simple_case)
        ground_truth = logic_tree.compute()
        
        # Run with retry
        retry_result = run_with_retry(
            case=simple_case,
            renderer=renderer,
            ground_truth=ground_truth,
            law_snippets=law_snippets,
            required_steps=["gross income", "donation cap"],
            allowed_law_refs=law_refs,
            facts_for_math={name: fact.value for name, fact in simple_case.facts.items()},
            threshold=0.95,  # High threshold to potentially trigger retry
            max_retries=2
        )
        
        # Should have a reasonable final score
        assert retry_result.final_score >= 0.8, f"Final retry score too low: {retry_result.final_score}"
        
        # Validate response structure
        assert retry_result.final_response is not None
        assert "law_citations" in retry_result.final_response


class TestBatchEvaluation:
    """Test batch evaluation functionality."""
    
    @pytest.mark.integration
    def test_batch_evaluation(self, renderer, retriever):
        """Test batch evaluation with multiple random cases."""
        def case_generator():
            return random_case()
        
        # Run small batch
        batch_results = run_batch(
            n=3,
            case_generator=case_generator,
            renderer=renderer,
            retriever=retriever,
            use_agentic=False,
            threshold=0.8,
            use_cache=False
        )
        
        # Validate batch results
        assert batch_results.total_cases == 3
        assert len(batch_results.individual_scores) == 3
        assert 0.0 <= batch_results.average_score <= 1.0
        assert batch_results.passed_cases + batch_results.failed_cases == 3
        
        # At least some cases should pass with improved system
        assert batch_results.average_score >= 0.6, f"Batch average too low: {batch_results.average_score}"


class TestSelfConsistency:
    """Test self-consistency evaluation."""
    
    @pytest.mark.integration
    def test_self_consistency(self, simple_case, renderer, retriever):
        """Test self-consistency with multiple samples."""
        # Run self-consistency test
        consistency_result = self_consistency(
            case=simple_case,
            renderer=renderer,
            retriever=retriever,
            k=3,  # Small number for faster testing
            temperature=0.3,
            use_cache=False
        )
        
        # Validate results
        assert consistency_result.num_samples == 3
        assert len(consistency_result.individual_amounts) == 3
        assert len(consistency_result.individual_scores) == 3
        assert 0.0 <= consistency_result.agreement_percentage <= 100.0
        
        # With good prompting, should have reasonable consistency
        assert consistency_result.agreement_percentage >= 60.0, \
            f"Consistency too low: {consistency_result.agreement_percentage}%"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.integration
    def test_high_income_edge_case(self, renderer, retriever):
        """Test very high income case with maximum phase-outs."""
        edge_case = manual_case(
            case_id="edge_high_income",
            facts={
                "salary": 150000,  # Above phase-out threshold
                "donation": 20000,  # Above cap
                "retirement_contribution": 8000,  # Above cap
                "children": 3
            }
        )
        
        # Get law snippets
        law_refs = []
        for rule in edge_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Should not crash and should produce reasonable response
        llm_response = renderer.render(edge_case, law_snippets, use_cache=False)
        
        # Validate response
        assert "final_answer" in llm_response
        assert "amount" in llm_response["final_answer"]
        
        # Should handle caps correctly
        final_amount = llm_response["final_answer"]["amount"]
        assert isinstance(final_amount, (int, float))
        assert final_amount > 0  # Should be positive
    
    @pytest.mark.integration
    def test_zero_income_edge_case(self, renderer, retriever):
        """Test edge case with zero or very low income."""
        edge_case = manual_case(
            case_id="edge_zero_income",
            facts={
                "salary": 5000,  # Very low income
                "donation": 1000,
                "retirement_contribution": 500,
                "children": 1
            }
        )
        
        # Get law snippets
        law_refs = []
        for rule in edge_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Should handle gracefully
        llm_response = renderer.render(edge_case, law_snippets, use_cache=False)
        
        # Validate response
        assert "final_answer" in llm_response
        final_amount = llm_response["final_answer"]["amount"]
        assert final_amount >= 0  # Should not be negative


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test performance characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_response_time_benchmark(self, simple_case, renderer, retriever):
        """Benchmark response times for performance regression."""
        import time
        
        # Get law snippets
        law_refs = []
        for rule in simple_case.rules.values():
            law_refs.extend(rule.law_refs)
        law_refs = list(set(law_refs))
        law_snippets = retriever.fetch_by_refs(law_refs)
        
        # Measure response time
        start_time = time.time()
        llm_response = renderer.render(simple_case, law_snippets, use_cache=False)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert response_time < 30.0, f"Response time too slow: {response_time}s"
        
        # Validate response quality wasn't sacrificed for speed
        assert "law_citations" in llm_response
        assert len(llm_response["law_citations"]) > 0
