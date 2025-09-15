"""
Report generation system for batch evaluation results.

Creates comprehensive summary reports in multiple formats for presentations
and analysis.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BatchSummary:
    """Comprehensive batch evaluation summary."""
    n_cases: int
    avg_score: float
    std_score: float
    pass_rate: float
    retry_rate: float
    retry_success_rate: float
    self_consistency_stats: Optional[Dict[str, Any]]
    top_failures: List[Tuple[str, int]]
    cache_stats: Optional[Dict[str, int]]
    config: Dict[str, Any]


@dataclass
class CaseRecord:
    """Individual case record for detailed reporting."""
    case_id: str
    narrative: str
    gross_income: float
    deduction_path: Optional[str]
    deduction_amount: float
    taxable_income: float
    child_credit: Optional[float]
    score: float
    retry_used: bool
    failed_checks: List[str]


class ReportGenerator:
    """Generates comprehensive evaluation reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(
        self,
        batch_results,
        self_consistency_results: Optional[Any] = None,
        cache_stats: Optional[Dict[str, int]] = None,
        config: Dict[str, Any] = None
    ) -> BatchSummary:
        """
        Generate comprehensive batch summary.

        Args:
            batch_results: Results from batch evaluation
            self_consistency_results: Self-consistency results if enabled
            cache_stats: Cache statistics if available
            config: Configuration used for the run

        Returns:
            Comprehensive batch summary
        """
        # Calculate retry statistics
        retry_used_count = sum(1 for report in batch_results.individual_scores if report.get('retry_used', False))
        retry_improved_count = sum(1 for report in batch_results.individual_scores 
                                 if report.get('retry_used', False) and report.get('retry_improved', False))

        retry_rate = (retry_used_count / batch_results.total_cases * 100) if batch_results.total_cases > 0 else 0
        retry_success_rate = (retry_improved_count / retry_used_count * 100) if retry_used_count > 0 else 0

        # Self-consistency stats
        sc_stats = None
        if self_consistency_results:
            sc_stats = {
                "agreement_percentage": self_consistency_results.agreement_percentage,
                "majority_amount": self_consistency_results.majority_amount,
                "std_dev": self_consistency_results.std_dev,
                "individual_amounts": self_consistency_results.individual_amounts
            }

        # Top failures from taxonomy
        top_failures = list(batch_results.failure_taxonomy.items())[:3]

        return BatchSummary(
            n_cases=batch_results.total_cases,
            avg_score=batch_results.average_score,
            std_score=batch_results.std_dev,
            pass_rate=(batch_results.passed_cases / batch_results.total_cases * 100) if batch_results.total_cases > 0 else 0,
            retry_rate=retry_rate,
            retry_success_rate=retry_success_rate,
            self_consistency_stats=sc_stats,
            top_failures=top_failures,
            cache_stats=cache_stats,
            config=config or {}
        )

    def create_case_records(self, batch_results) -> List[CaseRecord]:
        """
        Create individual case records from batch results.

        Args:
            batch_results: Results from batch evaluation

        Returns:
            List of case records
        """
        records = []

        for report in batch_results.individual_scores:
            # Extract case information
            case_id = report.get('case_id', 'unknown')

            # Extract narrative from LLM response
            llm_response = report.get('llm_response', {})
            if llm_response is None:
                llm_response = {}
            narrative = llm_response.get('narrative', 'No narrative available')

            # Extract facts (with safe defaults)
            facts = report.get('facts', {})
            gross_income = facts.get('gross_income', 0.0)

            # Extract outputs (with safe defaults)
            outputs = report.get('outputs', {})
            deduction_path = outputs.get('deduction_path')
            deduction_amount = outputs.get('deduction_choice', 0.0)
            taxable_income = outputs.get('taxable_income', 0.0)
            child_credit = outputs.get('child_credit')

            # Extract scoring information
            score = report.get('score', 0.0)
            retry_used = report.get('retry_used', False)

            # Extract failed checks
            failed_checks = []
            check_results = report.get('check_results', {})
            for check_name, passed in check_results.items():
                if not passed:
                    failed_checks.append(check_name)

            records.append(CaseRecord(
                case_id=case_id,
                narrative=narrative,
                gross_income=gross_income,
                deduction_path=deduction_path,
                deduction_amount=deduction_amount,
                taxable_income=taxable_income,
                child_credit=child_credit,
                score=score,
                retry_used=retry_used,
                failed_checks=failed_checks
            ))

        return records

    def print_console_table(self, case_records: List[CaseRecord]):
        """
        Print a formatted table to console.

        Args:
            case_records: List of case records to display
        """
        if not case_records:
            print("No cases to display.")
            return

        # Define column widths
        widths = {
            'case_id': 12,
            'gi': 8,
            'deduction_path': 12,
            'deduction': 10,
            'taxable_income': 12,
            'child_credit': 12,
            'score': 6,
            'retry': 6
        }

        # Print header
        header = (
            f"{'Case ID':<{widths['case_id']}} | "
            f"{'GI':<{widths['gi']}} | "
            f"{'Deduction Path':<{widths['deduction_path']}} | "
            f"{'Deduction':<{widths['deduction']}} | "
            f"{'Taxable Income':<{widths['taxable_income']}} | "
            f"{'Child Credit':<{widths['child_credit']}} | "
            f"{'Score':<{widths['score']}} | "
            f"{'Retry':<{widths['retry']}}"
        )
        print(header)
        print("-" * len(header))

        # Print data rows
        for record in case_records:
            # Format values with safe defaults
            case_id = record.case_id[:widths['case_id']-1] if len(record.case_id) > widths['case_id'] else record.case_id
            gi = f"{record.gross_income:,.0f}"
            deduction_path = record.deduction_path or "N/A"
            deduction = f"{record.deduction_amount:,.0f}"
            taxable_income = f"{record.taxable_income:,.0f}"
            child_credit = f"{record.child_credit:,.0f}" if record.child_credit is not None else "N/A"
            score = f"{record.score:.2f}"
            retry = "Yes" if record.retry_used else "No"

            row = (
                f"{case_id:<{widths['case_id']}} | "
                f"{gi:<{widths['gi']}} | "
                f"{deduction_path:<{widths['deduction_path']}} | "
                f"{deduction:<{widths['deduction']}} | "
                f"{taxable_income:<{widths['taxable_income']}} | "
                f"{child_credit:<{widths['child_credit']}} | "
                f"{score:<{widths['score']}} | "
                f"{retry:<{widths['retry']}}"
            )
            print(row)

    def save_summary_json(self, summary: BatchSummary):
        """
        Save summary as JSON file.

        Args:
            summary: Batch summary to save
        """
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"Summary saved to: {summary_path}")

    def save_summary_markdown(self, summary: BatchSummary, case_records: List[CaseRecord]):
        """
        Save summary as Markdown file.

        Args:
            summary: Batch summary to save
            case_records: Case records for table
        """
        md_path = self.output_dir / "summary.md"

        with open(md_path, "w") as f:
            f.write("# Batch Evaluation Summary\n\n")

            # Configuration
            f.write("## Configuration\n\n")
            config = summary.config
            f.write(f"- **Provider**: {config.get('provider', 'N/A')}\n")
            f.write(f"- **Model**: {config.get('model', 'N/A')}\n")
            f.write(f"- **Temperature**: {config.get('temperature', 'N/A')}\n")
            f.write(f"- **Seed**: {config.get('seed', 'N/A')}\n")
            f.write(f"- **Threshold**: {config.get('threshold', 'N/A')}\n")
            f.write(f"- **Agentic**: {config.get('agentic', False)}\n\n")

            # Statistics
            f.write("## Statistics\n\n")
            f.write(f"- **Cases**: {summary.n_cases}\n")
            f.write(f"- **Average Score**: {summary.avg_score:.2f}\n")
            f.write(f"- **Standard Deviation**: {summary.std_score:.2f}\n")
            f.write(f"- **Pass Rate**: {summary.pass_rate:.1f}%\n")
            f.write(f"- **Retry Rate**: {summary.retry_rate:.1f}%\n")
            f.write(f"- **Retry Success Rate**: {summary.retry_success_rate:.1f}%\n\n")

            # Self-consistency stats
            if summary.self_consistency_stats:
                f.write("## Self-Consistency\n\n")
                sc = summary.self_consistency_stats
                f.write(f"- **Agreement**: {sc['agreement_percentage']:.1f}%\n")
                f.write(f"- **Majority Amount**: {sc['majority_amount']}\n")
                f.write(f"- **Standard Deviation**: {sc['std_dev']:.2f}\n\n")

            # Top failures
            if summary.top_failures:
                f.write("## Top Failure Reasons\n\n")
                for reason, count in summary.top_failures:
                    f.write(f"- **{reason}**: {count} cases\n")
                f.write("\n")

            # Cache stats
            if summary.cache_stats:
                f.write("## Cache Statistics\n\n")
                cache = summary.cache_stats
                f.write(f"- **Total Entries**: {cache.get('total_entries', 0)}\n")
                f.write(f"- **Unique Models**: {cache.get('unique_models', 0)}\n")
                f.write(f"- **Unique Cases**: {cache.get('unique_cases', 0)}\n\n")

            # Case table
            f.write("## Case Results\n\n")
            f.write("| Case ID | GI | Deduction Path | Deduction | Taxable Income | Child Credit | Score | Retry |\n")
            f.write("|---------|----|----------------|-----------|----------------|--------------|-------|-------|\n")

            for record in case_records:
                case_id = record.case_id
                gi = f"{record.gross_income:,.0f}"
                deduction_path = record.deduction_path or "N/A"
                deduction = f"{record.deduction_amount:,.0f}"
                taxable_income = f"{record.taxable_income:,.0f}"
                child_credit = f"{record.child_credit:,.0f}" if record.child_credit is not None else "N/A"
                score = f"{record.score:.2f}"
                retry = "Yes" if record.retry_used else "No"

                f.write(f"| {case_id} | {gi} | {deduction_path} | {deduction} | {taxable_income} | {child_credit} | {score} | {retry} |\n")

        print(f"Markdown summary saved to: {md_path}")

    def save_report_jsonl(self, case_records: List[CaseRecord]):
        """
        Save comprehensive case evaluation records as JSONL file.
        
        This file contains complete information including:
        - Generated narratives and case details
        - Evaluation scores and metrics
        - Ground truth and LLM response data
        - Retry information and failure analysis
        
        This replaces the previous separate narratives.jsonl file to eliminate redundancy.

        Args:
            case_records: Case records to save
        """
        jsonl_path = self.output_dir / "report.jsonl"

        with open(jsonl_path, "w") as f:
            for record in case_records:
                record_dict = asdict(record)
                f.write(json.dumps(record_dict) + "\n")

        print(f"Comprehensive evaluation report saved to: {jsonl_path}")
        print("ℹ️  This file includes narratives, scores, and all evaluation data")

    def save_narratives_dataset(self, case_records: List[CaseRecord]):
        """
        DEPRECATED: Narratives are now included in report.jsonl to eliminate redundancy.
        This method is kept for backward compatibility but does nothing.
        
        To extract narratives from reports, use:
        cat report.jsonl | jq '{case_id, narrative, input_facts: {gross_income, deduction_path, taxable_income, child_credit}}'
        
        Args:
            case_records: Case records containing narratives (unused)
        """
        print("ℹ️  Narratives are included in report.jsonl - no separate file needed")
        print("ℹ️  To extract narratives: cat report.jsonl | jq '{case_id, narrative}'")
        pass

    def generate_all_reports(
        self,
        batch_results,
        self_consistency_results: Optional[Any] = None,
        cache_stats: Optional[Dict[str, int]] = None,
        config: Dict[str, Any] = None
    ):
        """
        Generate all report formats.

        Args:
            batch_results: Results from batch evaluation
            self_consistency_results: Self-consistency results if enabled
            cache_stats: Cache statistics if available
            config: Configuration used for the run
        """
        # Generate summary
        summary = self.generate_summary(batch_results, self_consistency_results, cache_stats, config)

        # Create case records
        case_records = self.create_case_records(batch_results)

        # Print console table
        print("\n=== Case Results Table ===")
        self.print_console_table(case_records)

        # Save all formats
        self.save_summary_json(summary)
        self.save_summary_markdown(summary, case_records)
        self.save_report_jsonl(case_records)
        # Narratives are now included in report.jsonl - no separate file needed
        
        return summary, case_records
