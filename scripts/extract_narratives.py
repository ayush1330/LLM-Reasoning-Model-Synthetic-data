#!/usr/bin/env python3
"""
Utility script to extract narratives from comprehensive report.jsonl file.

This script demonstrates how to extract narrative data from the consolidated
report.jsonl file, replacing the need for a separate narratives.jsonl file.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def extract_narratives(report_file: Path, output_file: Path = None, format_type: str = "simple"):
    """
    Extract narratives from report.jsonl file.
    
    Args:
        report_file: Path to report.jsonl file
        output_file: Optional output file (defaults to narratives_extracted.jsonl)
        format_type: Format type - 'simple', 'detailed', or 'dataset'
    """
    if not report_file.exists():
        print(f"❌ Report file not found: {report_file}")
        return
    
    if output_file is None:
        output_file = report_file.parent / "narratives_extracted.jsonl"
    
    narratives = []
    
    with open(report_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            
            if format_type == "simple":
                # Simple narrative extraction
                narrative_entry = {
                    "case_id": record.get("case_id"),
                    "narrative": record.get("narrative")
                }
            elif format_type == "detailed":
                # Detailed extraction with context
                narrative_entry = {
                    "case_id": record.get("case_id"),
                    "narrative": record.get("narrative"),
                    "complexity_indicators": {
                        "score_achieved": record.get("score"),
                        "retry_needed": record.get("retry_used", False),
                        "failed_checks": record.get("failed_checks", [])
                    },
                    "financial_context": {
                        "gross_income": record.get("gross_income"),
                        "deduction_path": record.get("deduction_path"),
                        "taxable_income": record.get("taxable_income"),
                        "child_credit": record.get("child_credit")
                    }
                }
            elif format_type == "dataset":
                # Dataset format for ML training
                narrative_entry = {
                    "case_id": record.get("case_id"),
                    "text": record.get("narrative"),
                    "labels": {
                        "difficulty": "high" if record.get("score", 1.0) < 0.8 else "medium" if record.get("score", 1.0) < 0.95 else "low",
                        "requires_retry": record.get("retry_used", False),
                        "domain": "tax_law"
                    },
                    "metadata": {
                        "deduction_type": record.get("deduction_path"),
                        "has_children": record.get("child_credit", 0) > 0,
                        "income_level": "high" if record.get("gross_income", 0) > 80000 else "medium" if record.get("gross_income", 0) > 40000 else "low"
                    }
                }
            
            narratives.append(narrative_entry)
    
    # Save extracted narratives
    with open(output_file, 'w') as f:
        for narrative in narratives:
            f.write(json.dumps(narrative) + "\n")
    
    print(f"✅ Extracted {len(narratives)} narratives to: {output_file}")
    print(f"📊 Format: {format_type}")
    
    # Show sample
    if narratives:
        print(f"\n📝 Sample extracted narrative:")
        sample = narratives[0]
        print(json.dumps(sample, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Extract narratives from report.jsonl")
    parser.add_argument("report_file", help="Path to report.jsonl file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-f", "--format", choices=["simple", "detailed", "dataset"], 
                       default="simple", help="Output format")
    
    args = parser.parse_args()
    
    report_path = Path(args.report_file)
    output_path = Path(args.output) if args.output else None
    
    extract_narratives(report_path, output_path, args.format)


if __name__ == "__main__":
    main()
