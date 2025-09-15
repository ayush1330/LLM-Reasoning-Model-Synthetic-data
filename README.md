# Taxfix MuSR - Synthetic Tax Law Reasoning Module

A Python module for generating synthetic tax law cases and evaluating LLM reasoning capabilities. Based on the MuSR (Multi-step Structured Reasoning) methodology.

## Overview

This system creates complex tax scenarios, generates ground truth via logic trees, and evaluates how well LLMs can reason through multi-step tax calculations. It's designed for testing LLM reasoning capabilities in domain-specific contexts.

## Key Features

- **Complex Narrative Generation**: Creates diverse tax scenarios with varying complexity levels
- **Ground Truth Computation**: Deterministic calculation of correct answers via logic trees
- **LLM Evaluation**: Multi-dimensional scoring of reasoning quality
- **Agentic Retry**: Self-correction mechanism for improved performance
- **Comprehensive Reporting**: JSON, JSONL, and Markdown output formats

## Quick Start

### Installation
```bash
# Install dependencies
uv add openai python-dotenv pydantic

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Basic Usage
```bash
# Single case evaluation
uv run python examples/run_manual.py

# Batch evaluation with 10 cases
uv run python examples/run_random.py --n 10

# Run with agentic retry
uv run python examples/run_random.py --n 5 --agentic --threshold 0.8
```

### Expected Output
```
=== Tax Law Reasoning Evaluation ===
Case ID: manual_case_1
Expected taxable income: 84000
LLM final answer: 84000.0
Total Score: 1.00
Passed: True
```

## Generated Narratives

The system generates complex tax scenarios instead of simple arithmetic problems:

**Basic**: "Calculate taxable income from salary of €80,000 and donation of €5,000"

**Complex**: "Sarah's income is near the child tax credit phase-out threshold. She needs to determine if her AGI will trigger the phase-out and calculate the reduced credit amount. Additionally, she's considering whether to increase retirement contributions to preserve more of her child tax credit."

### Complexity Levels
- **BASIC**: Simple arithmetic calculations
- **INTERMEDIATE**: Multi-step reasoning with caps and thresholds  
- **ADVANCED**: Strategic optimization scenarios
- **EXPERT**: Complex edge cases and exceptions

### Scenario Types
- **Phase-Out**: Income threshold calculations
- **Optimization**: Standard vs itemized deduction decisions
- **Family**: Child credit and dependent scenarios
- **Business**: Mixed personal/business income situations

## Programmatic Usage

```python
from taxfix_musr import random_case, LogicTree

# Generate a random tax case
case = random_case(case_id="example")

# Compute ground truth
logic_tree = LogicTree(case)
ground_truth = logic_tree.compute()

print(f"Expected taxable income: {ground_truth['taxable_income']}")
```

For full evaluation workflows, see the example scripts in `examples/`.

## Output Files

Running evaluations generates several output files:

- **`out/summary.json`**: Statistics and performance metrics
- **`out/summary.md`**: Human-readable report with tables
- **`out/report.jsonl`**: Complete case records including narratives
- **`out/run_manifest.json`**: Run metadata for reproducibility

### Extracting Narratives

To extract narratives for separate analysis:

```bash
# Extract narratives in different formats
uv run python scripts/extract_narratives.py out/report.jsonl -f simple
uv run python scripts/extract_narratives.py out/report.jsonl -f detailed
uv run python scripts/extract_narratives.py out/report.jsonl -f dataset
```

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
CACHE_DIR=out/cache  # optional
```

### Scoring Weights
The evaluation uses weighted scoring across four dimensions:
- **Amount Accuracy** (40%): Correctness of final numerical answer
- **Reasoning Quality** (30%): Presence of required reasoning steps
- **Law Citations** (20%): Proper reference to tax law snippets
- **Evidence Format** (10%): Structured evidence presentation

## Testing

```bash
# Run test suite
uv run pytest

# Run with coverage
uv run pytest --cov=src/taxfix_musr

# Lint code
uv run ruff check .
```

## Project Structure

```
src/taxfix_musr/          # Main package
├── models.py             # Core data structures
├── logic_tree.py         # Ground truth computation
├── schema_generator.py   # Case generation
├── llm_client.py         # LLM interface
├── renderer.py           # Narrative to LLM prompt conversion
├── narrative_generator.py # Complex scenario generation
└── evaluator/            # Evaluation system
    ├── scorer.py         # Scoring logic
    └── suite.py          # Batch evaluation

examples/                 # Usage examples
tests/                    # Test suite
docs/                     # Schema and prompt documentation
```

## Performance

- **Case Generation**: < 0.1 seconds per case
- **Logic Tree Computation**: < 0.01 seconds per case  
- **LLM Evaluation**: 2-10 seconds per case (model dependent)
- **Caching**: 50%+ hit rate for repeated cases

## Architecture

The system follows a three-stage MuSR methodology:

1. **Logic Tree Construction**: Build deterministic computation graphs from tax facts and rules
2. **Narrative Generation**: Convert structured cases into natural language scenarios
3. **Reasoning Evaluation**: Score LLM responses against ground truth with detailed metrics

This approach ensures reliable ground truth while testing complex reasoning capabilities.# LLM-Reasoning-Model-Synthetic-data
