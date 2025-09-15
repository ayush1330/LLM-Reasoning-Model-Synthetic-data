"""
Law snippet retriever for RAG functionality.

Retrieves relevant law snippets based on law references for grounding
LLM reasoning in actual tax law text.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class LawRetriever:
    """Retrieves law snippets by reference."""

    def __init__(self, laws_file: Optional[str] = None):
        """
        Initialize retriever with law snippets.

        Args:
            laws_file: Path to laws JSON file (defaults to laws/demo_laws.json)
        """
        if laws_file is None:
            laws_file = Path(__file__).parent.parent.parent / "laws" / "demo_laws.json"

        self.laws_file = Path(laws_file)
        self.law_snippets: Dict[str, str] = {}
        self._load_laws()

    def fetch_by_refs(self, refs: List[str]) -> Dict[str, str]:
        """
        Fetch law snippets by reference IDs.

        Args:
            refs: List of law reference IDs

        Returns:
            Dictionary mapping ref -> snippet
        """
        result = {}
        for ref in refs:
            if ref in self.law_snippets:
                result[ref] = self.law_snippets[ref]
        return result

    def get_all_refs(self) -> List[str]:
        """
        Get all available law reference IDs.

        Returns:
            List of all available law reference IDs
        """
        return list(self.law_snippets.keys())

    def _load_laws(self) -> None:
        """Load law snippets from file."""
        try:
            with open(self.laws_file, 'r', encoding='utf-8') as f:
                self.law_snippets = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Laws file not found: {self.laws_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in laws file: {e}")
