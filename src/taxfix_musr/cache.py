"""
LLM output caching system.

Provides persistent caching of LLM responses to avoid redundant API calls
and enable reproducible results across runs.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional



@dataclass
class CacheEntry:
    """A cached LLM response entry."""
    provider: str
    model: str
    seed: Optional[int]
    temperature: float
    case_id: str
    prompt_hash: str
    response: str
    timestamp: str
    template_version: str


class LLMCache:
    """SQLite-based cache for LLM responses."""

    def __init__(self, cache_dir: Path = Path("out/cache")):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache database
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    seed INTEGER,
                    temperature REAL NOT NULL,
                    case_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    template_version TEXT NOT NULL,
                    UNIQUE(provider, model, seed, temperature, case_id, prompt_hash)
                )
            """)

            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                ON cache_entries(provider, model, seed, temperature, case_id, prompt_hash)
            """)

    def get_cached(
        self,
        provider: str,
        model: str,
        seed: Optional[int],
        temperature: float,
        case_id: str,
        prompt_hash: str
    ) -> Optional[str]:
        """
        Retrieve cached response if available.

        Args:
            provider: LLM provider (e.g., 'openai')
            model: Model name (e.g., 'gpt-4o-mini')
            seed: Random seed (can be None)
            temperature: Sampling temperature
            case_id: Case identifier
            prompt_hash: Hash of the prompt

        Returns:
            Cached response string or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT response FROM cache_entries
                WHERE provider = ? AND model = ? AND seed IS ? AND temperature = ?
                AND case_id = ? AND prompt_hash = ?
            """, (provider, model, seed, temperature, case_id, prompt_hash))

            result = cursor.fetchone()
            return result[0] if result else None

    def set_cached(
        self,
        provider: str,
        model: str,
        seed: Optional[int],
        temperature: float,
        case_id: str,
        prompt_hash: str,
        response: str,
        template_version: str = "1.0"
    ):
        """
        Store response in cache.

        Args:
            provider: LLM provider
            model: Model name
            seed: Random seed
            temperature: Sampling temperature
            case_id: Case identifier
            prompt_hash: Hash of the prompt
            response: LLM response to cache
            template_version: Version of the prompt template
        """
        import datetime

        timestamp = datetime.datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries
                (provider, model, seed, temperature, case_id, prompt_hash, response, timestamp, template_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (provider, model, seed, temperature, case_id, prompt_hash, response, timestamp, template_version))

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            total_entries = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT COUNT(DISTINCT provider || '|' || model) FROM cache_entries
            """)
            unique_models = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT COUNT(DISTINCT case_id) FROM cache_entries
            """)
            unique_cases = cursor.fetchone()[0]

            return {
                "total_entries": total_entries,
                "unique_models": unique_models,
                "unique_cases": unique_cases
            }

    def clear_cache(self):
        """Clear all cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_entries")


def make_prompt_hash(
    facts: Dict[str, Any],
    law_snippets: Dict[str, str],
    question: str,
    template_version: str = "1.0"
) -> str:
    """
    Create a hash of the prompt components for cache keying.

    Args:
        facts: Case facts dictionary
        law_snippets: Law snippets dictionary
        question: Target question
        template_version: Version of the prompt template

    Returns:
        SHA-256 hash of the prompt components
    """
    # Create a deterministic string representation
    prompt_data = {
        "facts": sorted(facts.items()),
        "law_snippets": sorted(law_snippets.items()),
        "question": question,
        "template_version": template_version
    }

    # Convert to JSON string and hash
    prompt_str = json.dumps(prompt_data, sort_keys=True)
    return hashlib.sha256(prompt_str.encode()).hexdigest()


def get_cache_key(
    provider: str,
    model: str,
    seed: Optional[int],
    temperature: float,
    case_id: str,
    prompt_hash: str
) -> str:
    """
    Create a cache key from the parameters.

    Args:
        provider: LLM provider
        model: Model name
        seed: Random seed
        temperature: Sampling temperature
        case_id: Case identifier
        prompt_hash: Hash of the prompt

    Returns:
        Cache key string
    """
    return f"{provider}:{model}:{seed}:{temperature}:{case_id}:{prompt_hash}"


# Global cache instance
_cache_instance: Optional[LLMCache] = None


def get_cache(cache_dir: Optional[Path] = None) -> LLMCache:
    """
    Get the global cache instance.

    Args:
        cache_dir: Optional cache directory override

    Returns:
        Global cache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMCache(cache_dir or Path("out/cache"))
    return _cache_instance


def set_cache(cache: LLMCache):
    """
    Set the global cache instance.

    Args:
        cache: Cache instance to use globally
    """
    global _cache_instance
    _cache_instance = cache
