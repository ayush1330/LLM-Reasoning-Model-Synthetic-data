"""
Run manifest system for tracking evaluation runs.

Creates detailed manifests of each run including metadata, configuration,
and results for reproducibility and analysis.
"""

import datetime
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunManifest:
    """Manifest for a single evaluation run."""
    run_id: str
    timestamp: str
    provider: str
    model: str
    temperature: float
    seed: Optional[int]
    template_version: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    python_version: str
    command_line: List[str]
    case_ids: List[str]
    cache_enabled: bool
    cache_dir: Optional[str]
    cache_stats: Optional[Dict[str, int]]
    total_cases: int
    successful_cases: int
    failed_cases: int
    average_score: Optional[float]
    std_dev: Optional[float]
    pass_rate: Optional[float]


class ManifestManager:
    """Manages run manifests and metadata."""

    def __init__(self, output_dir: Path = Path("out")):
        """
        Initialize manifest manager.

        Args:
            output_dir: Directory to store manifests
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[RunManifest] = None

    def start_run(
        self,
        provider: str,
        model: str,
        temperature: float,
        seed: Optional[int],
        template_version: str = "1.0",
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Start a new run and return run ID.

        Args:
            provider: LLM provider
            model: Model name
            temperature: Sampling temperature
            seed: Random seed
            template_version: Prompt template version
            cache_enabled: Whether caching is enabled
            cache_dir: Cache directory path

        Returns:
            Run ID for this execution
        """
        run_id = self._generate_run_id()
        timestamp = datetime.datetime.now().isoformat()

        # Get git information
        git_commit, git_branch = self._get_git_info()

        # Get command line
        command_line = sys.argv.copy()

        self.current_run = RunManifest(
            run_id=run_id,
            timestamp=timestamp,
            provider=provider,
            model=model,
            temperature=temperature,
            seed=seed,
            template_version=template_version,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=sys.version,
            command_line=command_line,
            case_ids=[],
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            cache_stats=None,
            total_cases=0,
            successful_cases=0,
            failed_cases=0,
            average_score=None,
            std_dev=None,
            pass_rate=None
        )

        return run_id

    def add_case(self, case_id: str):
        """
        Add a case to the current run.

        Args:
            case_id: Case identifier
        """
        if self.current_run:
            self.current_run.case_ids.append(case_id)

    def update_cache_stats(self, cache_stats: Dict[str, int]):
        """
        Update cache statistics for the current run.

        Args:
            cache_stats: Cache statistics dictionary
        """
        if self.current_run:
            self.current_run.cache_stats = cache_stats

    def finish_run(
        self,
        total_cases: int,
        successful_cases: int,
        failed_cases: int,
        average_score: Optional[float] = None,
        std_dev: Optional[float] = None,
        pass_rate: Optional[float] = None
    ):
        """
        Finish the current run and save manifest.

        Args:
            total_cases: Total number of cases processed
            successful_cases: Number of successful cases
            failed_cases: Number of failed cases
            average_score: Average score across all cases
            std_dev: Standard deviation of scores
            pass_rate: Pass rate percentage
        """
        if not self.current_run:
            return

        # Update run statistics
        self.current_run.total_cases = total_cases
        self.current_run.successful_cases = successful_cases
        self.current_run.failed_cases = failed_cases
        self.current_run.average_score = average_score
        self.current_run.std_dev = std_dev
        self.current_run.pass_rate = pass_rate

        # Save manifest
        manifest_path = self.output_dir / "run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(self.current_run), f, indent=2)

        print(f"Run manifest saved to: {manifest_path}")

        # Reset current run
        self.current_run = None

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _get_git_info(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get git commit hash and branch.

        Returns:
            Tuple of (commit_hash, branch_name) or (None, None) if not available
        """
        try:
            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = commit_result.stdout.strip()[:8]  # Short hash

            # Get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            branch_name = branch_result.stdout.strip()

            return commit_hash, branch_name
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, None


# Global manifest manager instance
_manifest_manager: Optional[ManifestManager] = None


def get_manifest_manager(output_dir: Optional[Path] = None) -> ManifestManager:
    """
    Get the global manifest manager instance.

    Args:
        output_dir: Optional output directory override

    Returns:
        Global manifest manager instance
    """
    global _manifest_manager
    if _manifest_manager is None:
        _manifest_manager = ManifestManager(output_dir or Path("out"))
    return _manifest_manager


def set_manifest_manager(manager: ManifestManager):
    """
    Set the global manifest manager instance.

    Args:
        manager: Manifest manager instance to use globally
    """
    global _manifest_manager
    _manifest_manager = manager
