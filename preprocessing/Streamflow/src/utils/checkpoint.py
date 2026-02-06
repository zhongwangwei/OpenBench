"""Checkpoint system with HMAC-signed JSON for pipeline resumability."""

import os
import json
import hmac
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ..constants import PIPELINE_STEPS

logger = logging.getLogger(__name__)


def _get_secret() -> bytes:
    """Get the secret key for HMAC signing from environment or default."""
    return os.environ.get("STREAMFLOW_CHECKPOINT_KEY", "default-dev-key").encode()


def _sign(data: bytes) -> str:
    """Create HMAC-SHA256 signature for data."""
    return hmac.new(_get_secret(), data, hashlib.sha256).hexdigest()


def _verify(data: bytes, signature: str) -> bool:
    """Verify HMAC signature using constant-time comparison."""
    return hmac.compare_digest(_sign(data), signature)


class CheckpointManager:
    """Manages per-step checkpoint files with HMAC integrity verification."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_step(self, step: str, results: Dict[str, Any]) -> None:
        """Save step results as HMAC-signed JSON.

        Args:
            step: Pipeline step name.
            results: Dictionary of step results to persist.
        """
        data_json = json.dumps(results, sort_keys=True, default=str)
        signature = _sign(data_json.encode())
        content = json.dumps({"data": data_json, "signature": signature})
        filepath = self._step_file(step)
        with open(filepath, "w") as f:
            f.write(content)

    def load_step(self, step: str) -> Optional[Dict[str, Any]]:
        """Load step results, verifying HMAC signature.

        Args:
            step: Pipeline step name.

        Returns:
            The results dictionary if valid, or None if missing/invalid.
        """
        filepath = self._step_file(step)
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r") as f:
                parsed = json.loads(f.read())
            data_str = parsed["data"]
            if not _verify(data_str.encode(), parsed["signature"]):
                logger.warning("Checkpoint signature failed for step '%s'", step)
                return None
            return json.loads(data_str)
        except Exception as e:
            logger.warning("Failed to load checkpoint for step '%s': %s", step, e)
            return None

    def get_completed_steps(self) -> Set[str]:
        """Return the set of step names that have valid checkpoint files."""
        completed = set()
        for step in PIPELINE_STEPS:
            if self._step_file(step).exists():
                if self.load_step(step) is not None:
                    completed.add(step)
        return completed

    def get_resume_step(self) -> str:
        """Return the first uncompleted step from PIPELINE_STEPS.

        If all steps are completed, returns the last step.
        """
        completed = self.get_completed_steps()
        for step in PIPELINE_STEPS:
            if step not in completed:
                return step
        return PIPELINE_STEPS[-1]

    def clear(self) -> None:
        """Remove all checkpoint files."""
        for step in PIPELINE_STEPS:
            filepath = self._step_file(step)
            if filepath.exists():
                filepath.unlink()

    def _step_file(self, step: str) -> Path:
        """Get the file path for a step's checkpoint."""
        return self.checkpoint_dir / f"step_{step}.json"


# Backward-compatible stub: will be removed when __init__.py is updated.
class CheckpointData:  # noqa: E302
    """Deprecated: kept only for backward-compatible imports."""
    pass
