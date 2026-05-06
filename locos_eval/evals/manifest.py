"""Per-experiment manifest for tracking runs on HuggingFace.

Each experiment directory ({task}/{model_slug}/{variant}/) gets its own
manifest.json that tracks all runs (timestamps, metrics, files).
This avoids concurrent write conflicts when parallel pods finish.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentManifest:
    """Manifest tracking all runs for a single experiment configuration."""

    def __init__(
        self,
        experiment_key: str,
        task: str,
        model: str,
        decoding: str,
        variant: str,
    ) -> None:
        self.experiment_key = experiment_key
        self.task = task
        self.model = model
        self.decoding = decoding
        self.variant = variant
        self.runs: list[dict[str, Any]] = []
        self.status: str = "empty"
        self.latest_timestamp: str | None = None

    def add_run(
        self,
        timestamp: str,
        n_samples: int,
        limit: int | None,
        metrics: dict[str, float],
        config_hash: str,
        files: list[str],
    ) -> bool:
        """Add a run entry. Returns True if added, False if timestamp already exists."""
        if self.has_timestamp(timestamp):
            return False

        run: dict[str, Any] = {
            "timestamp": timestamp,
            "n_samples": n_samples,
            "limit": limit,
            "metrics": metrics,
            "config_hash": config_hash,
            "files": files,
        }

        # Flag duplicate config (same hash, different timestamp)
        existing_hashes = {r["config_hash"] for r in self.runs}
        if config_hash in existing_hashes:
            run["duplicate_config"] = True

        self.runs.append(run)
        self.status = "complete"
        if self.latest_timestamp is None or timestamp > self.latest_timestamp:
            self.latest_timestamp = timestamp
        return True

    def has_timestamp(self, timestamp: str) -> bool:
        """Check if a run with this timestamp already exists."""
        return any(r["timestamp"] == timestamp for r in self.runs)

    @property
    def is_complete(self) -> bool:
        """True if at least one scored run exists."""
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_key": self.experiment_key,
            "task": self.task,
            "model": self.model,
            "decoding": self.decoding,
            "variant": self.variant,
            "runs": self.runs,
            "status": self.status,
            "latest_timestamp": self.latest_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        m = cls(
            experiment_key=data["experiment_key"],
            task=data["task"],
            model=data["model"],
            decoding=data["decoding"],
            variant=data["variant"],
        )
        m.runs = data.get("runs", [])
        m.status = data.get("status", "empty")
        m.latest_timestamp = data.get("latest_timestamp")
        return m

    def save(self, path: Path) -> None:
        """Write manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ExperimentManifest:
        """Load manifest from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
