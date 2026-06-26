"""Structured experiment results with JSON + npz persistence.

Provides a unified format for saving and loading experiment results,
including metadata (params, timestamps) and large numerical arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ExperimentResult:
    """Container for experiment results with serialization support.

    Scalar/dict data goes to JSON; large arrays go to .npz.
    Both are saved together under the same base path.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, NDArray] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def save(self, directory: str | Path) -> Path:
        """Save results to directory/name_timestamp/.

        Creates:
            meta.json  — name, params, timestamp, scalar data
            arrays.npz — numpy arrays (if any)

        Returns the output directory path.
        """
        directory = Path(directory)
        slug = self.name.replace(" ", "_").lower()
        ts_short = self.timestamp[:19].replace(":", "").replace("-", "")
        out_dir = directory / f"{slug}_{ts_short}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "timestamp": self.timestamp,
            "params": self.params,
            "data": _serialize(self.data),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        if self.arrays:
            np.savez_compressed(out_dir / "arrays.npz", **self.arrays)

        return out_dir

    @classmethod
    def load(cls, directory: str | Path) -> ExperimentResult:
        """Load results from a directory created by save()."""
        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        arrays = {}
        npz_path = directory / "arrays.npz"
        if npz_path.exists():
            with np.load(npz_path) as npz:
                for key in npz.files:
                    arrays[key] = npz[key]

        return cls(
            name=meta["name"],
            timestamp=meta.get("timestamp", ""),
            params=meta.get("params", {}),
            data=meta.get("data", {}),
            arrays=arrays,
        )

    def summary(self) -> str:
        """One-line summary of the result."""
        n_scalars = len(self.data)
        n_arrays = len(self.arrays)
        return (
            f"ExperimentResult('{self.name}', "
            f"{n_scalars} data fields, {n_arrays} arrays, "
            f"params={self.params})"
        )


def _serialize(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj
