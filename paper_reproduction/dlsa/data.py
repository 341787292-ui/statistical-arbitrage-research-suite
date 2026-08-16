from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResidualDataAudit:
    shape: tuple[int, int]
    dtype: str
    finite_fraction: float
    zero_fraction: float
    nonzero_std: float


def load_numpy_array(path: str | Path) -> np.ndarray:
    """Load `.npy` or the author's `.npy.gz` residual files."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as compressed:
            payload = compressed.read()
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(source, allow_pickle=False)


def audit_residual_array(values: np.ndarray) -> ResidualDataAudit:
    residuals = np.asarray(values)
    if residuals.ndim != 2:
        raise ValueError("Residual data must have shape (time, assets).")
    finite = np.isfinite(residuals)
    finite_values = residuals[finite]
    nonzero = finite_values[finite_values != 0]
    return ResidualDataAudit(
        shape=(int(residuals.shape[0]), int(residuals.shape[1])),
        dtype=str(residuals.dtype),
        finite_fraction=float(finite.mean()),
        zero_fraction=float(np.mean(finite_values == 0)) if finite_values.size else 1.0,
        nonzero_std=float(np.std(nonzero)) if nonzero.size else 0.0,
    )

