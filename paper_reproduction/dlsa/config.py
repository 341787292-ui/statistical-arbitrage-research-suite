from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    """Core settings used by the paper's main specification."""

    pca_covariance_window: int = 252
    loading_window: int = 60
    n_factors: int = 5
    signal_lookback: int = 30
    ou_entry_threshold: float = 1.25
    ou_min_r2: float = 0.25
    annualization: int = 252

    def validate(self) -> None:
        if self.n_factors < 0:
            raise ValueError("n_factors must be non-negative.")
        if self.loading_window < 2:
            raise ValueError("loading_window must be at least 2.")
        if self.pca_covariance_window < self.loading_window:
            raise ValueError("PCA covariance window must cover the loading window.")
        if self.signal_lookback < 3:
            raise ValueError("signal_lookback must be at least 3.")
        if self.ou_entry_threshold <= 0:
            raise ValueError("OU entry threshold must be positive.")
        if not 0 <= self.ou_min_r2 <= 1:
            raise ValueError("OU minimum R-squared must lie in [0, 1].")

