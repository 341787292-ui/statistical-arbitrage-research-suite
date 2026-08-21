"""Paper-first reference implementation for Deep Learning Statistical Arbitrage."""

from paper_reproduction.dlsa.config import PaperConfig
from paper_reproduction.dlsa.factor_models import PCAResidualResult, rolling_pca_residuals
from paper_reproduction.dlsa.ou import OUFit, fit_ou, ou_threshold_weight

__all__ = [
    "OUFit",
    "PCAResidualResult",
    "PaperConfig",
    "fit_ou",
    "ou_threshold_weight",
    "rolling_pca_residuals",
]

