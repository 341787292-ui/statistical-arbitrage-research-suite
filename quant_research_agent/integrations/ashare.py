from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.residual_audit import audit_residual_continuity
from ashare_stat_arb.residual_comparison import compare_residual_definitions
from ashare_stat_arb.residual_diagnostics import (
    evaluate_residual_positions,
    ou_residual_positions,
)
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


@dataclass(frozen=True)
class AshareExperimentContract:
    panel_path: str
    diagnostics_path: str
    data_fingerprint: str
    latest_allowed_date: str
    factor_count: int
    covariance_window: int
    loading_window: int
    residual_lookback: int
    ou_entry_threshold: float
    ou_min_r_squared: float
    parameter_search_allowed: bool = False
    sealed_holdout_access_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_ashare_experiment_contract(
    panel_path: str | Path,
    diagnostics_path: str | Path,
) -> AshareExperimentContract:
    panel_file = Path(panel_path).resolve()
    diagnostic_file = Path(diagnostics_path).resolve()
    if not panel_file.exists():
        raise FileNotFoundError(f"A-share panel does not exist: {panel_file}")
    if not diagnostic_file.exists():
        raise FileNotFoundError(f"A-share diagnostics do not exist: {diagnostic_file}")

    panel = load_panel(panel_file)
    config = DEFAULT_CONFIG
    last_date = str(panel.dates[-1])
    if panel.dates[-1] >= np.datetime64(config.periods.holdout_start):
        raise ValueError(
            "The Agent cannot access the sealed 2023-2025 holdout during diagnosis."
        )
    diagnostic = json.loads(diagnostic_file.read_text(encoding="utf-8"))
    fingerprint = panel.fingerprint()
    if diagnostic.get("data_fingerprint") != fingerprint:
        raise ValueError("Diagnostic fingerprint does not match the A-share panel.")
    if diagnostic.get("label") != "a-share-free-data-ou-direction-diagnostic":
        raise ValueError("Unexpected diagnostic label; refusing an unaudited input.")

    return AshareExperimentContract(
        panel_path=str(panel_file),
        diagnostics_path=str(diagnostic_file),
        data_fingerprint=fingerprint,
        latest_allowed_date=last_date,
        factor_count=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
        residual_lookback=config.signal.residual_lookback,
        ou_entry_threshold=config.signal.ou_entry_threshold,
        ou_min_r_squared=config.signal.ou_min_r_squared,
    )


class AshareResearchTools:
    """Bounded Agent tools for one frozen A-share diagnostic milestone."""

    def __init__(self, contract: AshareExperimentContract) -> None:
        self.contract = contract
        self._cached_panel: Any | None = None
        self._cached_signal: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return (
            "inspect_ashare_direction_diagnostic",
            "run_fixed_residual_ou_mechanism_test",
            "audit_ashare_residual_continuity",
            "compare_ashare_residual_definitions",
        )

    def invoke(self, name: str) -> dict[str, Any]:
        if name == "inspect_ashare_direction_diagnostic":
            return self.inspect_direction_diagnostic()
        if name == "run_fixed_residual_ou_mechanism_test":
            return self.run_residual_mechanism_test()
        if name == "audit_ashare_residual_continuity":
            return self.audit_residual_continuity()
        if name == "compare_ashare_residual_definitions":
            return self.compare_residual_definitions()
        raise KeyError(f"A-share tool '{name}' is not allowed. Available: {self.names}")

    def inspect_direction_diagnostic(self) -> dict[str, Any]:
        payload = json.loads(Path(self.contract.diagnostics_path).read_text(encoding="utf-8"))
        if payload.get("data_fingerprint") != self.contract.data_fingerprint:
            raise ValueError("A-share diagnostic changed after contract creation.")
        rank_ic = payload["forward_rank_ic"]
        portfolios = payload["portfolio_results"]
        original = portfolios["original"]
        reversed_result = portfolios["reversed"]
        neutral = portfolios["neutral"]
        return {
            "experiment": "ashare_direction_diagnostic_inspection",
            "facts": {
                "rank_ic": rank_ic,
                "original_gross_excess": original["annualized_gross_excess_return"],
                "reversed_gross_excess": reversed_result[
                    "annualized_gross_excess_return"
                ],
                "neutral_gross_excess": neutral["annualized_gross_excess_return"],
                "original_cost_drag": original["annualized_cost_drag"],
                "neutral_cost_drag": neutral["annualized_cost_drag"],
                "signal_nonzero_rate": payload["signal_coverage"]["nonzero_signal_rate"],
            },
            "decisions": {
                "simple_sign_flip_supported": (
                    reversed_result["annualized_gross_excess_return"]
                    > neutral["annualized_gross_excess_return"]
                ),
                "rank_ic_gate_passed": any(
                    item["mean_rank_ic"] >= DEFAULT_CONFIG.admission.minimum_rank_ic
                    for item in rank_ic.values()
                ),
                "cost_problem_is_signal_induced": (
                    original["annualized_cost_drag"]
                    > neutral["annualized_cost_drag"] * 5.0
                ),
            },
        }

    def run_residual_mechanism_test(self) -> dict[str, Any]:
        if self.contract.parameter_search_allowed:
            raise RuntimeError("This tool only supports a frozen no-search experiment.")
        panel, signal = self._panel_and_signal()
        positions = ou_residual_positions(
            signal.residual_returns,
            lookback=self.contract.residual_lookback,
            entry_threshold=self.contract.ou_entry_threshold,
            min_r_squared=self.contract.ou_min_r_squared,
        )
        start = max(
            int(
                np.searchsorted(
                    panel.dates,
                    np.datetime64(DEFAULT_CONFIG.periods.development_start),
                )
            ),
            self.contract.covariance_window + self.contract.residual_lookback - 1,
        )
        original = evaluate_residual_positions(
            signal.residual_returns,
            positions,
            start=start,
        )
        reversed_result = evaluate_residual_positions(
            signal.residual_returns,
            positions,
            start=start,
            direction=-1.0,
        )
        return {
            "experiment": "fixed_residual_ou_mechanism_test",
            "scope": (
                "theoretical residual-space diagnostic; no A-share execution, "
                "parameter search, or holdout access"
            ),
            "contract": self.contract.to_dict(),
            "original": original.to_dict(),
            "reversed": reversed_result.to_dict(),
            "mechanism_assessment": {
                "paper_direction_positive": original.annualized_mean > 0.0,
                "paper_direction_sharpe_above_half": original.annualized_sharpe >= 0.5,
                "reversal_outperforms_original": (
                    reversed_result.annualized_mean > original.annualized_mean
                ),
            },
        }

    def audit_residual_continuity(self) -> dict[str, Any]:
        panel, signal = self._panel_and_signal()
        audit = audit_residual_continuity(
            signal,
            panel.dates,
            panel.member,
            lookback=self.contract.residual_lookback,
        )
        result = audit.to_dict()
        return {
            "experiment": "fixed_pca_residual_continuity_audit",
            "scope": "diagnostic only; no parameter search or holdout access",
            "audit": result,
            "assessment": {
                "all_ou_windows_cross_refits": (
                    audit.ou_candidate_windows > 0
                    and audit.cross_model_ou_window_rate == 1.0
                ),
                "coverage_complete_when_model_runs": (
                    audit.member_residual_coverage >= 0.999
                ),
                "model_day_gap_detected": audit.model_day_rate < 0.99,
                "visible_refit_day_spike": (
                    audit.refit_residual_scale_ratio >= 1.5
                    or audit.refit_alpha_change_ratio >= 1.5
                ),
            },
        }

    def compare_residual_definitions(self) -> dict[str, Any]:
        if self.contract.parameter_search_allowed:
            raise RuntimeError("This tool only supports a frozen no-search comparison.")
        panel, signal = self._panel_and_signal()
        start = max(
            int(
                np.searchsorted(
                    panel.dates,
                    np.datetime64(DEFAULT_CONFIG.periods.development_start),
                )
            ),
            self.contract.covariance_window + self.contract.residual_lookback - 1,
        )
        comparison = compare_residual_definitions(
            panel.adjusted_returns(),
            panel.dates,
            panel.member,
            signal,
            start=start,
            n_factors=self.contract.factor_count,
            covariance_window=self.contract.covariance_window,
            loading_window=self.contract.loading_window,
            residual_lookback=self.contract.residual_lookback,
            entry_threshold=self.contract.ou_entry_threshold,
            min_r_squared=self.contract.ou_min_r_squared,
        )
        result = comparison.to_dict()
        return {
            "experiment": "fixed_residual_definition_comparison",
            "scope": "pre-registered diagnostic; no parameter search or holdout access",
            "comparison": result,
            "assessment": {
                "current_composition_improves_sharpe": comparison.sharpe_delta > 0.0,
                "current_composition_rescues_mechanism": (
                    comparison.current_composition_gate_passed
                    and comparison.sharpe_delta > 0.0
                ),
                "stitched_asof_gate_passed": (
                    comparison.stitched_asof.annualized_sharpe
                    >= comparison.mechanism_sharpe_gate
                ),
            },
        }

    def _panel_and_signal(self) -> tuple[Any, Any]:
        if self._cached_panel is not None and self._cached_signal is not None:
            return self._cached_panel, self._cached_signal
        panel = load_panel(self.contract.panel_path)
        if panel.fingerprint() != self.contract.data_fingerprint:
            raise ValueError("A-share panel changed after contract creation.")
        signal = rolling_monthly_pca_ou_stock_alpha(
            panel.adjusted_returns(),
            panel.dates,
            panel.member,
            n_factors=self.contract.factor_count,
            covariance_window=self.contract.covariance_window,
            loading_window=self.contract.loading_window,
            residual_lookback=self.contract.residual_lookback,
            entry_threshold=self.contract.ou_entry_threshold,
            min_r_squared=self.contract.ou_min_r_squared,
        )
        self._cached_panel = panel
        self._cached_signal = signal
        return panel, signal
