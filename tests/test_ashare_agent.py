from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ashare_stat_arb.data_pipeline import save_panel
from ashare_stat_arb.panel import DailyPanel
from quant_research_agent.agent.ashare_workflow import run_ashare_diagnostic_agent
from quant_research_agent.integrations.ashare import (
    AshareResearchTools,
    create_ashare_experiment_contract,
)


def _panel(*, start: str = "2022-01-04") -> DailyPanel:
    dates = np.arange(
        np.datetime64(start),
        np.datetime64(start) + np.timedelta64(8, "D"),
        dtype="datetime64[D]",
    )
    rows = dates.size
    columns = 3
    close = 10.0 + np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    member = np.ones((rows, columns), dtype=bool)
    weights = np.full((rows, columns), 1.0 / columns, dtype=np.float64)
    return DailyPanel(
        dates=dates,
        symbols=("000001.XSHE", "000002.XSHE", "600000.XSHG"),
        adjusted_close=close,
        open_price=close - 0.1,
        close_price=close,
        high_limit=close * 1.1,
        low_limit=close * 0.9,
        volume=np.full((rows, columns), 1_000_000.0),
        money=np.full((rows, columns), 10_000_000.0),
        paused=np.zeros((rows, columns), dtype=bool),
        is_st=np.zeros((rows, columns), dtype=bool),
        member=member,
        benchmark_weight=weights,
    )


def _diagnostic(panel: DailyPanel) -> dict[str, object]:
    rank_ic = {
        str(horizon): {
            "mean_rank_ic": value,
            "annualized_icir": 0.1,
        }
        for horizon, value in ((1, 0.004), (5, -0.002), (10, -0.009), (20, -0.002))
    }
    return {
        "label": "a-share-free-data-ou-direction-diagnostic",
        "data_fingerprint": panel.fingerprint(),
        "forward_rank_ic": rank_ic,
        "portfolio_results": {
            "original": {
                "annualized_gross_excess_return": -0.025,
                "annualized_cost_drag": 0.070,
            },
            "reversed": {
                "annualized_gross_excess_return": -0.006,
                "annualized_cost_drag": 0.069,
            },
            "neutral": {
                "annualized_gross_excess_return": 0.001,
                "annualized_cost_drag": 0.001,
            },
        },
        "signal_coverage": {"nonzero_signal_rate": 0.94},
    }


def _write_inputs(directory: Path, panel: DailyPanel) -> tuple[Path, Path]:
    panel_path, _ = save_panel(panel, directory / "panel.npz")
    diagnostic_path = directory / "diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(_diagnostic(panel), indent=2),
        encoding="utf-8",
    )
    return panel_path, diagnostic_path


class FakeAshareTools:
    def invoke(self, name: str) -> dict[str, object]:
        if name == "inspect_ashare_direction_diagnostic":
            return {
                "facts": {
                    "rank_ic": _diagnostic(_panel())["forward_rank_ic"],
                    "original_gross_excess": -0.025,
                    "reversed_gross_excess": -0.006,
                    "neutral_gross_excess": 0.001,
                    "original_cost_drag": 0.070,
                    "neutral_cost_drag": 0.001,
                    "signal_nonzero_rate": 0.94,
                },
                "decisions": {
                    "simple_sign_flip_supported": False,
                    "rank_ic_gate_passed": False,
                    "cost_problem_is_signal_induced": True,
                },
            }
        if name == "run_fixed_residual_ou_mechanism_test":
            return {
                "original": {
                    "annualized_mean": -0.03,
                    "annualized_sharpe": -0.4,
                    "active_day_rate": 0.8,
                    "average_daily_turnover": 0.2,
                },
                "reversed": {
                    "annualized_mean": 0.03,
                    "annualized_sharpe": 0.4,
                    "active_day_rate": 0.8,
                    "average_daily_turnover": 0.2,
                },
                "mechanism_assessment": {
                    "paper_direction_positive": False,
                    "paper_direction_sharpe_above_half": False,
                    "reversal_outperforms_original": True,
                },
            }
        raise KeyError(name)


class AshareAgentTests(unittest.TestCase):
    def test_contract_freezes_data_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel = _panel()
            panel_path, diagnostic_path = _write_inputs(Path(directory), panel)
            contract = create_ashare_experiment_contract(panel_path, diagnostic_path)

            self.assertEqual(contract.data_fingerprint, panel.fingerprint())
            self.assertFalse(contract.parameter_search_allowed)
            self.assertFalse(contract.sealed_holdout_access_allowed)
            self.assertEqual(contract.factor_count, 5)

    def test_contract_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel = _panel()
            panel_path, diagnostic_path = _write_inputs(Path(directory), panel)
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            payload["data_fingerprint"] = "changed"
            diagnostic_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                create_ashare_experiment_contract(panel_path, diagnostic_path)

    def test_contract_rejects_sealed_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel = _panel(start="2023-01-03")
            panel_path, diagnostic_path = _write_inputs(Path(directory), panel)

            with self.assertRaisesRegex(ValueError, "sealed 2023-2025 holdout"):
                create_ashare_experiment_contract(panel_path, diagnostic_path)

    def test_tool_registry_rejects_unknown_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path, diagnostic_path = _write_inputs(Path(directory), _panel())
            contract = create_ashare_experiment_contract(panel_path, diagnostic_path)
            tools = AshareResearchTools(contract)

            with self.assertRaisesRegex(KeyError, "not allowed"):
                tools.invoke("search_parameters")

    def test_agent_completes_bounded_research_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path, diagnostic_path = _write_inputs(Path(directory), _panel())
            result = run_ashare_diagnostic_agent(
                panel_path=panel_path,
                diagnostics_path=diagnostic_path,
                tools=FakeAshareTools(),
            )

            phases = {item["phase"] for item in result.trace}
            self.assertEqual(result.status, "completed")
            self.assertIn("quant_execution", phases)
            self.assertIn("reflection", phases)
            self.assertFalse(result.final_assessment["parameter_search_authorized"])
            self.assertFalse(result.final_assessment["holdout_accessed"])
            self.assertIn("A-Share Quant Research Agent Report", result.report_markdown)


if __name__ == "__main__":
    unittest.main()
