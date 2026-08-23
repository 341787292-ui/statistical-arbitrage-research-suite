from __future__ import annotations

import argparse
from pathlib import Path

from quant_research_agent.agent.ashare_workflow import run_ashare_diagnostic_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded A-share research Agent.")
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--diagnostics",
        default="ashare_stat_arb/output/baostock_pilot100_signal_diagnostics.json",
    )
    parser.add_argument(
        "--report",
        default="reports/ashare_agent_diagnostic.md",
    )
    parser.add_argument(
        "--json",
        default="reports/ashare_agent_diagnostic.json",
    )
    args = parser.parse_args()

    result = run_ashare_diagnostic_agent(
        panel_path=args.panel,
        diagnostics_path=args.diagnostics,
    )
    report = Path(args.report)
    output = Path(args.json)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.report_markdown, encoding="utf-8")
    output.write_text(result.to_json(indent=2), encoding="utf-8")
    print("A-share Quant Research Agent completed.")
    print(f"Verdict: {result.final_assessment['verdict']}")
    print(f"Report: {report.resolve()}")
    print(f"JSON: {output.resolve()}")


if __name__ == "__main__":
    main()
