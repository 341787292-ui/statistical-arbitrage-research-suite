from __future__ import annotations

import argparse
from pathlib import Path

from quant_research_agent.pipeline import run_paper_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI Quant Research Agent baseline workflow."
    )
    parser.add_argument(
        "--paper",
        default="samples/stat_arb_note.txt",
        help="Path to a .txt, .md, or .pdf paper/note.",
    )
    parser.add_argument(
        "--query",
        default="Help me reproduce and analyze this statistical arbitrage paper.",
        help="Research request for the agent.",
    )
    parser.add_argument(
        "--report",
        default="reports/paper_research_spec.md",
        help="Where to write the markdown report.",
    )
    parser.add_argument(
        "--spec-json",
        default="reports/paper_research_spec.json",
        help="Where to write the structured JSON output.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable OpenAI API usage and force deterministic local extraction.",
    )
    parser.add_argument(
        "--run-quant",
        action="store_true",
        help="Run the deterministic local statistical arbitrage baseline.",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Run the complete baseline research loop with autonomous validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_paper_pipeline(
        paper_path=Path(args.paper),
        query=args.query,
        use_llm=not args.no_llm,
        run_quant=args.run_quant,
        run_agent=args.run_agent,
    )

    report_path = Path(args.report)
    spec_path = Path(args.spec_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(result.report_markdown, encoding="utf-8")
    spec_path.write_text(result.to_json(indent=2), encoding="utf-8")

    print("AI Quant Research Agent v0.3 completed.")
    print(f"Status: {result.status}")
    print(f"Agent steps: {len(result.agent_trace)}")
    print(f"Validation experiments: {len(result.validation_results)}")
    print(f"Paper: {Path(args.paper).resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"Spec JSON: {spec_path.resolve()}")


if __name__ == "__main__":
    main()
