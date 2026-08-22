from __future__ import annotations

import argparse
import json

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import build_csi500_panel, save_panel
from ashare_stat_arb.jqdata import JQDataProvider
from ashare_stat_arb.panel import audit_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the local CSI 500 research panel.")
    parser.add_argument("--start", default=DEFAULT_CONFIG.periods.raw_start)
    parser.add_argument("--end", default=DEFAULT_CONFIG.periods.raw_end)
    parser.add_argument("--output", default="ashare_stat_arb/data/csi500_panel.npz")
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    provider = JQDataProvider()
    try:
        provider.authenticate()
    except RuntimeError as exc:
        parser.exit(2, f"JQData setup required: {exc}\n")
    panel = build_csi500_panel(
        provider,
        start_date=args.start,
        end_date=args.end,
        index_symbol=DEFAULT_CONFIG.portfolio.benchmark,
        symbol_chunk_size=args.chunk_size,
    )
    panel_path, manifest_path = save_panel(panel, args.output)
    payload = {
        "panel": str(panel_path),
        "manifest": str(manifest_path),
        "audit": audit_panel(panel).to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
