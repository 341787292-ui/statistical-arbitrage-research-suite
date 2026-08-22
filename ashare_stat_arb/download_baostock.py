from __future__ import annotations

import argparse
import json

from ashare_stat_arb.baostock import BaoStockProvider
from ashare_stat_arb.baostock_pipeline import build_csi500_panel_baostock
from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import save_panel
from ashare_stat_arb.panel import audit_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the free BaoStock CSI 500 panel.")
    parser.add_argument("--start", default=DEFAULT_CONFIG.periods.raw_start)
    parser.add_argument("--end", default=DEFAULT_CONFIG.periods.raw_end)
    parser.add_argument("--output", default="ashare_stat_arb/data/baostock_csi500_panel.npz")
    parser.add_argument("--cache", default="ashare_stat_arb/data/baostock_cache")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help=(
            "Use a deterministic point-in-time subset per month for a bounded "
            "real-data engineering pilot."
        ),
    )
    args = parser.parse_args()

    def progress(done: int, total: int, symbol: str) -> None:
        if done == 1 or done == total or done % 25 == 0:
            print(f"BaoStock history: {done}/{total} ({symbol})", flush=True)

    with BaoStockProvider() as provider:
        panel, metadata = build_csi500_panel_baostock(
            provider,
            start_date=args.start,
            end_date=args.end,
            cache_directory=args.cache,
            max_symbols=args.max_symbols,
            progress=progress,
        )
    panel_path, manifest_path = save_panel(panel, args.output, metadata=metadata)
    print(
        json.dumps(
            {
                "panel": str(panel_path),
                "manifest": str(manifest_path),
                "metadata": metadata,
                "audit": audit_panel(panel).to_dict(),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
