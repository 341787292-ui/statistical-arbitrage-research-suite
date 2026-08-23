from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.residual_audit import audit_residual_continuity
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A-share PCA residual continuity.")
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_residual_audit.json",
    )
    args = parser.parse_args()

    panel = load_panel(args.panel)
    config = DEFAULT_CONFIG
    signal = rolling_monthly_pca_ou_stock_alpha(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
        residual_lookback=config.signal.residual_lookback,
        entry_threshold=config.signal.ou_entry_threshold,
        min_r_squared=config.signal.ou_min_r_squared,
    )
    audit = audit_residual_continuity(
        signal,
        panel.dates,
        panel.member,
        lookback=config.signal.residual_lookback,
    )
    payload = {
        "label": "a-share-free-data-pca-residual-continuity-audit",
        "result_scope": "diagnostic only; no parameters selected from this result",
        "data_fingerprint": panel.fingerprint(),
        "audit": audit.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
