from __future__ import annotations

from dataclasses import asdict
import json

from ashare_stat_arb.execution import (
    PortfolioState,
    begin_trading_day,
    execute_target_portfolio,
)


def main() -> None:
    initial = PortfolioState(cash=100_000.0)
    day_one = execute_target_portfolio(
        initial,
        {"000001.SZ": 1_000},
        {"000001.SZ": 10.0},
    )
    same_day_exit = execute_target_portfolio(
        day_one.state,
        {"000001.SZ": 0},
        {"000001.SZ": 10.1},
    )
    day_two = execute_target_portfolio(
        begin_trading_day(same_day_exit.state),
        {"000001.SZ": 0},
        {"000001.SZ": 10.1},
    )
    output = {
        "day_one_buy": [asdict(record) for record in day_one.records],
        "same_day_exit_attempt": [
            asdict(record) for record in same_day_exit.records
        ],
        "day_two_exit": [asdict(record) for record in day_two.records],
        "final_state": asdict(day_two.state),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
