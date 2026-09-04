from __future__ import annotations

from quant_research_agent.verification.ir import (
    DataSlice,
    ExecutionEvent,
    MarketTime,
    ModelFitEvent,
    SelectionProtocol,
    SignalEvent,
    StatArbExperimentIR,
)


def temporal_ir_from_backtest(result: dict) -> StatArbExperimentIR:
    """Translate an instrumented backtest result into StatArb-IR v0.1."""

    timing = result.get("timing_contract")
    events = result.get("execution_events")
    if not isinstance(timing, dict) or not timing:
        raise ValueError("Backtest result has no timing_contract.")
    if not isinstance(events, list) or not events:
        raise ValueError("Backtest result has no execution event to verify.")

    lookback = int(result.get("parameters", {}).get("lookback", 30))
    development_end = int(timing["development_end_session"])
    holdout_start = int(timing["holdout_start_session"])
    holdout_end = int(timing["holdout_end_session"])
    training_start = int(timing["training_start_session"])
    training_end = int(timing["training_end_session"])
    model_fit_session = int(timing["model_fit_session"])

    data_slices: list[DataSlice] = [
        DataSlice(
            data_id="training_returns",
            purpose="fit hedge model",
            window_start=MarketTime(training_start, "close"),
            window_end=MarketTime(training_end, "close"),
            available_at=MarketTime(training_end, "close"),
        ),
        DataSlice(
            data_id="development_returns",
            purpose="development and fixed parameter selection",
            window_start=MarketTime(training_start, "close"),
            window_end=MarketTime(development_end, "close"),
            available_at=MarketTime(development_end, "close"),
        ),
        DataSlice(
            data_id="sealed_holdout",
            purpose="final evaluation only",
            window_start=MarketTime(holdout_start, "open"),
            window_end=MarketTime(holdout_end, "close"),
            available_at=MarketTime(holdout_end, "close"),
        ),
    ]
    signals: list[SignalEvent] = []
    executions: list[ExecutionEvent] = []
    for event in events:
        signal_session = int(event["signal_session"])
        execution_session = int(event["execution_session"])
        feature_data_id = f"signal_residual_history_{signal_session}"
        signal_id = f"residual-signal-{signal_session}"
        data_slices.append(
            DataSlice(
                data_id=feature_data_id,
                purpose="generate an executed residual signal",
                window_start=MarketTime(
                    max(0, signal_session - lookback),
                    "close",
                ),
                window_end=MarketTime(signal_session, "close"),
                available_at=MarketTime(signal_session, "close"),
                derived_from=("training_returns",),
            )
        )
        signals.append(
            SignalEvent(
                signal_id=signal_id,
                model_id="hedge-model-v1",
                feature_data_id=feature_data_id,
                generated_at=MarketTime(
                    signal_session,
                    str(timing["signal_generation_phase"]),
                ),
            )
        )
        executions.append(
            ExecutionEvent(
                order_id=f"rebalance-{execution_session}",
                signal_id=signal_id,
                executed_at=MarketTime(
                    execution_session,
                    str(timing["execution_phase"]),
                ),
                return_window_start=MarketTime(
                    execution_session,
                    str(timing["return_window_start_phase"]),
                ),
            )
        )

    return StatArbExperimentIR(
        experiment_id="instrumented-next-open-pair-baseline",
        market="US_EQUITY_SYNTHETIC",
        data_slices=tuple(data_slices),
        model_fits=(
            ModelFitEvent(
                model_id="hedge-model-v1",
                training_data_id="training_returns",
                fitted_at=MarketTime(model_fit_session, "before_open"),
            ),
        ),
        signals=tuple(signals),
        executions=tuple(executions),
        selection=SelectionProtocol(
            development_data_ids=("training_returns", "development_returns"),
            selection_completed_at=MarketTime(development_end, "after_close"),
            holdout_data_id="sealed_holdout",
            holdout_accessed_during_selection=bool(
                timing["holdout_accessed_during_selection"]
            ),
        ),
        metadata={
            "source": "instrumented deterministic backtest",
            "signal_to_execution_policy": "close_t_to_open_t_plus_1",
            "verified_event_count": len(events),
        },
    )
