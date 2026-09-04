from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from quant_research_agent.verification.ir import (
    DataSlice,
    ExecutionEvent,
    MarketTime,
    ModelFitEvent,
    SelectionProtocol,
    SignalEvent,
    StatArbExperimentIR,
)


@dataclass(frozen=True)
class MutationCase:
    mutation_id: str
    description: str
    expected_rule_id: str
    apply: Callable[[StatArbExperimentIR], StatArbExperimentIR]


def build_reference_temporal_ir() -> StatArbExperimentIR:
    """Create a small valid daily stat-arb experiment contract."""

    return StatArbExperimentIR(
        experiment_id="daily-pair-reference",
        market="US_EQUITY",
        data_slices=(
            DataSlice(
                data_id="training_returns",
                purpose="fit hedge model",
                window_start=MarketTime(0, "close"),
                window_end=MarketTime(59, "close"),
                available_at=MarketTime(59, "close"),
            ),
            DataSlice(
                data_id="residual_history",
                purpose="generate residual z-score signal",
                window_start=MarketTime(60, "close"),
                window_end=MarketTime(89, "close"),
                available_at=MarketTime(89, "close"),
                derived_from=("training_returns",),
            ),
            DataSlice(
                data_id="development_returns",
                purpose="model and parameter selection",
                window_start=MarketTime(60, "close"),
                window_end=MarketTime(149, "close"),
                available_at=MarketTime(149, "close"),
            ),
            DataSlice(
                data_id="sealed_holdout",
                purpose="final evaluation only",
                window_start=MarketTime(150, "open"),
                window_end=MarketTime(199, "close"),
                available_at=MarketTime(199, "close"),
            ),
        ),
        model_fits=(
            ModelFitEvent(
                model_id="hedge-model-v1",
                training_data_id="training_returns",
                fitted_at=MarketTime(60, "before_open"),
            ),
        ),
        signals=(
            SignalEvent(
                signal_id="residual-signal-89",
                model_id="hedge-model-v1",
                feature_data_id="residual_history",
                generated_at=MarketTime(89, "after_close"),
            ),
        ),
        executions=(
            ExecutionEvent(
                order_id="rebalance-90",
                signal_id="residual-signal-89",
                executed_at=MarketTime(90, "open"),
                return_window_start=MarketTime(90, "open"),
            ),
        ),
        selection=SelectionProtocol(
            development_data_ids=("training_returns", "development_returns"),
            selection_completed_at=MarketTime(149, "after_close"),
            holdout_data_id="sealed_holdout",
            holdout_accessed_during_selection=False,
        ),
        metadata={
            "frequency": "daily",
            "signal_to_execution_policy": "close_t_to_open_t_plus_1",
        },
    )


def mutation_cases() -> tuple[MutationCase, ...]:
    return (
        MutationCase(
            mutation_id="future_training_data",
            description="Fit the hedge model before its training data is available.",
            expected_rule_id="TEMP-002",
            apply=_future_training_data,
        ),
        MutationCase(
            mutation_id="model_after_signal",
            description="Generate a signal before the referenced model has been fitted.",
            expected_rule_id="TEMP-003",
            apply=_model_after_signal,
        ),
        MutationCase(
            mutation_id="future_signal_feature",
            description="Use a residual observation from the next session in today's signal.",
            expected_rule_id="TEMP-004",
            apply=_future_signal_feature,
        ),
        MutationCase(
            mutation_id="same_close_execution",
            description="Use the closing price before an after-close signal exists.",
            expected_rule_id="TEMP-005",
            apply=_same_close_execution,
        ),
        MutationCase(
            mutation_id="pre_execution_return",
            description="Credit the strategy with returns earned before order execution.",
            expected_rule_id="TEMP-006",
            apply=_pre_execution_return,
        ),
        MutationCase(
            mutation_id="holdout_peek",
            description="Read the sealed holdout while selecting the method.",
            expected_rule_id="TEMP-008",
            apply=_holdout_peek,
        ),
        MutationCase(
            mutation_id="selection_overlaps_holdout",
            description="Continue model selection after the holdout period starts.",
            expected_rule_id="TEMP-009",
            apply=_selection_overlaps_holdout,
        ),
    )


def apply_named_mutation(
    ir: StatArbExperimentIR,
    mutation_id: str,
) -> tuple[StatArbExperimentIR, MutationCase]:
    for mutation in mutation_cases():
        if mutation.mutation_id == mutation_id:
            return mutation.apply(ir), mutation
    available = ", ".join(item.mutation_id for item in mutation_cases())
    raise ValueError(f"Unknown mutation '{mutation_id}'. Available: {available}")


def _future_training_data(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    fit_time = ir.model_fits[0].fitted_at
    return _replace_data(
        ir,
        ir.model_fits[0].training_data_id,
        available_at=MarketTime(fit_time.session, "close"),
    )


def _model_after_signal(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    signal_session = ir.signals[0].generated_at.session
    changed = replace(
        ir.model_fits[0],
        fitted_at=MarketTime(signal_session + 1, "before_open"),
    )
    return replace(ir, model_fits=(changed,))


def _future_signal_feature(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    future_session = ir.signals[0].generated_at.session + 1
    return _replace_data(
        ir,
        ir.signals[0].feature_data_id,
        window_end=MarketTime(future_session, "close"),
        available_at=MarketTime(future_session, "close"),
    )


def _same_close_execution(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    signal_session = ir.signals[0].generated_at.session
    changed = replace(
        ir.executions[0],
        executed_at=MarketTime(signal_session, "close"),
    )
    return replace(ir, executions=(changed,))


def _pre_execution_return(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    execution_session = ir.executions[0].executed_at.session
    changed = replace(
        ir.executions[0],
        return_window_start=MarketTime(execution_session - 1, "close"),
    )
    return replace(ir, executions=(changed,))


def _holdout_peek(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    changed = replace(
        ir.selection,
        development_data_ids=(*ir.selection.development_data_ids, "sealed_holdout"),
        holdout_accessed_during_selection=True,
    )
    return replace(ir, selection=changed)


def _selection_overlaps_holdout(ir: StatArbExperimentIR) -> StatArbExperimentIR:
    holdout_id = ir.selection.holdout_data_id
    holdout = next(item for item in ir.data_slices if item.data_id == holdout_id)
    changed = replace(
        ir.selection,
        selection_completed_at=MarketTime(
            holdout.window_start.session + 10,
            "after_close",
        ),
    )
    return replace(ir, selection=changed)


def _replace_data(
    ir: StatArbExperimentIR,
    data_id: str,
    **changes,
) -> StatArbExperimentIR:
    data_slices = tuple(
        replace(item, **changes) if item.data_id == data_id else item
        for item in ir.data_slices
    )
    return replace(ir, data_slices=data_slices)
