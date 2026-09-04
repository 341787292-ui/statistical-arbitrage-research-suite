from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from quant_research_agent.verification.ir import (
    DataSlice,
    StatArbExperimentIR,
    is_at_or_before,
    is_before,
)


TEMPORAL_RULES = (
    "TEMP-001",
    "TEMP-002",
    "TEMP-003",
    "TEMP-004",
    "TEMP-005",
    "TEMP-006",
    "TEMP-007",
    "TEMP-008",
    "TEMP-009",
)


@dataclass(frozen=True)
class VerificationFinding:
    rule_id: str
    severity: str
    message: str
    counterexample: str
    repair: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    checked_rules: tuple[str, ...]
    findings: tuple[VerificationFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked_rules": list(self.checked_rules),
            "findings": [item.to_dict() for item in self.findings],
        }


def verify_temporal_causality(ir: StatArbExperimentIR) -> VerificationReport:
    """Check whether every decision uses information available at that time."""

    findings: list[VerificationFinding] = []
    data_by_id = _index_unique(
        ir.data_slices,
        id_field="data_id",
        object_name="data slice",
        findings=findings,
    )
    models_by_id = _index_unique(
        ir.model_fits,
        id_field="model_id",
        object_name="model fit",
        findings=findings,
    )
    signals_by_id = _index_unique(
        ir.signals,
        id_field="signal_id",
        object_name="signal",
        findings=findings,
    )

    for data in ir.data_slices:
        if not is_at_or_before(data.window_start, data.window_end):
            findings.append(
                _finding(
                    "TEMP-001",
                    f"Data window '{data.data_id}' ends before it starts.",
                    f"{data.window_start.label()} -> {data.window_end.label()}",
                    "Correct the declared data window boundaries.",
                )
            )
        if not is_at_or_before(data.window_end, data.available_at):
            findings.append(
                _finding(
                    "TEMP-001",
                    f"Data '{data.data_id}' is declared available before its last observation exists.",
                    f"window ends {data.window_end.label()}, available {data.available_at.label()}",
                    "Move data availability to the observation time or later.",
                )
            )
        for parent_id in data.derived_from:
            if parent_id not in data_by_id:
                findings.append(
                    _missing_reference(
                        owner=f"data slice '{data.data_id}'",
                        field="derived_from",
                        missing_id=parent_id,
                    )
                )

    for model in ir.model_fits:
        training = data_by_id.get(model.training_data_id)
        if training is None:
            findings.append(
                _missing_reference(
                    owner=f"model '{model.model_id}'",
                    field="training_data_id",
                    missing_id=model.training_data_id,
                )
            )
            continue
        if not is_at_or_before(training.available_at, model.fitted_at):
            findings.append(
                _finding(
                    "TEMP-002",
                    f"Model '{model.model_id}' is fitted before its training data is available.",
                    f"data available {training.available_at.label()}, fit {model.fitted_at.label()}",
                    "Delay model fitting or shorten the training window.",
                )
            )

    for signal in ir.signals:
        model = models_by_id.get(signal.model_id)
        features = data_by_id.get(signal.feature_data_id)
        if model is None:
            findings.append(
                _missing_reference(
                    owner=f"signal '{signal.signal_id}'",
                    field="model_id",
                    missing_id=signal.model_id,
                )
            )
        elif not is_before(model.fitted_at, signal.generated_at):
            findings.append(
                _finding(
                    "TEMP-003",
                    f"Signal '{signal.signal_id}' is generated before its model is fitted.",
                    f"fit {model.fitted_at.label()}, signal {signal.generated_at.label()}",
                    "Fit and freeze the model before generating this signal.",
                )
            )
        if features is None:
            findings.append(
                _missing_reference(
                    owner=f"signal '{signal.signal_id}'",
                    field="feature_data_id",
                    missing_id=signal.feature_data_id,
                )
            )
        elif not is_at_or_before(features.available_at, signal.generated_at):
            findings.append(
                _finding(
                    "TEMP-004",
                    f"Signal '{signal.signal_id}' uses features that were not yet available.",
                    f"features available {features.available_at.label()}, signal {signal.generated_at.label()}",
                    "Lag the feature window or generate the signal later.",
                )
            )

    for execution in ir.executions:
        signal = signals_by_id.get(execution.signal_id)
        if signal is None:
            findings.append(
                _missing_reference(
                    owner=f"order '{execution.order_id}'",
                    field="signal_id",
                    missing_id=execution.signal_id,
                )
            )
        elif not is_before(signal.generated_at, execution.executed_at):
            findings.append(
                _finding(
                    "TEMP-005",
                    f"Order '{execution.order_id}' executes before its signal exists.",
                    f"signal {signal.generated_at.label()}, execution {execution.executed_at.label()}",
                    "Move execution to the next feasible market event.",
                )
            )
        if not is_at_or_before(execution.executed_at, execution.return_window_start):
            findings.append(
                _finding(
                    "TEMP-006",
                    f"Order '{execution.order_id}' receives returns from before execution.",
                    (
                        f"execution {execution.executed_at.label()}, return window starts "
                        f"{execution.return_window_start.label()}"
                    ),
                    "Start performance attribution at or after the execution time.",
                )
            )

    selection = ir.selection
    for data_id in selection.development_data_ids:
        data = data_by_id.get(data_id)
        if data is None:
            findings.append(
                _missing_reference(
                    owner="selection protocol",
                    field="development_data_ids",
                    missing_id=data_id,
                )
            )
        elif not is_at_or_before(data.available_at, selection.selection_completed_at):
            findings.append(
                _finding(
                    "TEMP-007",
                    f"Selection uses '{data_id}' before that data is available.",
                    (
                        f"data available {data.available_at.label()}, selection completed "
                        f"{selection.selection_completed_at.label()}"
                    ),
                    "Move selection later or remove the unavailable data.",
                )
            )

    holdout = _holdout_slice(ir, data_by_id, findings)
    if selection.holdout_accessed_during_selection or (
        selection.holdout_data_id
        and selection.holdout_data_id in selection.development_data_ids
    ):
        findings.append(
            _finding(
                "TEMP-008",
                "The holdout is used during model or parameter selection.",
                (
                    f"holdout={selection.holdout_data_id}, "
                    f"accessed={selection.holdout_accessed_during_selection}, "
                    f"development_ids={list(selection.development_data_ids)}"
                ),
                "Seal the holdout until every model and parameter choice is frozen.",
            )
        )
    if holdout is not None and not is_before(
        selection.selection_completed_at,
        holdout.window_start,
    ):
        findings.append(
            _finding(
                "TEMP-009",
                "Model or parameter selection overlaps the declared holdout period.",
                (
                    f"selection completed {selection.selection_completed_at.label()}, "
                    f"holdout starts {holdout.window_start.label()}"
                ),
                "End and freeze selection before the first holdout observation.",
            )
        )

    return VerificationReport(
        passed=not findings,
        checked_rules=("IR-001", "IR-002", *TEMPORAL_RULES),
        findings=tuple(findings),
    )


def _index_unique(
    items: tuple[Any, ...],
    *,
    id_field: str,
    object_name: str,
    findings: list[VerificationFinding],
) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for item in items:
        item_id = str(getattr(item, id_field))
        if item_id in index:
            findings.append(
                _finding(
                    "IR-001",
                    f"Duplicate {object_name} identifier '{item_id}'.",
                    f"At least two {object_name} objects use '{item_id}'.",
                    f"Give every {object_name} a unique identifier.",
                )
            )
        else:
            index[item_id] = item
    return index


def _holdout_slice(
    ir: StatArbExperimentIR,
    data_by_id: dict[str, DataSlice],
    findings: list[VerificationFinding],
) -> DataSlice | None:
    holdout_id = ir.selection.holdout_data_id
    if holdout_id is None:
        return None
    holdout = data_by_id.get(holdout_id)
    if holdout is None:
        findings.append(
            _missing_reference(
                owner="selection protocol",
                field="holdout_data_id",
                missing_id=holdout_id,
            )
        )
    return holdout


def _missing_reference(*, owner: str, field: str, missing_id: str) -> VerificationFinding:
    return _finding(
        "IR-002",
        f"{owner} references unknown {field} '{missing_id}'.",
        f"No object with identifier '{missing_id}' exists in the experiment IR.",
        "Add the referenced object or correct the identifier.",
    )


def _finding(rule_id: str, message: str, counterexample: str, repair: str) -> VerificationFinding:
    return VerificationFinding(
        rule_id=rule_id,
        severity="error",
        message=message,
        counterexample=counterexample,
        repair=repair,
    )
