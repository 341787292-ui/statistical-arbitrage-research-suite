from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PHASE_ORDER = {
    "before_open": 0,
    "open": 1,
    "close": 2,
    "after_close": 3,
}


@dataclass(frozen=True)
class MarketTime:
    """A comparable point in an ordered sequence of trading sessions."""

    session: int
    phase: str

    def __post_init__(self) -> None:
        if self.session < 0:
            raise ValueError("session must be non-negative")
        if self.phase not in PHASE_ORDER:
            allowed = ", ".join(PHASE_ORDER)
            raise ValueError(f"Unknown market phase '{self.phase}'. Expected: {allowed}")

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.session, PHASE_ORDER[self.phase])

    def label(self) -> str:
        return f"session {self.session} {self.phase}"


@dataclass(frozen=True)
class DataSlice:
    """A data window and the earliest time at which it is usable."""

    data_id: str
    purpose: str
    window_start: MarketTime
    window_end: MarketTime
    available_at: MarketTime
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFitEvent:
    model_id: str
    training_data_id: str
    fitted_at: MarketTime


@dataclass(frozen=True)
class SignalEvent:
    signal_id: str
    model_id: str
    feature_data_id: str
    generated_at: MarketTime


@dataclass(frozen=True)
class ExecutionEvent:
    order_id: str
    signal_id: str
    executed_at: MarketTime
    return_window_start: MarketTime


@dataclass(frozen=True)
class SelectionProtocol:
    development_data_ids: tuple[str, ...]
    selection_completed_at: MarketTime
    holdout_data_id: str | None = None
    holdout_accessed_during_selection: bool = False


@dataclass(frozen=True)
class StatArbExperimentIR:
    """Typed intermediate representation for one stat-arb experiment.

    The first version deliberately models only temporal provenance. Market
    execution rules and residual-construction semantics are later extensions.
    """

    experiment_id: str
    market: str
    data_slices: tuple[DataSlice, ...]
    model_fits: tuple[ModelFitEvent, ...]
    signals: tuple[SignalEvent, ...]
    executions: tuple[ExecutionEvent, ...]
    selection: SelectionProtocol
    schema_version: str = "statarb-ir/0.1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def is_before(left: MarketTime, right: MarketTime) -> bool:
    return left.sort_key < right.sort_key


def is_at_or_before(left: MarketTime, right: MarketTime) -> bool:
    return left.sort_key <= right.sort_key


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
