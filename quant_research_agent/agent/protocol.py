from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from quant_research_agent.methodology import FOUNDATION_BY_ID


@dataclass(frozen=True)
class ProtocolCheck:
    check_id: str
    foundation_ids: tuple[str, ...]
    passed: bool
    evidence: str
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["foundation_ids"] = list(self.foundation_ids)
        return payload


@dataclass(frozen=True)
class ProtocolAudit:
    passed: bool
    checks: tuple[ProtocolCheck, ...]
    critical_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [item.to_dict() for item in self.checks],
            "critical_failures": list(self.critical_failures),
        }


def audit_research_run(
    *,
    trace: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    final_assessment: dict[str, Any],
    experiment_result: dict[str, Any] | None,
    validation_results: list[dict[str, Any]],
    retrieved_evidence: list[Any] | None = None,
    retrieved_evidence_count: int = 0,
    require_retrieval: bool = False,
) -> ProtocolAudit:
    """Audit whether a run follows the paper-backed research protocol.

    The audit evaluates observable workflow artifacts. It deliberately does not
    inspect or expose private chain-of-thought.
    """

    phases = [str(event.get("phase", "")) for event in trace]
    checks: list[ProtocolCheck] = []

    invalid_method_ids = sorted(
        {
            method_id
            for event in trace
            for method_id in event.get("method_ids", [])
            if method_id not in FOUNDATION_BY_ID
        }
    )
    missing_method_phases = [
        str(event.get("phase", "unresolved"))
        for event in trace
        if not event.get("method_ids")
    ]
    checks.append(
        ProtocolCheck(
            check_id="method_provenance",
            foundation_ids=("react-2023",),
            passed=not invalid_method_ids and not missing_method_phases,
            evidence=(
                f"All {len(trace)} trace events name valid method foundations."
                if not invalid_method_ids and not missing_method_phases
                else (
                    f"Invalid IDs={invalid_method_ids}; phases without method IDs="
                    f"{missing_method_phases}."
                )
            ),
        )
    )

    evidence_items = retrieved_evidence or []
    evidence_count = len(evidence_items) if retrieved_evidence is not None else retrieved_evidence_count
    grounded_evidence_count = sum(
        1
        for item in evidence_items
        if _evidence_field(item, "claim")
        and _evidence_field(item, "source")
        and _evidence_field(item, "source_chunk_id")
    )
    evidence_is_auditable = (
        grounded_evidence_count == evidence_count
        if retrieved_evidence is not None
        else evidence_count > 0
    )
    retrieval_ok = not require_retrieval or (
        "retrieval" in phases and evidence_count > 0 and evidence_is_auditable
    )
    checks.append(
        ProtocolCheck(
            check_id="retrieval_grounding",
            foundation_ids=("rag-2020", "self-rag-2024"),
            passed=retrieval_ok,
            evidence=(
                (
                    f"The run retained {evidence_count} evidence items; "
                    f"{grounded_evidence_count} have a claim, source, and chunk ID."
                )
                if require_retrieval and retrieved_evidence is not None
                else f"The run retained {evidence_count} retrieved evidence items."
                if require_retrieval
                else "This diagnostic run grounds claims in frozen tool outputs; paper retrieval is not required."
            ),
        )
    )

    reasoning_phase = _first_index(phases, ("planning", "hypothesis_generation"))
    action_phase = _first_index(phases, ("quant_execution",))
    reflection_phase = _first_index(phases, ("reflection",))
    loop_ok = (
        reasoning_phase is not None
        and action_phase is not None
        and reflection_phase is not None
        and reasoning_phase < action_phase < reflection_phase
    )
    checks.append(
        ProtocolCheck(
            check_id="reason_action_observation_loop",
            foundation_ids=("react-2023",),
            passed=loop_ok,
            evidence=(
                "A planning or hypothesis phase precedes quantitative action, and reflection follows the observation."
                if loop_ok
                else f"Observed phase order: {phases}."
            ),
        )
    )

    numerical_ok = experiment_result is not None and "quant_execution" in phases
    checks.append(
        ProtocolCheck(
            check_id="executable_numerical_reasoning",
            foundation_ids=("finqa-2021",),
            passed=numerical_ok,
            evidence=(
                "A deterministic quantitative tool produced the primary experiment result."
                if numerical_ok
                else "No tool-produced primary experiment result was attached."
            ),
        )
    )

    external_feedback_ok = bool(validation_results) and all(
        isinstance(item, dict) and bool(item) for item in validation_results
    )
    checks.append(
        ProtocolCheck(
            check_id="external_feedback_before_reflection",
            foundation_ids=("reflexion-2023", "external-feedback-2024"),
            passed=external_feedback_ok and reflection_phase is not None,
            evidence=(
                f"Reflection used {len(validation_results)} attached external validation observations."
                if external_feedback_ok
                else "Reflection had no attached external validation result."
            ),
        )
    )

    hypothesis_failures = _hypothesis_failures(hypotheses)
    checks.append(
        ProtocolCheck(
            check_id="hypothesis_evidence_ledger",
            foundation_ids=("self-rag-2024", "external-feedback-2024"),
            passed=not hypothesis_failures,
            evidence=(
                f"All {len(hypotheses)} hypotheses carry evidence and an auditable state or validation requirement."
                if not hypothesis_failures
                else f"Incomplete hypotheses: {hypothesis_failures}."
            ),
        )
    )

    final_evidence = final_assessment.get("evidence", [])
    decision_ok = bool(
        final_assessment.get("verdict")
        and final_assessment.get("next_action")
        and final_assessment.get("limitations")
        and final_evidence
    )
    checks.append(
        ProtocolCheck(
            check_id="evidence_bounded_decision",
            foundation_ids=("self-rag-2024", "finqa-2021"),
            passed=decision_ok,
            evidence=(
                f"The verdict cites {len(final_evidence)} evidence statements and discloses limitations."
                if decision_ok
                else "The final verdict is missing evidence, limitations, or a next action."
            ),
        )
    )

    selection_controls_ok = (
        final_assessment.get("parameter_search_authorized") is False
        and final_assessment.get("holdout_accessed") is False
    )
    checks.append(
        ProtocolCheck(
            check_id="selection_bias_controls",
            foundation_ids=("dsr-2014",),
            passed=selection_controls_ok,
            evidence=(
                "Parameter search is not authorized and the holdout remains untouched."
                if selection_controls_ok
                else "Search authorization or holdout status is absent or unsafe."
            ),
        )
    )

    failures = tuple(item.check_id for item in checks if item.critical and not item.passed)
    return ProtocolAudit(
        passed=not failures,
        checks=tuple(checks),
        critical_failures=failures,
    )


def _first_index(phases: list[str], candidates: tuple[str, ...]) -> int | None:
    indexes = [phases.index(item) for item in candidates if item in phases]
    return min(indexes) if indexes else None


def _evidence_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _hypothesis_failures(hypotheses: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for index, item in enumerate(hypotheses, start=1):
        label = str(item.get("id") or item.get("hypothesis") or f"hypothesis-{index}")
        has_claim = bool(item.get("claim") or item.get("hypothesis"))
        has_evidence = bool(item.get("evidence") or item.get("supporting_evidence"))
        has_state = bool(
            item.get("status_after_new_experiment")
            or item.get("status_before_new_experiment")
            or item.get("validation_needed")
        )
        if not (has_claim and has_evidence and has_state):
            failures.append(label)
    return failures
