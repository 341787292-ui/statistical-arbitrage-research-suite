"""Domain verification for statistical-arbitrage research workflows."""

from quant_research_agent.verification.ir import StatArbExperimentIR
from quant_research_agent.verification.temporal import (
    VerificationReport,
    verify_temporal_causality,
)

__all__ = [
    "StatArbExperimentIR",
    "VerificationReport",
    "verify_temporal_causality",
]
