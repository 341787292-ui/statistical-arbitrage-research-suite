from __future__ import annotations

import json
import unittest

from quant_research_agent.verification.benchmark import (
    run_temporal_mutation_benchmark,
)
from quant_research_agent.verification.fixtures import (
    build_reference_temporal_ir,
    mutation_cases,
)
from quant_research_agent.verification.temporal import verify_temporal_causality


class TemporalVerifierTests(unittest.TestCase):
    def test_reference_contract_passes(self) -> None:
        report = verify_temporal_causality(build_reference_temporal_ir())

        self.assertTrue(report.passed)
        self.assertFalse(report.findings)

    def test_every_mutation_triggers_its_expected_rule(self) -> None:
        reference = build_reference_temporal_ir()
        for mutation in mutation_cases():
            with self.subTest(mutation=mutation.mutation_id):
                report = verify_temporal_causality(mutation.apply(reference))
                rule_ids = {item.rule_id for item in report.findings}
                self.assertFalse(report.passed)
                self.assertIn(mutation.expected_rule_id, rule_ids)
                for finding in report.findings:
                    self.assertTrue(finding.counterexample)
                    self.assertTrue(finding.repair)

    def test_ir_serializes_to_json(self) -> None:
        payload = build_reference_temporal_ir().to_dict()

        encoded = json.dumps(payload)
        self.assertIn("statarb-ir/0.1", encoded)
        self.assertEqual(payload["signals"][0]["generated_at"]["phase"], "after_close")

    def test_old_agent_is_a_real_control_for_temporal_faults(self) -> None:
        result = run_temporal_mutation_benchmark()
        summary = result["summary"]

        self.assertTrue(result["valid_reference_accepted"])
        self.assertEqual(summary["total_faults"], len(mutation_cases()))
        self.assertEqual(summary["old_agent_fault_recall"], 0.0)
        self.assertEqual(summary["verified_agent_fault_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
