# Statistical-Arbitrage Verification Research

## Research question

Can a statistical-arbitrage-specific temporal verifier reduce invalid
experiments and unsupported research conclusions produced by an LLM research
Agent?

The Agent is deliberately limited to statistical arbitrage. This lets the
system check domain facts that a general paper assistant cannot represent
precisely: when return data became available, when a hedge or factor model was
fitted, which residual history formed a signal, when an order could execute,
when returns may be credited, and whether model selection touched a holdout.

## Experimental groups

The existing `run_baseline_agent` is frozen as the first control. It checks
whether research artifacts and workflow stages exist, but it receives no
typed experiment semantics.

The new `run_verified_agent` adds:

```text
instrumented quant execution
    -> StatArb-IR temporal provenance
    -> deterministic temporal verifier
    -> concrete counterexample and repair
    -> allow or block the research conclusion
```

Later evaluation should also include a plain LLM, LLM plus retrieval, and a
free-form tool Agent. The current old-Agent comparison is not sufficient for a
publication claim.

## StatArb-IR v0.1

The first typed intermediate representation contains:

- data windows and their actual availability time;
- model-fit events and referenced training data;
- signal events and referenced feature data;
- execution events and the first return time credited to the strategy;
- development/holdout boundaries and holdout-access state;
- stable identifiers linking the provenance chain.

It intentionally does not yet formalize residual mathematics, portfolio
weights, A-share T+1 inventory, limit states, suspensions, or transaction-cost
semantics.

## Implemented temporal rules

| Rule | Invalid condition |
|---|---|
| `TEMP-001` | A data window is reversed or declared available before its final observation. |
| `TEMP-002` | A model is fitted before its training data is available. |
| `TEMP-003` | A signal is generated before its referenced model is fitted. |
| `TEMP-004` | A signal uses features that are unavailable at signal time. |
| `TEMP-005` | An order executes before its signal exists. |
| `TEMP-006` | The strategy receives returns from before execution. |
| `TEMP-007` | Model selection uses development data before it is available. |
| `TEMP-008` | The holdout is read or included during selection. |
| `TEMP-009` | Selection continues into the holdout period. |

Structural checks also reject duplicate identifiers and broken references.
Each failure contains the violating event times and a proposed repair.

## Mutation benchmark v0.1

The initial benchmark starts from one valid daily experiment and introduces
seven known faults, one at a time. The existing Agent protocol detects 0/7
because temporal semantics are outside its input. The temporal verifier detects
7/7 and accepts the valid reference.

This is a unit-level, author-constructed benchmark. It proves that the
implementation recognizes the encoded faults. It does not establish:

- prevalence of these faults in real quantitative research;
- superiority to a prompted frontier model;
- generalization to unseen implementations;
- usefulness to professional researchers;
- a publishable novelty claim.

Run it with:

```bash
python -m quant_research_agent.verification.benchmark
```

## Paper-to-code foundations

- Agentproof motivates deterministic policy checks and witness traces:
  <https://arxiv.org/abs/2603.20356>
- SIGIL motivates translating procedural requirements into a typed
  intermediate form rather than trusting runtime prose:
  <https://arxiv.org/abs/2607.27309>
- W3C PROV-O provides the general provenance vocabulary that StatArb-IR narrows
  to data, model, signal, execution, and selection events:
  <https://www.w3.org/TR/prov-o/>

This project adapts those ideas. It does not reproduce their systems, and the
combination alone is not claimed as novel.

## Next research threshold

The next meaningful step is not adding more Agent roles. It is building a
labelled `StatArbBench` containing real and systematically mutated errors from:

1. classic pairs and cointegration research;
2. PCA/IPCA residual plus OU/Fourier research;
3. learned residual-signal research;
4. U.S.-to-A-share execution transfer.

The evaluation must report fault precision/recall, valid-plan rejection rate,
repair success, repeated-run reliability, cost, latency, and ablations that
remove the IR, verifier, provenance, or retrieval.
