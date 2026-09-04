# Quant Research Agent

This product studies whether an AI system can make a quantitative research
process more evidence-grounded, executable, and auditable. It is separate from
the U.S. paper reproduction and the A-share method-migration result.

## Two Agent paths

The original Agent remains unchanged as the experimental control:

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-agent
```

The verified path adds typed temporal provenance and deterministic checks:

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-verified-agent
```

The verified Quant tool generates signals after close and executes them at the
next session's open. It emits a `StatArb-IR` trace linking data, model fit,
signal, execution, return attribution, and holdout selection.

To demonstrate that a bad experiment is blocked rather than merely described:

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-verified-agent --verification-mutation same_close_execution
```

This flag is benchmark-only fault injection; it does not represent an observed
error in the U.S. or A-share empirical projects.

## Research protocol

The runnable loop is:

```text
paper or frozen empirical evidence
    -> retrieve / inspect evidence
    -> form explicit hypotheses
    -> select deterministic quant tools
    -> observe validation results
    -> update the decision
    -> audit the entire protocol
```

Each trace step contains `method_ids` linking it to a technical foundation.
The final output contains both a methodology manifest and an eight-gate
protocol audit. See [TECHNICAL_FOUNDATIONS.md](TECHNICAL_FOUNDATIONS.md).

The deterministic local mode is intentional: it lets the protocol tests run
without an API key. Supplying an OpenAI key changes paper extraction, not the
quantitative calculation or the protocol gates.

## Run the A-share diagnostic case

After recreating the public-data panel and diagnostic inputs:

```bash
python run_ashare_agent.py
```

The A-share adapter is bounded. It cannot search parameters or inspect the
sealed holdout.

## Test

```bash
python -m unittest discover -s tests -v
```

Key tests deliberately remove external feedback, retrieved evidence, or
selection-bias controls and verify that the protocol fails.

## Current limitation

The initial mutation benchmark contains seven author-constructed temporal
faults. The old protocol detects 0/7 and the temporal verifier detects 7/7,
while accepting the valid reference. This proves implementation behavior only.
The Agent does not yet have a researcher-labelled benchmark, real-code fault
corpus, strong prompted-LLM baselines, or ablations. Until those exist, the
project demonstrates a working research instrument, not measured superiority
or a publication-level novelty claim. See
[VERIFICATION_RESEARCH.md](VERIFICATION_RESEARCH.md).
