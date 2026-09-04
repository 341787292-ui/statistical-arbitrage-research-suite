# Quant Research Agent

This product studies whether an AI system can make a quantitative research
process more evidence-grounded, executable, and auditable. It is separate from
the U.S. paper reproduction and the A-share method-migration result.

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

## Run the paper baseline

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-agent
```

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

The Agent now has a defensible paper-backed protocol, but it does not yet have
a researcher-labelled evaluation benchmark. Until that benchmark exists, the
project demonstrates methodological discipline and engineering validity, not
measured superiority over a strong ChatGPT baseline.
