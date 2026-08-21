# Statistical Arbitrage Research Suite

This repository contains three deliberately separate products built around
statistical-arbitrage research.

| Product | Purpose | Current status |
|---|---|---|
| `paper_reproduction/` | Internship project: reproduce *Deep Learning Statistical Arbitrage* | OU and all three five-factor Fourier+FFN public-data approximations complete |
| `ashare_stat_arb/` | Applied research: adapt the method to A-share market constraints | Product scaffold and tested T+1 execution baseline complete |
| `quant_research_agent/` | Research project: build a statistical-arbitrage research Agent | First deterministic Agent MVP retained; development currently paused |

The products share research concepts, but they do not share empirical claims.
The U.S. paper result, A-share adaptation, and Agent evaluation must always be
reported separately.

## 1. U.S. paper reproduction

`paper_reproduction/` is the frozen reference implementation for the paper.
It includes:

- no-lookahead rolling residual construction;
- OU+Threshold, Fourier+FFN, and CNN+Transformer implementations;
- rolling neural training and resumable checkpoints;
- official-data stability and Table I comparison tools;
- tests for signal timing, normalization, training, and audit reconciliation.

The completed public-data runs use the authors' Fama-French, PCA, and IPCA
five-factor residual arrays. The repository does not contain the licensed raw
CRSP/Compustat inputs or the unpublished residual composition matrices.
Results are therefore labeled **residual-space approximations**, not exact
empirical reproductions.

See:

- `paper_reproduction/RESULTS.md`
- `paper_reproduction/STATUS.md`
- `paper_reproduction/REPRODUCTION_SPEC.md`
- `paper_reproduction/RESEARCHER_WALKTHROUGH.md`

Run its tests with a Python 3.11 or 3.12 environment:

```bash
pip install -r paper_reproduction/requirements.txt
python -m unittest discover -s paper_reproduction/tests -v
```

## 2. A-share market adaptation

`ashare_stat_arb/` studies how the paper's residual mean-reversion mechanism
changes under A-share execution constraints.

The first runnable component supports:

- long-only cash-equity holdings;
- T+1 sellable inventory;
- suspension handling;
- conservative upper-limit buy and lower-limit sell rejection;
- lot-size constraints;
- configurable directional trading fees.

Run the baseline demonstration and tests:

```bash
python -m ashare_stat_arb.run_execution_demo
python -m unittest discover -s ashare_stat_arb/tests -v
```

See `ashare_stat_arb/RESEARCH_SPEC.md` for the point-in-time data contract and
the separation between theoretical long-short and executable A-share tracks.

## 3. Quant research Agent

`quant_research_agent/` is an early research-engineering prototype. It turns a
paper or note into an auditable workflow:

```text
Paper / note
  -> retrieval and structured paper analysis
  -> baseline experiment plan
  -> deterministic quant tools
  -> hypothesis generation and validation
  -> reflected research report
```

The Agent's synthetic pair-spread backtest is an engineering baseline, not a
reproduction of the paper.

Run it locally:

```bash
pip install -r requirements.txt
python main.py --paper samples/stat_arb_note.txt --no-llm --run-agent
python -m unittest discover -s tests -v
```

An optional OpenAI-backed extraction path is available through `.env`; API
keys must never be committed.

## Repository data policy

The repository contains source code, tests, research specifications, and
human-readable result summaries. It intentionally excludes:

- API keys and `.env` files;
- licensed or large market datasets;
- author residual arrays and paper PDFs;
- generated checkpoints, figures, and reports;
- local virtual environments and IDE preferences.

Download scripts and manifests are retained so allowed public inputs can be
recreated locally.

## Clone

```bash
git clone https://github.com/341787292-ui/statistical-arbitrage-research-suite.git
cd statistical-arbitrage-research-suite
```

The Docker configuration at the repository root runs the Agent MVP. The paper
and A-share products are currently research workflows rather than hosted web
services.
