# AI Quant Research Agent

An early MVP for an AI-assisted quantitative research workflow focused on statistical arbitrage.

The first baseline agent converts a paper or note into an auditable research loop:

```text
Paper / note
  -> text chunks
  -> lightweight retrieval
  -> structured paper analysis
  -> baseline reproduction plan
  -> baseline backtest
  -> hypothesis generation
  -> cost and period validation
  -> reflected research report
```

## Quick Start

Use Python 3.11 or newer. Create an isolated environment before installing dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --paper samples\stat_arb_note.txt --no-llm --run-agent
```

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --paper samples/stat_arb_note.txt --no-llm --run-agent
```

The generated report and JSON result are written under `reports/`.

### Optional LLM extraction

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then export the variables in your shell or IDE. Never commit `.env`.

```bash
python main.py --paper samples/stat_arb_note.txt
```

The output report is written to `reports/paper_research_spec.md`.

To include the deterministic local statistical arbitrage baseline:

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-quant
```

To run the complete baseline Agent, including automatic validation and reflection:

```bash
python main.py --paper samples/stat_arb_note.txt --no-llm --run-agent
```

To analyze a PDF, install the optional PDF dependency and pass the file path:

```bash
pip install -r requirements.txt
python main.py --paper "knowledge/papers/Deep Learning Statistical Arbitrage.pdf"
```

If `OPENAI_API_KEY` is set, the analyzer will try to use the OpenAI API for structured extraction. Without an API key, it falls back to deterministic rule-based extraction so the pipeline remains runnable locally.

## Use On Another Computer

After this repository is pushed to GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/ai-quant-research-agent.git
cd ai-quant-research-agent
```

Then follow the Windows, macOS, or Linux setup above. Paper PDFs and generated reports are local research materials and are intentionally not committed by default.

## Docker

The baseline Agent can also run in a reproducible container:

```bash
docker build -t ai-quant-research-agent .
docker run --rm -v "${PWD}/reports:/app/reports" ai-quant-research-agent
```

On Windows PowerShell, replace `${PWD}` with `${PWD}.Path` if your Docker setup requires an absolute path.

GitHub stores and distributes the code; it does not keep this CLI running as a cloud service. A browser-accessible cloud version will require an API/UI layer and a deployment target in the next phase.

## Project Shape

```text
quant_research_agent/
  agent/      Paper analysis, planning, and report generation
  rag/        Document loading, chunking, and retrieval
  llm.py      Optional OpenAI client wrapper
  pipeline.py End-to-end orchestration
samples/     Local sample documents for tests and demos
tests/       Smoke tests for the first runnable slice
```

## Current MVP

Implemented:

- Load `.txt`, `.md`, and `.pdf` documents.
- Split text into overlapping chunks.
- Retrieve relevant chunks for research concepts.
- Extract a structured paper research specification.
- Generate a baseline statistical arbitrage reproduction plan.
- Run a deterministic pair-spread statistical arbitrage baseline without external data.
- Choose and execute transaction-cost and period-stability validation tools.
- Update the initial hypothesis after observing validation evidence.
- Record an auditable Agent execution trace.
- Write a markdown report.

Next:

- Add real market data loading and swap the deterministic sample into the same Quant Engine interface.
