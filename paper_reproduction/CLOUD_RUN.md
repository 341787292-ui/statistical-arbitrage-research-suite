# Cloud and Second-Computer Runbook

## Hardware

- OU+Threshold runs comfortably on CPU.
- Fourier+FFN can run on CPU, but CUDA is faster for the full grid.
- CNN+Transformer should use a CUDA GPU. Start with at least 16 GB of GPU
  memory and lower `--chunk-size` if memory is tight.
- The three compressed five-factor residual files require about 120 MB of disk
  space and expand to roughly 1.1 GB in memory when loaded one at a time.

## Setup

```bash
git clone https://github.com/341787292-ui/ai-quant-research-agent.git
cd ai-quant-research-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r paper_reproduction/requirements.txt
python paper_reproduction/download_official_data.py
python -m unittest discover -s paper_reproduction/tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Experiments

Run the complete OU+Threshold five-factor comparison:

```bash
python paper_reproduction/run_official_table1_ou.py
```

Run a short neural check before spending GPU time:

```bash
python paper_reproduction/run_official_neural_smoke.py --model cnn --device cuda
```

Run the formal PCA-5 neural approximations:

```bash
python paper_reproduction/run_official_neural_full.py --factor-model pca --model fourier --device cuda
python paper_reproduction/run_official_neural_full.py --factor-model pca --model cnn --device cuda --chunk-size 2048
```

Repeat with `--factor-model ff` and `--factor-model ipca` for the other Table I
columns. Each rolling origin is saved under
`paper_reproduction/output/periods/`. Re-running the same command resumes from
completed origins; add `--force` to recompute them. Use `--start-origin` and
`--max-retrains` to split rolling origins across machines.

## Colab

Select a GPU runtime, then run the setup and experiment commands above in
notebook shell cells by prefixing them with `!`. Keep the repository in mounted
Google Drive if checkpoints must survive a runtime reset.

## Interpretation Boundary

The public author repository does not include the residual composition
matrices used for stock-space L1 normalization. Cloud hardware does not solve
that data limitation. Until those matrices are rebuilt from licensed or
equivalent point-in-time raw data, label every neural result from the public
arrays as a **residual-space approximation**, not an exact Table I replication.
