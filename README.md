# EpiGABM

A hybrid generative agent-based forecasting pipeline for seasonal influenza outbreaks. Agent decisions are produced by a large language model. The pipeline switches from the agent layer to a compartmental SEIR tail at a detected regime change and continues the forecast with one of eleven beta-prediction methods.

This repository contains the code release of the master's thesis *Large Language Model Agents Usage in Predicting the Spread of Infectious Diseases* (Gurniak, ITMO University, 2026).

## Pipeline overview

```
synthetic population -> GABM (LLM-driven behaviour) -> switch detector -> SEIR tail
                                                                            |
                                                                            v
                                                                  predicted I(t) on horizon h
```

A 10000-agent simulation of an outbreak in Vasileostrovsky district, Saint Petersburg. Each dot is a household; colour shows the share of infected residents.

![Outbreak spread on the city map](docs/figures/spread.gif)

Four stages:

1. GABM behavioural layer. Six demographic archetypes, phase-aware prompts, per-archetype LLM calls, compliance-ceiling clipping of intent rates.
2. Switch-day detection. Threshold-variance detector on the empirical beta trace.
3. Beta prediction. Eleven methods (constant baselines, trend extrapolators, sequence-aware learners) extrapolate beta over the forecast horizon.
4. SEIR submodel. Discrete-time forward integration from the agent-layer state.

## Repository layout

```
src/                    library code (importable as `epigabm`)
  agents/               GABM behavioural layer + LLM backends
  calibration/          11 beta-prediction methods + ABC variants
  regime/               switch-day detectors
  models/               ABM, GABM, hybrid simulators, SEIR submodel
  orchestrator/         end-to-end pipeline driver
  uncertainty/          bootstrap confidence intervals
  utils/                data loaders, helpers
  visualization/        geo-map rendering, agent inspector
  logging/              run-log infrastructure

config/                 *.yaml pipeline configurations
app/                    Streamlit dashboard (visualisation app)
experiments/            exp1-exp14 experiment runners
analyses/               analysis scripts (sweeps + audits)
scripts/                CLI utilities (data download, smoke tests, runners)
tests/                  pytest suite
docs/                   documentation (architecture, config, data, experiments)
data/                   empty skeleton, populate with synthetic population
```

The headline run results, the thesis text and the defense slides are kept outside this code release.

## Installation

Python >=3.11, <3.13 is required (Python 3.13+ does not have prebuilt wheels for `hmmlearn` on Windows).

```bash
git clone <repo-url>
cd EpiGABM-release
python -m venv .venv
.venv\Scripts\activate                     # on Windows
# source .venv/bin/activate                # on macOS / Linux
pip install -e ".[llm,viz,calibration]"
```

The `pyproject.toml` defines several optional dependency groups:

- `llm`: OpenAI client, Ollama client, tiktoken
- `viz`: Streamlit, folium, plotly
- `calibration`: hmmlearn, SALib, TensorFlow, joblib
- `geo`: geopandas, osmnx
- `dev`: pytest, ruff, black, mypy

Install all at once:

```bash
pip install -e ".[llm,viz,calibration,geo,dev]"
```

## Quick start

### 1. Provide a synthetic population

Place a synthetic-population file at `data/population/households.txt` (tab-separated, columns `sp_id latitude longitude` and per-agent demographics). See `docs/DATA.md` for the expected schema and where to obtain a compatible population dataset.

### 2. Configure an LLM backend

The behavioural layer queries one of nine supported language-model backends. Copy `.env.example` to `.env` and set:

```
OPENROUTER_API_KEY=...        # for cloud models
```

A local Llama-3.1-8B can be served via Ollama (`ollama pull llama3.1:8b` then `ollama serve`).

### 3. Run a single GABM simulation

```bash
python scripts/run_all_llama.py --config config/default.yaml --seed 42
```

This produces a per-day trajectory and the empirical beta trace.

### 4. Run a cross-LLM benchmark

```bash
python analyses/run_all.py --config config/default.yaml
```

The default configuration sweeps nine backends, ten seeds per backend, with the compliance ceiling and editorial prompt components both on.

### 5. Run the four-configuration ablation

```bash
python analyses/run_phase2A.sh        # ceiling off, prompt on
python analyses/run_phase2BC.sh       # prompt stripped, ceiling on/off
```

### 6. Launch the visualisation dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard renders the per-day spread on the city map, lets the operator inspect any agent's diary, and shows the SEIR forecast overlay.

See `docs/EXPERIMENTS.md` for the full experiment matrix and how to reproduce the results reported in the thesis.

## Configuration

Four pre-registered configurations sit in `config/`:

| File | Compliance ceiling | Prompt add-ons |
|---|---|---|
| `default.yaml`              | on  | on  |
| `no_ceilings.yaml`          | off | on  |
| `no_addons.yaml`            | on  | off |
| `no_ceilings_no_addons.yaml`| off | off |

The four cells are the basis of the controlled four-configuration ablation reported in the thesis. See `docs/CONFIG.md` for the field-level reference.

## Documentation

| File | Topic |
|---|---|
| `docs/ARCHITECTURE.md` | Pipeline stages and module boundaries |
| `docs/CONFIG.md` | YAML configuration field reference |
| `docs/DATA.md` | Synthetic population format and where to obtain it |
| `docs/EXPERIMENTS.md` | Experiment matrix and reproduction commands |
| `docs/STREAMLIT.md` | Visualisation dashboard usage |
| `docs/API.md` | Module-level API of `src/` |

## Citation

The hybrid-pipeline calibration framework is described in:

> Koshkareva M., Guseva E., Sharova A., Leonenko V. *Predicting Disease Transmission Rates for Hybrid Modeling of Epidemic Outbreaks: Statistical and Machine Learning Approaches.* Computational Science, ICCS 2025 Workshops. LNCS 15908. Springer, 2025. doi:10.1007/978-3-031-97557-8_12.

## License

MIT (see `LICENSE`).
