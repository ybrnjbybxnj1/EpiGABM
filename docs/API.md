# Module API

The library is importable as `epigabm` once installed with `pip install -e .`. Public entry points are grouped by submodule below.

Note: the example signatures below describe the intended public surface. Before importing in your own code, check the actual definitions in `src/` since some helpers are private to the runners.

## `epigabm.orchestrator`

```python
from epigabm.orchestrator import Orchestrator

orch = Orchestrator.from_yaml("config/default.yaml")
result = orch.run(seed=42)
```

Returns a `RunResult` dataclass with the per-day SEIR trace, the empirical beta trace, the detected switch day, and the forecast.

## `epigabm.agents`

```python
from epigabm.agents.archetypes import load_archetypes
from epigabm.agents.base import Agent
from epigabm.agents.backends.ollama import OllamaBackend
from epigabm.agents.prompts import build_prompt

archetypes = load_archetypes("config/archetypes.yaml")
backend = OllamaBackend(model="llama3.1:8b")
prompt = build_prompt(archetype="elderly", phase="GROWTH", prevalence=0.11)
response = backend.query(prompt)
```

Available backends:

| Module | Provider |
|---|---|
| `agents.backends.ollama` | Local Ollama server. |
| `agents.backends.openrouter` | OpenRouter API. |
| `agents.backends.gpt4` | OpenAI direct. |
| `agents.backends.gemini` | Google Gemini. |
| `agents.backends.qwen` | Qwen direct. |
| `agents.backends.llama` | Llama direct. |
| `agents.backends.mock` | Deterministic mock for tests. |

All backends expose the same `query(prompt: str) -> dict` interface.

## `epigabm.regime`

```python
from epigabm.regime.threshold import ThresholdDetector
from epigabm.regime.hmm_detector import HMMDetector
from epigabm.regime.combined_detector import CombinedDetector

detector = CombinedDetector.from_config(cfg["regime"])
switch_day = detector.detect(beta_trace)   # int or None
```

## `epigabm.calibration`

### Beta prediction: eleven methods

```python
from epigabm.calibration.beta_prediction import predict_beta

beta_future = predict_beta(
    method="mlp",          # one of the eleven methods listed below
    beta_observed=trace,
    horizon=14,
    t_obs=12,
)
```

The eleven supported methods:

| Family | Method key |
|---|---|
| Constant baseline | `last_value`, `mean_last_k`, `median_last_k`, `mean_growth`, `mean_decline`, `regression_day` |
| Trend extrapolator | `linear`, `exponential`, `regression` |
| Sequence-aware learner | `lstm`, `mlp` |

### ABC calibration

```python
from epigabm.calibration.abc import abc_rejection, abc_smc, abc_annealing

posterior = abc_smc(observed=incidence, simulator=run_seir, n_particles=1000)
```

## `epigabm.models`

```python
from epigabm.models.gabm import GABM
from epigabm.models.hybrid import HybridSimulator
from epigabm.models.seir import seir_step

sim = HybridSimulator(config=cfg, backend=backend, detector=detector)
trajectory = sim.run(population=pop, days=150)
```

`seir_step(state, beta, sigma, gamma)` advances one day and is reusable outside the simulator (the beta-prediction tail calls it).

## `epigabm.uncertainty`

```python
from epigabm.uncertainty.bootstrap import cluster_bootstrap_ci

ci = cluster_bootstrap_ci(
    samples=peak_heights,
    cluster_ids=backend_ids,
    n_boot=1000,
    level=0.95,
)
```

Returns `(lower, upper)`.

## `epigabm.utils.data_loader`

```python
from epigabm.utils.data_loader import load_population

pop = load_population("data/population/households.txt", n_agents=3000)
```

Returns a `Population` object exposing `agents`, `households`, `contact_network`.

## `epigabm.visualization`

```python
from epigabm.visualization.dashboard import run_dashboard
from epigabm.visualization.map_component import render_day_map

# or just launch from the CLI:  streamlit run app/dashboard.py
```

See `STREAMLIT.md` for usage.

## `epigabm.logging`

Run-log helpers used by every entry point. Writes per-day JSONL records and prompt/response transcripts to `data/results/<run-id>/logs/`. Not normally called directly by user code.
