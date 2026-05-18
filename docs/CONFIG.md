# Configuration reference

YAML configurations live in `config/`. Each entry point (`scripts/run_all_llama.py`, `experiments/run_all.py`, `analyses/run_all.py`) takes a `--config` argument pointing at one of these files.

Four pre-registered configurations form the controlled four-configuration ablation reported in the thesis:

| File | Compliance ceiling | Prompt add-ons |
|---|---|---|
| `default.yaml`              | on  | on  |
| `no_ceilings.yaml`          | off | on  |
| `no_addons.yaml`            | on  | off |
| `no_ceilings_no_addons.yaml`| off | off |

The differences between the four cells are confined to two flags (`agents.apply_compliance_ceilings` and the archetype prompt-addon block). All other fields are identical.

## Top-level sections

```yaml
model:           # SEIR / population parameters
agents:          # GABM behavioural layer + LLM backend selection
regime:          # switch-day detector
calibration:     # beta-prediction method + ABC variant
uncertainty:     # bootstrap parameters
visualization:   # Streamlit / map rendering parameters
data:            # paths to population, geo file, archetypes file
output:          # where the run writes its results
```

## `model:` population and SEIR parameters

| Field | Type | Default | Meaning |
|---|---|---|---|
| `population_size` | int | 3000 | Number of agents drawn from the synthetic population (households are not split, a whole household is included or excluded). |
| `network.type` | str | `barabasi_albert` | Contact network topology. |
| `network.m` | int | 5 | Mean degree parameter. |
| `seir.sigma` | float | 0.2 | Latency rate (1/incubation period in days). |
| `seir.gamma` | float | 0.14 | Recovery rate (1/infectious period in days). |
| `seir.initial_immunity` | float | 0.3 | Fraction of the population starting in R. |
| `lmbd` | float | 0.4 | Base transmission rate (lambda). |
| `strains` | list[str] | `[H1N1, H3N2, B]` | Influenza strains tracked. |
| `infected_init` | dict | n/a | Initial infected count per strain. |
| `alpha` | dict | n/a | Fraction susceptible per strain. |
| `days` | [start, end] | `[1, 150]` | Simulation window. |

## `agents:` GABM behavioural layer

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_archetypes` | int | 6 | Number of demographic archetypes. |
| `update_trigger` | str | `phase_change` | When to re-query the LLM. `phase_change` issues a new call when the detected phase flips; `every_n_steps` re-queries on a fixed schedule. |
| `update_every_n` | int | 7 | Cadence for `every_n_steps` mode. |
| `stochastic_noise` | float | 0.3 | Per-agent jitter added to archetype rates before the Bernoulli draw. |
| `apply_compliance_ceilings` | bool | `true` | Top-level ablation switch. When `true`, archetype-level intent rates are clipped to literature-calibrated upper bounds (`compliance_rate`). When `false`, raw LLM rates propagate unchanged. |
| `behavior_activation.min_prevalence` | float | 0.01 | Prevalence floor below which agents ignore the outbreak. |
| `behavior_activation.news_threshold` | float | 0.05 | Prevalence above which the prompt mentions news/measures. |
| `compliance_rate.isolate_healthy` | float | 0.25 | Ceiling on self-isolation rate. |
| `compliance_rate.mask` | float | 0.30 | Ceiling on mask-wearing rate. |
| `compliance_rate.reduce_contacts` | float | 0.40 | Ceiling on contact-reduction rate. |
| `compliance_rate.see_doctor` | float | 0.20 | Ceiling on doctor-visit rate. |
| `mask_reduction_factor` | float | 0.5 | Transmission multiplier when an agent wears a mask. |
| `contact_reduction_factor` | float | 0.7 | Network-edge-weight multiplier when an agent reduces contacts. |
| `doctor_recovery_day` | int | 6 | Day of illness after which doctor visits reduce contagiousness. |
| `max_cost_usd` | float | 50 | Soft cap on cloud-LLM spend per run. |
| `primary_backend` | str | `llama` | Key into the `backends:` dict to use for the headline simulation. |
| `compare_backends` | bool | `true` | If `true`, the orchestrator runs the full sweep over every defined backend. |
| `backends.*` | dict | n/a | One entry per backend; see below. |

### Backend entries

Each backend has the same skeleton:

```yaml
backend_name:
  provider: openrouter        # implicit "local" if absent (Ollama / direct provider)
  model: <model id>
  endpoint: <url>             # for Ollama-style local servers
  temperature: 0.3
  max_tokens: 500
  cost_per_1m_input: 0.05     # optional, only used for spend tracking
  cost_per_1m_output: 0.08
```

The free OpenRouter slots are upstream-throttled; the paid slots (`or_mistral24b`, `or_qwen235b`, ...) carry the cross-LLM sweep.

## `regime:` switch-day detector

| Field | Type | Default | Meaning |
|---|---|---|---|
| `method` | str | `combined` | Detector to use: `combined`, `relative_rt`, `hmm`, or `threshold`. |
| `hmm.n_states` | int | 4 | HMM state count. |
| `hmm.features` | list[str] | `[incidence, growth_rate]` | Features fed to the HMM. |
| `threshold.method` | str | `roll_var_npeople` | Variance smoother. |
| `relative_rt.si_mean` | float | 2.6 | Mean serial interval in days. |
| `relative_rt.si_sd` | float | 1.5 | SD of the serial interval. |
| `relative_rt.tau` | int | 5 | Rt-estimation window. |
| `relative_rt.alpha` | float | 0.7 | Triggers a switch when Rt < `alpha * plateau`. |
| `relative_rt.warmup_days` | int | 5 | Days after the prevalence floor before the plateau is fixed. |
| `relative_rt.confirm_days` | int | 2 | Required consecutive days below trigger. |
| `relative_rt.min_prevalence` | float | 0.02 | Prevalence floor that suppresses stochastic dips. |
| `relative_rt.absolute_fallback` | float | 1.0 | Safety net: also trigger if Rt < 1. |

## `calibration:` beta prediction and ABC

| Field | Type | Default | Meaning |
|---|---|---|---|
| `method` | str | `hm_abc` | Calibration method (`hm_abc` runs history matching then ABC). |
| `hm_waves` | int | 3 | History-matching wave count. |
| `abc_method` | str | `smc` | ABC variant (`rejection`, `smc`, `annealing`). |
| `abc_n_particles` | int | 1000 | ABC particle count. |
| `beta_prediction` | str | `regression` | Beta-extrapolation method. See `API.md` for the full list of eleven methods. |

## `uncertainty:` bootstrap

| Field | Type | Default | Meaning |
|---|---|---|---|
| `bootstrap_runs` | int | 100 | Bootstrap iterations. The headline runs use 1000; bump this field for confirmatory runs. |
| `confidence_level` | float | 0.95 | Two-sided CI level. |

## `data:` paths

| Field | Type | Default | Meaning |
|---|---|---|---|
| `archetypes_file` | path | `config/archetypes.yaml` | Archetype rate table. The `no_addons` variants point at `archetypes_no_addons.yaml`. |
| `population_dir` | path | `data/population/` | Where `households.txt` is read from (see `DATA.md`). |
| `geo_file` | path | `data/geo/vasileostrovsky.geojson` | District polygon for the map dashboard. |

## `output:` results

| Field | Type | Default | Meaning |
|---|---|---|---|
| `metrics_dir` | path | `results/metrics/` | Per-run scalar metrics. |
| `snapshots_dir` | path | `results/snapshots/` | Per-day population snapshots used by the Streamlit map. |
| `logs_dir` | path | `results/logs/` | Run logs and prompt/response transcripts. |
| `figures_dir` | path | `results/figures/` | Run-level plots. |
