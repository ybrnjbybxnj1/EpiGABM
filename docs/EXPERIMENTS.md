# Experiments

The repository carries two layers of experiment runners:

- `experiments/exp1`-`exp14`: the original experiment matrix used during the research stage of the project.
- `analyses/`: the analysis runners that produce the headline numbers reported in the thesis.

Both layers consume the same YAML configs and share the same library code in `src/`.

## Headline runs

Each entry point below corresponds to a numbered result reported in the thesis.

### exp01: beta-predictor sweep

```bash
python analyses/exp01_beta_predictor_sweep.py --config config/default.yaml
```

Runs all eleven beta-prediction methods (constant baselines, trend extrapolators, sequence-aware learners) at every `(T_obs, h)` pair on the same observed beta trace. Output: per-method RMSE table.

### exp02: calibration audit

```bash
python analyses/exp02_calibration_audit.py --config config/default.yaml
```

Compares three ABC calibration variants (rejection, SMC, annealing) on predictive RMSE.

### exp03: detector benchmark

```bash
python analyses/exp03_detector_benchmark.py --config config/default.yaml
python analyses/exp03b_synthetic_phase_benchmark.py --config config/default.yaml
```

Threshold-variance vs HMM vs Rt detectors on real and synthetic phase data.

### run_all: cross-LLM sweep

```bash
python analyses/run_all.py --config config/default.yaml
```

Headline cross-model comparison. Sweeps nine backends with ten seeds each on the `default.yaml` cell (compliance ceiling on, prompt add-ons on).

### run_phase2A / run_phase2BC: four-configuration ablation

```bash
bash analyses/run_phase2A.sh         # ceiling off, prompt add-ons on
bash analyses/run_phase2BC.sh        # prompt stripped, ceiling on/off
```

Two scripts together populate the three ablation cells beyond `default.yaml`. Each cell runs all nine backends, ten seeds.

### run_10k_seed: scalability run

```bash
python analyses/run_10k_seed.py --config config/default.yaml --seed 42
```

10000-agent simulation that confirms the pipeline runs to completion at the JoCS-scale population. Reports final attack rate.

### phase2_bootstrap / bootstrap_nrmse_ci: confidence intervals

```bash
python analyses/phase2_bootstrap.py
python analyses/bootstrap_nrmse_ci.py
```

Hierarchical cluster bootstrap (B=1000) on the cross-model peak-height results and on the beta-predictor NRMSE table. Paired Wilcoxon comparisons with rank-biserial effect size and Bonferroni adjustment.

### train_lstm_beta / train_mlp_beta: sequence-aware predictors

```bash
python analyses/train_lstm_beta.py
python analyses/train_mlp_beta.py
```

Trains the LSTM and MLP beta-extrapolation models on simulated beta traces. Trained weights are loaded by `exp01`.

## Original experiment matrix

These are the experiments from the earlier research stage. `experiments/run_all.py` runs the full matrix; `experiments/run_from.py <N>` resumes from experiment N.

| Experiment | What it tests |
|---|---|
| `exp1_gabm_vs_rulebased.py` | GABM vs rule-based ABM baseline on the same population. |
| `exp2_archetype_scaling.py` | Sensitivity to archetype count. |
| `exp3_abm_vs_gabm_divergence.py` | Divergence of GABM and rule-based trajectories over time. |
| `exp4_prompt_sensitivity.py` | Variance under small prompt perturbations. |
| `exp5_llm_stochasticity.py` | Variance under repeated calls with `temperature > 0`. |
| `exp6_hybrid_gabm_seir.py` | Full hybrid GABM-then-SEIR pipeline. |
| `exp7_hmm_switching.py` | HMM detector on simulated traces. |
| `exp8_calibration_forecast.py` | ABC calibration to forecast pipeline. |
| `exp9_cross_model.py` | Cross-backend comparison (precursor to `run_all`). |
| `exp10_temperature_sweep.py` | Temperature sweep at fixed model and seed. |
| `exp11_ablation.py` | Component-level ablation. |
| `exp12_switch_robustness.py` | Switch-day robustness under noise. |
| `exp13_forecasting_benchmark.py` | Beta-predictor benchmark (precursor to `exp01`). |
| `exp14_forecasting_smc.py` | SMC-ABC forecast pipeline. |
| `diagnose_detector.py` | Standalone detector diagnostics. |

## Reproducing the four-configuration ablation

The headline ablation reported in the thesis covers four cells:

| Config | Ceiling | Prompt add-ons | Run with |
|---|---|---|---|
| `default.yaml`              | on  | on  | `run_all.py --config config/default.yaml` |
| `no_ceilings.yaml`          | off | on  | `run_phase2A.sh` |
| `no_addons.yaml`            | on  | off | `run_phase2BC.sh` (first half) |
| `no_ceilings_no_addons.yaml`| off | off | `run_phase2BC.sh` (second half) |

Each cell runs nine backends x ten seeds = 90 trajectories. Total compute: roughly 360 simulations.

## Smoke tests

Before running a full sweep:

```bash
python scripts/smoke_test_compliance_toggle.py    # 50-agent dry run, both ceiling settings
python scripts/probe_openrouter_model.py          # checks that the OpenRouter slots respond
python scripts/in_distribution_lstm_test.py       # verifies the LSTM beta predictor loads
```
