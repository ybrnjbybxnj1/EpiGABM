# EpiGABM

> GABM flu model on a Saint-Petersburg synthetic population, with an
> R_t-triggered handoff to SEIR once the peak passes.

EpiGABM runs an agent-based flu simulation where the behavioural rules
come from an LLM. Instead of querying the LLM per agent, agents are
bucketed into six archetypes, so a run costs ~15 LLM calls total.
Once a phase detector flags deceleration, the trajectory is handed off
to a compartmental SEIR tail.

There are five experiment blocks: behavioural effect of the GABM layer
(E1), robustness across LLM backends and prompts (E2), hybrid handoff
quality (E3), Bayesian calibration of the post-switch SEIR (E4), and
out-of-sample forecasting against classical baselines (E5).

This repo backs the НИР4 report and the draft EpiDAMIK-2026 submission.

## Table of contents

- [Install](#install)
- [Usage](#usage)
- [Experiments](#experiments)
- [Documentation](#documentation)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Citing](#citing)
- [License](#license)

## Install

The runtime is Python 3.12 (3.11 also works; 3.13+ breaks `hmmlearn`
on Windows because no prebuilt wheel exists). An Ollama endpoint with
`llama3.1:8b` is required for the generative backend; `gemma3:12b` and
`qwen3:8b` are optional for the cross-LLM experiment.

```bash
git clone https://github.com/ybrnjbybxnj1/nir4.git
cd nir4
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Ollama (local LLM server)
ollama serve &
ollama pull llama3.1:8b
ollama pull gemma3:12b      # optional
ollama pull qwen3:8b        # optional
```

## Usage

Run any single experiment on 3000 agents with the default seed 42:

```bash
py -3.12 -m experiments.exp1_gabm_vs_rulebased
py -3.12 -m experiments.exp6_hybrid_gabm_seir
py -3.12 -m experiments.exp14_forecasting_smc
```

Run the full sweep (exp1 through exp12) with local Llama 3.1 8B:

```bash
py -3.12 -m experiments.run_all 3000
```

Resume the sweep starting from a specific experiment index:

```bash
py -3.12 -m experiments.run_from 3000 7     # start at exp7
```

Each experiment writes a JSON metrics file to `results/metrics/` and
one or more figures to `results/figures/`.

## Experiments

Fourteen experiments in five blocks (E1-E5 in the preprint). `run_all`
runs exp1..exp12; the two forecasting experiments (exp13, exp14) run
separately because they take longer and have their own seeds:

| Block                   | Experiments             | What it measures                                                    |
| ----------------------- | ----------------------- | ------------------------------------------------------------------- |
| E1 behavioural effect   | exp1, exp2, exp3, exp11 | peak reduction, archetype scaling, per-day divergence, ablation     |
| E2 robustness           | exp4, exp5, exp9, exp10 | prompt framing CV, bootstrap CI, cross-LLM range, temperature sweep |
| E3 hybrid & detector    | exp6, exp7, exp12       | hybrid RMSE vs full GABM; phase detector F1; switch-day reliability |
| E4 Bayesian calibration | exp8                    | four-method posterior, coverage on the active window                |
| E5 forecasting          | exp13, exp14            | out-of-sample MAE at horizons {1, 7, 14, 21} days                   |

Headline numbers at N=3000 agents:

- GABM cuts the epidemic peak by 37.5% vs a rule-based ABM.
- Cross-LLM peak-reduction range: 3.2 pp across Llama/Gemma/Qwen.
  Parse success is 100% for all three.
- Hybrid GABM->SEIR is 2.2x faster than full GABM. Trajectory RMSE 93
  vs 208 for a SEIR-only baseline.
- Phase detector F1: relative-R_t 0.895, rolling-variance 0.774,
  4-state Gaussian HMM 0.330.
- SMC-ABC calibration hits 100% posterior-predictive coverage on the
  22-day active window.
- Hybrid GABM->SEIR with the Koshkareva 2025 quadratic-beta predictor
  beats a classical SEIR MLE fit by ~5x at h=7 days (MAE 149 vs 757).

## Documentation

- `docs/thesis_notes.md` - per-experiment design notes, limitations,
  and known issues. Internal master doc.
- `docs/paper/NIR4.tex` - EpiDAMIK-2026 preprint draft (ACM sigconf).
- `docs/defense_theory_checklist.md` - crib sheet for defence Q&A
  (epi theory, ABM, GABM, hybrid architectures, UQ).
- `docs/architecture.md`, `docs/models.md`, `docs/agents.md`,
  `docs/regime-calibration.md`, `docs/visualization.md`,
  `docs/config.md`, `docs/testing.md`, `docs/experiments.md` -
  per-subsystem docs.

## Tests

```bash
py -3.12 -m pytest -v
```

80 tests, all passing on Python 3.12 with the pinned deps. Covers the
ABM/GABM cores, the three phase detectors, the agent backends and
prompt parser, beta-prediction, bootstrap, and sensitivity helpers.

## Repository layout

```
src/
  agents/       archetypes, prompts, LLM backends (Llama / Gemma / Qwen)
  models/       ABM, GABM, SEIR, hybrid, shared data structures
  regime/       threshold, HMM, relative-R_t, combined detectors
  calibration/  history matching, ABC (rejection / annealing / SMC)
  uncertainty/  bootstrap, sensitivity helpers
  visualization/ Streamlit dashboard, geographic map, epidemic curve
experiments/    exp1..exp14 + run_all, run_from, diagnose_detector
tests/          80 tests
config/         default.yaml, archetypes.yaml
data/population/ people.txt, households.txt (synthetic SPb)
docs/           thesis notes, paper draft, subsystem docs
results/        metrics/ (JSON), figures/ (PNG), logs/ (JSONL)
```

## Citing

If you reuse the detector or the calibration audit, cite the EpiGABM
preprint and the lab papers it builds on:

- Gurniak and Leonenko (2026). *Generative agent-based modelling of
  influenza outbreaks with a hybrid handoff to a compartmental
  sub-model.* Preprint (EpiDAMIK workshop at KDD 2026).
- Koshkareva, Guseva, Sharova, and Leonenko (2025). *Predicting
  disease transmission rates for hybrid modelling of epidemic
  outbreaks: statistical and machine-learning approaches.*
  ICCS 2025, LNCS 15908, Springer.
- Cori, Ferguson, Fraser, and Cauchemez (2013). *A new framework and
  software to estimate time-varying reproduction numbers during
  epidemics.* American Journal of Epidemiology, 178(9):1505–1512.
- Chopra, S. Kumar, Giray-Kuru, Raskar, and Quera-Bofarull (2025).
  *On the limits of agency in agent-based models.* AAMAS 2025.
- Leonenko, Arzamastsev, and Bobashev (2020). *Contact patterns and
  influenza outbreaks in Russian cities: a proof-of-concept study.*
  Journal of Computational Science, 44:101156.

## License

Academic use for now. The synthetic population files under
`data/population/` inherit the ITMO DHL licensing terms.
