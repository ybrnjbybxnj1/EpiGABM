from __future__ import annotations

import sys
import time

from loguru import logger

def run_all(n_agents: int = 3000, seed: int = 42) -> None:
    logger.info("running all experiments, n_agents={}", n_agents)
    t0 = time.time()

    from experiments.exp1_gabm_vs_rulebased import run_experiment as exp1
    from experiments.exp2_archetype_scaling import run_experiment as exp2
    from experiments.exp3_gpt4_vs_llama import run_experiment as exp3
    from experiments.exp4_prompt_sensitivity import run_experiment as exp4
    from experiments.exp5_llm_stochasticity import run_experiment as exp5
    from experiments.exp6_hybrid_gabm_seir import run_experiment as exp6
    from experiments.exp7_hmm_switching import run_experiment as exp7
    from experiments.exp8_calibration_forecast import run_experiment as exp8
    from experiments.exp9_cross_model import run_experiment as exp9
    from experiments.exp10_temperature_sweep import run_experiment as exp10
    from experiments.exp11_ablation import run_experiment as exp11
    from experiments.exp12_switch_robustness import run_experiment as exp12

    n_bootstrap = 10 if n_agents >= 1000 else 10

    n_runs_per_temp = 5 if n_agents >= 1000 else 2

    n_seeds_robustness = 10 if n_agents >= 1000 else 5

    plan = [
        ("exp1",  exp1,  {}),
        ("exp2",  exp2,  {"archetype_counts": [2, 3, 5, 6]}),
        ("exp3",  exp3,  {}),
        ("exp4",  exp4,  {}),
        ("exp5",  exp5,  {"n_runs": n_bootstrap}),
        ("exp6",  exp6,  {}),
        ("exp7",  exp7,  {}),
        ("exp8",  exp8,  {}),
        ("exp9",  exp9,  {}),
        ("exp10", exp10, {"n_runs_per_temp": n_runs_per_temp}),
        ("exp11", exp11, {}),
        ("exp12", exp12, {"n_seeds": n_seeds_robustness, "use_abm_for_speed": True}),
    ]

    for i, (name, fn, kwargs) in enumerate(plan, 1):
        logger.info("{}/{}: {} (n={})", i, len(plan), name, n_agents)
        try:
            fn(seed=seed, n_agents=n_agents, **kwargs)
        except Exception:
            logger.exception("failed: {}", name)

    elapsed = time.time() - t0
    logger.info("all done, n_agents={}, total {:.1f}s ({:.1f} min)",
                n_agents, elapsed, elapsed / 60)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run_all(n_agents=n)
