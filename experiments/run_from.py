from __future__ import annotations

import argparse
import time

from loguru import logger

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n_agents", type=int, nargs="?", default=3000)
    parser.add_argument("start", type=int, nargs="?", default=1,
                        help="first experiment index to run (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", type=str, default="",
                        help="comma-separated experiment indices to skip")
    args = parser.parse_args()

    skip = {int(s) for s in args.skip.split(",") if s.strip()}

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

    n_agents = args.n_agents
    n_bootstrap = 10 if n_agents >= 1000 else 10
    n_runs_per_temp = 5 if n_agents >= 1000 else 2
    n_seeds_robustness = 10 if n_agents >= 1000 else 5

    plan = [
        (1,  "exp1",  exp1,  {}),
        (2,  "exp2",  exp2,  {"archetype_counts": [2, 3, 5, 6]}),
        (3,  "exp3",  exp3,  {}),
        (4,  "exp4",  exp4,  {}),
        (5,  "exp5",  exp5,  {"n_runs": n_bootstrap}),
        (6,  "exp6",  exp6,  {}),
        (7,  "exp7",  exp7,  {}),
        (8,  "exp8",  exp8,  {}),
        (9,  "exp9",  exp9,  {}),
        (10, "exp10", exp10, {"n_runs_per_temp": n_runs_per_temp}),
        (11, "exp11", exp11, {}),
        (12, "exp12", exp12, {"n_seeds": n_seeds_robustness, "use_abm_for_speed": True}),
    ]

    logger.info(
        "resuming run, n_agents={}, start=exp{}, skip={}",
        n_agents, args.start, sorted(skip) or "-",
    )
    t0 = time.time()

    for idx, name, fn, kwargs in plan:
        if idx < args.start or idx in skip:
            logger.info("{}/{}: {} SKIPPED", idx, len(plan), name)
            continue
        logger.info("{}/{}: {} (n={})", idx, len(plan), name, n_agents)
        try:
            fn(seed=args.seed, n_agents=n_agents, **kwargs)
        except Exception:
            logger.exception("failed: {}", name)

    elapsed = time.time() - t0
    logger.info(
        "run-from done, start=exp{}, total {:.1f}s ({:.1f} min)",
        args.start, elapsed, elapsed / 60,
    )

if __name__ == "__main__":
    main()
