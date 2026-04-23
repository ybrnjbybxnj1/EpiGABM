import time
import traceback

from loguru import logger

def _run_one(name: str, run_fn, **kwargs) -> dict | None:
    logger.info("starting {}", name)
    t0 = time.time()
    try:
        result = run_fn(**kwargs)
        elapsed = time.time() - t0
        logger.info("{} finished in {:.1f}s", name, elapsed)
        return {"status": "ok", "time": round(elapsed, 1), "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("{} failed after {:.1f}s: {}", name, elapsed, e)
        traceback.print_exc()
        return {"status": "error", "time": round(elapsed, 1), "error": str(e)}

def main(n_agents: int = 3000):
    total_start = time.time()
    results = {}

    # snapshots are viz-only and 20x slowdown at 3000 agents, kill them
    import experiments.helpers as _h
    _orig_load = _h.load_config

    def _patched_load(path="config/default.yaml"):
        cfg = _orig_load(path)
        cfg.setdefault("visualization", {})["snapshot_every_n_days"] = 999
        return cfg

    _h.load_config = _patched_load

    logger.info("running all experiments with n_agents={}", n_agents)

    from experiments.exp1_gabm_vs_rulebased import run_experiment as run_exp1
    results["exp1"] = _run_one("exp1_gabm_vs_rulebased", run_exp1, n_agents=n_agents)

    from experiments.exp2_archetype_scaling import run_experiment as run_exp2
    results["exp2"] = _run_one("exp2_archetype_scaling", run_exp2, n_agents=n_agents)

    from experiments.exp3_gpt4_vs_llama import run_experiment as run_exp3
    results["exp3"] = _run_one("exp3_gabm_vs_abm", run_exp3, n_agents=n_agents)

    from experiments.exp4_prompt_sensitivity import run_experiment as run_exp4
    results["exp4"] = _run_one("exp4_prompt_sensitivity", run_exp4, n_agents=n_agents)

    from experiments.exp5_llm_stochasticity import run_experiment as run_exp5
    results["exp5"] = _run_one("exp5_llm_stochasticity", run_exp5, n_agents=n_agents, n_runs=5)

    from experiments.exp6_hybrid_gabm_seir import run_experiment as run_exp6
    results["exp6"] = _run_one("exp6_hybrid_gabm_seir", run_exp6, n_agents=n_agents)

    from experiments.exp7_hmm_switching import run_experiment as run_exp7
    results["exp7"] = _run_one("exp7_hmm_switching", run_exp7, n_agents=n_agents)

    total_time = time.time() - total_start

    print()
    print("all experiments summary")
    for name, r in results.items():
        status = r["status"] if r else "skipped"
        elapsed = r["time"] if r else 0
        print(f"  {name}: {status} ({elapsed:.1f}s)")
    print(f"\n  Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")

if __name__ == "__main__":
    main()
