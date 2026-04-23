from __future__ import annotations

import numpy as np
from loguru import logger

from experiments.helpers import load_config, make_llama_gabm, make_small_population
from src.models.data_structures import EpidemicContext
from src.regime.relative_rt_detector import RelativeRtDetector

def diagnose(seed: int = 42, n_agents: int = 3000) -> None:
    config = load_config("config/default.yaml")
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("diagnose: running single gabm for {} agents", n_agents)
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    result = gabm.run(days=days, seed=seed)

    rel_cfg = config.get("regime", {}).get("relative_rt", {})
    det = RelativeRtDetector(population_size=n_agents, config=rel_cfg)

    print("\nday | I(t) | new_inf | prev  | R_t     | ratio | streak | switched")
    
    prev_inf = 0
    switched_day = None
    for dr in result.days:
        total_inf = dr.I.get(strain, 0)
        new_inf = dr.new_infections.get(strain, 0)
        gr = (total_inf - prev_inf) / max(prev_inf, 1) if prev_inf > 0 else 0.0
        ctx = EpidemicContext(
            day=dr.day,
            total_infected=total_inf,
            total_susceptible=dr.S.get(strain, 0),
            total_recovered=dr.R.get(strain, 0),
            growth_rate=gr,
            new_infections_today=new_inf,
            phase=None,
        )
        switched = det.should_switch(ctx)
        idx = dr.day - 1
        rt = det._switch_rt_history[idx] if idx < len(det._switch_rt_history) else float("nan")
        ratio = det._switch_ratio_history[idx] if idx < len(det._switch_ratio_history) else float("nan")
        streak = det._switch_streak_history[idx] if idx < len(det._switch_streak_history) else -1
        prevalence = total_inf / n_agents
        if total_inf == 0 and dr.day > 20:
            continue
        rt_s = f"{rt:7.3f}" if not np.isnan(rt) else "    nan"
        ratio_s = f"{ratio:5.2f}" if not np.isnan(ratio) else "  nan"
        mark = "<- SWITCH" if switched and det._switch_day == dr.day else ""
        print(f"{dr.day:3d} | {total_inf:4d} | {new_inf:7d} | {prevalence:.3f} | {rt_s} | {ratio_s} | {streak:6d} | {mark}")
        if switched and switched_day is None:
            switched_day = dr.day
        prev_inf = total_inf

    print("\nsummary:")
    print(f"  switch_day: {det._switch_day}")
    print(f"  plateau: {det._plateau}")
    print(f"  peak day (max I): {int(np.argmax([dr.I.get(strain, 0) for dr in result.days])) + 1}")
    print(f"  peak I: {max(dr.I.get(strain, 0) for dr in result.days)}")
    valid_rt = [r for r in det._switch_rt_history if not np.isnan(r)]
    print(f"  min R_t (where defined): {min(valid_rt, default=float('nan')):.3f}")
    print(f"  max R_t (where defined): {max(valid_rt, default=float('nan')):.3f}")

if __name__ == "__main__":
    diagnose()
