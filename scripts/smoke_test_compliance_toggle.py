"""smoke test: verify apply_compliance_ceilings toggle gates the cap layer.

builds a tiny synthetic agent dataframe, mocks an LLM decision of isolate=True
for every queried archetype, runs apply_decisions_vectorized twice (toggle on
and toggle off), and prints the resulting will_isolate fractions per archetype.

if the gate is wired correctly:
- toggle ON: healthy young_active should clip to ~25% will_isolate (compliance ceiling)
- toggle OFF: healthy young_active should be ~100% (modulo stochastic_noise jitter)

does NOT touch LLM, OpenRouter, or any GABM run. pure unit-level smoke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.archetypes import ArchetypeManager, load_archetypes
from src.agents.base import BehaviorDecision
from src.logging.agent_log import AgentLogger


def _mock_decision() -> BehaviorDecision:
    return BehaviorDecision(
        isolate=True,
        isolate_confidence=0.9,
        mask=True,
        mask_confidence=0.9,
        reduce_contacts=True,
        reduce_contacts_confidence=0.9,
        see_doctor=False,
        see_doctor_confidence=0.1,
        reasoning="smoke-test mock",
    )


def _build_data(n_per_archetype: int = 200) -> pd.DataFrame:
    """500 agents per archetype = 5 archetypes x 200 = 1000."""
    archetypes = ["young_active", "middle_family", "healthcare", "service_worker", "elderly"]
    rows = []
    for arch in archetypes:
        for i in range(n_per_archetype):
            rows.append({
                "sp_id": f"{arch}_{i}",
                "archetype": arch,
                "illness_day": 0,
                "will_isolate": False,
                "wears_mask": False,
                "reduces_contacts": False,
                "sees_doctor": False,
            })
    df = pd.DataFrame(rows)
    df.index = range(len(df))
    return df


def _run(toggle: bool, seed: int = 42) -> dict:
    np.random.seed(seed)
    archetypes = load_archetypes("config/archetypes.yaml")[:6]
    log = AgentLogger(run_id=f"smoke_{int(toggle)}", backend_name="smoke", output_dir="logs")
    manager = ArchetypeManager(
        archetypes=archetypes,
        agent_logger=log,
        agents_config={
            "compliance_rate": {"isolate_healthy": 0.25, "mask": 0.30, "reduce_contacts": 0.40, "see_doctor": 0.20},
            "stochastic_noise": 0.1,
            "apply_compliance_ceilings": toggle,
        },
    )
    data = _build_data(n_per_archetype=200)
    decision = _mock_decision()
    for arch in data["archetype"].unique():
        manager.cached_rules[arch] = decision
    manager.apply_decisions_vectorized(data)
    out = data
    rates = {}
    for arch in out["archetype"].unique():
        sub = out[out["archetype"] == arch]
        rates[arch] = {
            "n": len(sub),
            "isolate_rate": float(sub["will_isolate"].mean()),
            "mask_rate": float(sub["wears_mask"].mean()),
            "contacts_rate": float(sub["reduces_contacts"].mean()),
            "doctor_rate": float(sub["sees_doctor"].mean()),
        }
    return rates


def main() -> None:
    print("ceiling ON (default, toggle=true)")
    on = _run(toggle=True)
    for arch, r in on.items():
        print(f"  {arch:18s} n={r['n']:4d}  isolate={r['isolate_rate']:.3f}  mask={r['mask_rate']:.3f}  contacts={r['contacts_rate']:.3f}")

    print()
    print("ceiling OFF (toggle=false, GABM-pure)")
    off = _run(toggle=False)
    for arch, r in off.items():
        print(f"  {arch:18s} n={r['n']:4d}  isolate={r['isolate_rate']:.3f}  mask={r['mask_rate']:.3f}  contacts={r['contacts_rate']:.3f}")

    print()
    print("verdict")
    for arch in on.keys():
        delta = off[arch]["isolate_rate"] - on[arch]["isolate_rate"]
        print(f"  {arch:18s}  isolate delta (off - on) = {delta:+.3f}")
    if max(off[a]["isolate_rate"] - on[a]["isolate_rate"] for a in on) < 0.4:
        print("  WARNING: no clear gating signal; toggle may be broken")
    else:
        print("  OK: toggle gates the cap layer (off = unclipped, on = clipped near 0.25)")


if __name__ == "__main__":
    main()
