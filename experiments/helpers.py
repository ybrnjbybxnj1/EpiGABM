from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.agents.archetypes import ArchetypeManager, load_archetypes
from src.agents.base import BaseLLMBackend
from src.logging.agent_log import AgentLogger
from src.models.data_structures import SimulationResult
from src.models.gabm import GABMSimulation

_LlamaBackend = None

def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def make_small_population(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)

    hh_size_probs = [0.30, 0.28, 0.22, 0.13, 0.07]
    hh_sizes = []
    total = 0
    hh_id = 0
    while total < n:
        size = rng.choice([1, 2, 3, 4, 5], p=hh_size_probs)
        size = min(size, n - total)
        hh_sizes.append((hh_id, size))
        total += size
        hh_id += 1

    agent_hh_ids = []
    for hid, size in hh_sizes:
        agent_hh_ids.extend([hid] * size)

    ages = rng.randint(5, 80, n)

    for hid, size in hh_sizes:
        start = sum(s for h, s in hh_sizes if h < hid)
        hh_slice = slice(start, start + size)
        hh_ages = ages[hh_slice]
        if size == 1 and hh_ages[0] < 18:
            ages[start] = rng.randint(20, 50)

    work_ids = []
    for i, age in enumerate(ages):
        if age > 17:
            work_ids.append(str(i // 10))
        else:
            work_ids.append(str(100 + age // 3) if age >= 7 else "X")

    data = pd.DataFrame({
        "sp_id": range(1000, 1000 + n),
        "sp_hh_id": agent_hh_ids,
        "age": ages,
        "sex": rng.choice(["M", "F"], n),
        "work_id": work_ids,
    })

    n_hh = len(hh_sizes)
    households = pd.DataFrame({
        "sp_id": [hid for hid, _ in hh_sizes],
        "latitude": rng.uniform(59.93, 59.95, n_hh),
        "longitude": rng.uniform(30.25, 30.30, n_hh),
    })

    return data, households

def make_gabm(
    config: dict,
    data: pd.DataFrame,
    households: pd.DataFrame,
    backend: BaseLLMBackend,
    n_archetypes: int | None = None,
) -> GABMSimulation:
    archetype_path = config.get("data", {}).get(
        "archetypes_file", "config/archetypes.yaml",
    )
    archetypes = load_archetypes(archetype_path)
    if n_archetypes is not None:
        archetypes = archetypes[:n_archetypes]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = config.get("output", {}).get("logs_dir", "results/logs/")
    agent_logger = AgentLogger(run_id=ts, backend_name="mock", output_dir=log_dir)

    agents_cfg = dict(config.get("agents", {}))
    agents_cfg["stochastic_noise"] = 0.0

    manager = ArchetypeManager(
        archetypes=archetypes,
        agent_logger=agent_logger,
        agents_config=agents_cfg,
    )

    return GABMSimulation(
        config=config,
        data=data,
        households=households,
        llm_backend=backend,
        archetype_manager=manager,
    )

def make_ollama_gabm(
    config: dict,
    data: pd.DataFrame,
    households: pd.DataFrame,
    backend_key: str = "llama",
    n_archetypes: int | None = None,
) -> GABMSimulation:
    global _LlamaBackend
    if _LlamaBackend is None:
        from src.agents.backends.llama import LlamaBackend
        _LlamaBackend = LlamaBackend

    backend_cfg = config["agents"]["backends"][backend_key]
    backend = _LlamaBackend(backend_cfg)

    archetype_path = config.get("data", {}).get(
        "archetypes_file", "config/archetypes.yaml",
    )
    archetypes = load_archetypes(archetype_path)
    if n_archetypes is not None:
        archetypes = archetypes[:n_archetypes]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = config.get("output", {}).get("logs_dir", "results/logs/")
    log_name = backend_cfg["model"].replace(":", "_").replace(".", "_")
    agent_logger = AgentLogger(run_id=ts, backend_name=log_name, output_dir=log_dir)

    agents_cfg = dict(config.get("agents", {}))
    agents_cfg["stochastic_noise"] = 0.1

    manager = ArchetypeManager(
        archetypes=archetypes,
        agent_logger=agent_logger,
        agents_config=agents_cfg,
    )

    return GABMSimulation(
        config=config,
        data=data,
        households=households,
        llm_backend=backend,
        archetype_manager=manager,
    )

def make_llama_gabm(
    config: dict,
    data: pd.DataFrame,
    households: pd.DataFrame,
    n_archetypes: int | None = None,
) -> GABMSimulation:
    return make_ollama_gabm(config, data, households, "llama", n_archetypes)

def compute_metrics(result: SimulationResult, strain: str = "H1N1") -> dict:
    prevalences = [dr.prevalence.get(strain, 0) for dr in result.days]
    new_infections = [dr.new_infections.get(strain, 0) for dr in result.days]
    days = [dr.day for dr in result.days]

    peak_idx = int(np.argmax(prevalences))
    peak_day = days[peak_idx]
    peak_mag = prevalences[peak_idx]
    total_inf = sum(new_infections)

    r_effs = []
    for i in range(1, len(new_infections)):
        if prevalences[i - 1] > 0:
            r_effs.append(new_infections[i] / prevalences[i - 1])
    r_eff_mean = float(np.mean(r_effs)) if r_effs else 0.0

    threshold = peak_mag * 0.1
    waves = 0
    in_wave = False
    for p in prevalences:
        if p > threshold and not in_wave:
            waves += 1
            in_wave = True
        elif p < threshold:
            in_wave = False

    n_updates = sum(1 for dr in result.days if dr.archetype_rules_updated)

    return {
        "peak_day": peak_day,
        "peak_magnitude": peak_mag,
        "total_infected": total_inf,
        "rmse_vs_observed": None,
        "r_eff_mean": round(r_eff_mean, 4),
        "wave_count": waves,
        "n_behavior_updates": n_updates,
        "runtime_seconds": round(result.runtime_seconds, 2),
        "llm_calls": result.total_llm_calls,
        "llm_cost_usd": round(result.total_llm_cost_usd, 4),
    }

def save_experiment_json(
    experiment_name: str,
    seed: int,
    config: dict,
    variants: dict,
    comparison: dict | None = None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    config_hash = hashlib.md5(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

    output = {
        "experiment": experiment_name,
        "timestamp": ts,
        "seed": seed,
        "config_hash": config_hash,
        "variants": variants,
    }
    if comparison:
        output["comparison"] = comparison

    out_dir = Path(config.get("output", {}).get("metrics_dir", "results/metrics/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{experiment_name}_{ts.replace(':', '')}.json"

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("saved experiment results to {}", path)
    return path

def save_figure(fig: object, name: str, config: dict) -> Path:
    out_dir = Path(config.get("output", {}).get("figures_dir", "results/figures/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("saved figure to {}", path)
    return path
