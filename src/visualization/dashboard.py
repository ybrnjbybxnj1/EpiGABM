from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import yaml

from src.visualization.map_component import render_animated_map

def _load_config() -> dict:
    with open("config/default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _list_runs(metrics_dir: str) -> list[str]:
    p = Path(metrics_dir)
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("run_*.json")], reverse=True)

def _load_run(metrics_dir: str, run_name: str) -> dict:
    path = Path(metrics_dir) / f"{run_name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _find_run_id(run_data: dict, snapshots_dir: str) -> str:
    run_id = run_data.get("run_id")
    if run_id:
        return run_id
    snap_dir = Path(snapshots_dir)
    candidates = sorted(snap_dir.glob("run_*_day1.json"), reverse=True)
    if candidates:
        parts = candidates[0].stem.split("_")
        if len(parts) >= 3:
            return "_".join(parts[1:-1])
    return ""

def _load_all_snapshots(
    snapshots_dir: str, run_id: str,
) -> dict[int, list[dict]]:
    snap_dir = Path(snapshots_dir)
    result = {}
    for f in sorted(snap_dir.glob(f"run_{run_id}_day*.json")):
        with open(f, encoding="utf-8") as fh:
            snap = json.load(fh)
        result[snap["day"]] = snap.get("agents", [])
    return result

def _load_households(config: dict) -> dict[int, tuple[float, float]]:
    import pandas as pd
    pop_dir = Path(config["data"]["population_dir"])
    hh_path = pop_dir / "households.txt"
    if not hh_path.exists():
        return {}
    hh = pd.read_csv(hh_path, sep="\t")
    return {
        int(row.sp_id): (float(row.latitude), float(row.longitude))
        for _, row in hh.iterrows()
    }

def main() -> None:
    st.set_page_config(page_title="EpiGABM", layout="wide")
    config = _load_config()
    metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics/")
    snapshots_dir = config.get("output", {}).get("snapshots_dir", "results/snapshots/")
    with st.sidebar:
        st.header("EpiGABM")
        runs = _list_runs(metrics_dir)
        if not runs:
            st.warning("no simulation runs found")
            st.stop()
        selected_run = st.selectbox("run", runs)
        run_data = _load_run(metrics_dir, selected_run)
        day_results = run_data.get("days", [])
        st.markdown("---")
        st.write(f"**backend:** {run_data.get('backend', '?')}")
        st.write(f"**days:** {len(day_results)}")
        st.write(f"**seed:** {run_data.get('seed', '?')}")
        total_cost = run_data.get("total_llm_cost_usd", 0)
        st.write(f"**llm cost:** ${total_cost:.2f}")
        runtime = run_data.get("runtime_seconds", 0)
        st.write(f"**runtime:** {runtime:.1f}s")
        st.write(f"**llm calls:** {run_data.get('total_llm_calls', 0)}")
    run_id = _find_run_id(run_data, snapshots_dir)
    all_snapshots = _load_all_snapshots(snapshots_dir, run_id)
    households = _load_households(config)
    if not all_snapshots:
        st.warning("no snapshots found for this run")
        st.stop()
    render_animated_map(
        all_snapshots=all_snapshots,
        households=households,
        day_results=day_results,
        initial_day=min(all_snapshots.keys()),
        height=650,
    )

if __name__ == "__main__":
    main()
