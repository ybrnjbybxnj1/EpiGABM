from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import yaml

_STATE_BADGE = {
    "S": ("susceptible", "#4caf50"),
    "E": ("exposed", "#ffc107"),
    "I": ("infectious", "#f44336"),
    "R": ("recovered", "#9e9e9e"),
}

def _load_archetype_names() -> dict[str, str]:
    path = Path("config/archetypes.yaml")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {a["id"]: a["name"] for a in data.get("archetypes", [])}

def _load_archetype_stats() -> dict[str, dict]:
    path = Path("config/archetypes.yaml")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        a["id"]: {
            "health_literacy": a.get("health_literacy", 5),
            "risk_tolerance": a.get("risk_tolerance", 5),
        }
        for a in data.get("archetypes", [])
    }

def _generate_daily_narrative(agent: dict, history: list[dict], current_day: int) -> str:
    state = agent.get("state", "S")
    illness_day = agent.get("illness_day", 0)
    archetype = agent.get("archetype", "")
    prev_state = None
    if history:
        earlier = [h for h in history if h["day"] < current_day]
        if earlier:
            prev_state = earlier[-1].get("state")
    just_infected = prev_state == "S" and state in ("E", "I")
    just_recovered = prev_state in ("E", "I") and state == "R"
    just_exposed = prev_state == "S" and state == "E"
    parts = []
    if just_exposed:
        parts.append("Got exposed to the flu today - probably from a contact at work or on the commute.")
    elif just_infected and state == "I":
        parts.append("Woke up feeling awful - fever and body aches. Definitely caught the flu.")
    elif just_recovered:
        parts.append("Finally feeling better today. The worst is over.")
    elif state == "I" and illness_day > 0:
        if illness_day <= 2:
            parts.append(f"Day {illness_day} of being sick. Still feeling terrible.")
        elif illness_day <= 5:
            parts.append(f"Day {illness_day} of illness. Slowly getting better but still weak.")
        else:
            parts.append(f"Day {illness_day}. Dragging on, but should recover soon.")
    elif state == "S":
        parts.append("Healthy so far.")
    elif state == "R":
        parts.append("Recovered and back to normal routine.")
    behaviors = []
    if agent.get("will_isolate"):
        behaviors.append("staying home")
    if agent.get("wears_mask"):
        behaviors.append("wearing a mask")
    if agent.get("reduces_contacts"):
        behaviors.append("avoiding crowds")
    if agent.get("sees_doctor"):
        behaviors.append("planning to see a doctor")
    if behaviors:
        parts.append("Currently: " + ", ".join(behaviors) + ".")
    return " ".join(parts)

def render_agent_inspector(
    agent: dict,
    logs_gpt4: dict | None = None,
    logs_llama: dict | None = None,
    history: list[dict] | None = None,
    current_day: int | None = None,
) -> None:
    arch_names = _load_archetype_names()
    arch_stats = _load_archetype_stats()
    archetype_id = agent.get("archetype", "")
    display_name = arch_names.get(archetype_id, archetype_id)
    state_label, state_color = _STATE_BADGE.get(agent["state"], ("unknown", "gray"))
    st.markdown(f"### agent #{agent['sp_id']} -- {display_name}")
    st.markdown(
        f'<span style="background:{state_color};color:white;padding:2px 8px;'
        f'border-radius:4px">{state_label}</span>',
        unsafe_allow_html=True,
    )
    if agent.get("illness_day", 0) > 0:
        st.write(f"illness day: {agent['illness_day']}")
    stats = arch_stats.get(archetype_id, {})
    hl = stats.get("health_literacy", 5)
    rt = stats.get("risk_tolerance", 5)
    col_hl, col_rt = st.columns(2)
    col_hl.progress(hl / 10, text=f"health literacy: {hl}/10")
    col_rt.progress(rt / 10, text=f"risk tolerance: {rt}/10")
    st.markdown("**decisions:**")
    cols = st.columns(4)
    decisions = [
        ("isolate", agent.get("will_isolate", False)),
        ("mask", agent.get("wears_mask", False)),
        ("reduce", agent.get("reduces_contacts", False)),
        ("doctor", agent.get("sees_doctor", False)),
    ]
    for col, (label, val) in zip(cols, decisions):
        col.metric(label, "yes" if val else "no")
    reasoning = None
    for log in (logs_llama, logs_gpt4):
        if log:
            dec = log.get("parsed_decision") or {}
            r = dec.get("reasoning", "")
            if r:
                reasoning = r
                break
    if reasoning:
        st.markdown("**reasoning:**")
        st.info(reasoning)
    if logs_gpt4 or logs_llama:
        st.markdown("---")
        st.markdown("**GPT-4 vs Llama:**")
        _render_comparison(logs_gpt4, logs_llama)
    filtered_history = history
    if history and current_day is not None:
        filtered_history = [h for h in history if h["day"] <= current_day]
    if current_day is not None:
        narrative = _generate_daily_narrative(agent, filtered_history or [], current_day)
        st.markdown(f"**day {current_day}:**")
        st.caption(narrative)
    if filtered_history:
        st.markdown("**diary:**")
        _render_diary(filtered_history)
    if logs_gpt4:
        with st.expander("full prompt (gpt4)"):
            st.code(logs_gpt4.get("prompt", ""), language=None)
        with st.expander("raw response (gpt4)"):
            st.code(logs_gpt4.get("raw_response", ""), language=None)
    if logs_llama:
        with st.expander("full prompt (llama)"):
            st.code(logs_llama.get("prompt", ""), language=None)
        with st.expander("raw response (llama)"):
            st.code(logs_llama.get("raw_response", ""), language=None)

def _render_comparison(gpt4: dict | None, llama: dict | None) -> None:
    col_g, col_l = st.columns(2)
    gpt4_dec = (gpt4 or {}).get("parsed_decision", {}) or {}
    llama_dec = (llama or {}).get("parsed_decision", {}) or {}
    col_g.markdown("**gpt4**")
    col_l.markdown("**llama**")
    g_reason = gpt4_dec.get("reasoning", "--")
    l_reason = llama_dec.get("reasoning", "--")
    col_g.markdown(f"_{g_reason}_")
    col_l.markdown(f"_{l_reason}_")
    actions = ["isolate", "mask", "reduce_contacts", "see_doctor"]
    for action in actions:
        g_val = gpt4_dec.get(action, False)
        l_val = llama_dec.get(action, False)
        g_conf = gpt4_dec.get(f"{action}_confidence", 0)
        l_conf = llama_dec.get(f"{action}_confidence", 0)
        g_str = f"{'yes' if g_val else 'no'} ({g_conf:.0%})"
        l_str = f"{'yes' if l_val else 'no'} ({l_conf:.0%})"
        differs = g_val != l_val or abs(g_conf - l_conf) > 0.3
        fmt = "**{}**" if differs else "{}"
        col_g.write(f"{action}: {fmt.format(g_str)}")
        col_l.write(f"{action}: {fmt.format(l_str)}")

def _render_timeline(history: list[dict]) -> None:
    if not history:
        return
    parts = []
    prev_state = None
    for entry in history:
        state = entry.get("state", "?")
        day = entry.get("day", "?")
        if state != prev_state:
            _, color = _STATE_BADGE.get(state, ("?", "gray"))
            parts.append(
                f'<span style="color:{color}">day {day}: {state}</span>'
            )
            prev_state = state
    arrow_html = " &rarr; ".join(parts)
    st.markdown(arrow_html, unsafe_allow_html=True)

_TRANSITION_NARRATIVES = {
    ("S", "E"): "Came into contact with someone infected. The virus is now incubating.",
    ("S", "I"): "Got infected and symptoms appeared immediately.",
    ("E", "I"): "Woke up with fever, chills, and body aches. The flu has hit.",
    ("I", "R"): "Finally recovered. Feeling much better today.",
    ("S", "S"): None,
    ("R", "R"): None,
    ("E", "E"): None,
    ("I", "I"): None,
}

def _render_diary(history: list[dict]) -> None:
    if not history:
        return
    for entry in history:
        state = entry.get("state", "?")
        day = entry.get("day", "?")
        narrative = entry.get("narrative", "")
        label, color = _STATE_BADGE.get(state, ("?", "gray"))
        if narrative:
            st.markdown(
                f'<div style="border-left: 3px solid {color}; padding-left: 10px; '
                f'margin-bottom: 8px;">'
                f'<strong style="color:{color}">Day {day}</strong> - {label}<br>'
                f'<span style="color: #888; font-style: italic;">{narrative}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="border-left: 3px solid {color}; padding-left: 10px; '
                f'margin-bottom: 8px;">'
                f'<strong style="color:{color}">Day {day}</strong> - {label}'
                f'</div>',
                unsafe_allow_html=True,
            )

def find_agent_logs(
    logs_dir: str,
    run_prefix: str,
    archetype_id: str,
    sim_day: int,
) -> tuple[dict | None, dict | None]:
    logs_path = Path(logs_dir)
    gpt4_entries: dict[int, dict] = {}
    llama_entries: dict[int, dict] = {}
    for log_file in logs_path.glob(f"run_{run_prefix}*.jsonl"):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("archetype_id") != archetype_id:
                    continue
                entry_day = entry.get("sim_day", -1)
                if entry.get("backend") == "gpt4":
                    gpt4_entries[entry_day] = entry
                elif entry.get("backend") == "llama":
                    llama_entries[entry_day] = entry
    gpt4 = gpt4_entries.get(sim_day) or _nearest_earlier(gpt4_entries, sim_day)
    llama = llama_entries.get(sim_day) or _nearest_earlier(llama_entries, sim_day)
    return gpt4, llama

def _nearest_earlier(entries: dict[int, dict], target_day: int) -> dict | None:
    candidates = [d for d in entries if d <= target_day]
    if not candidates:
        return None
    return entries[max(candidates)]

def build_agent_history(
    snapshots_dir: str,
    run_prefix: str,
    sp_id: int,
) -> list[dict]:
    snap_path = Path(snapshots_dir)
    history: list[dict] = []
    for snap_file in sorted(snap_path.glob(f"run_{run_prefix}*.json")):
        with open(snap_file, encoding="utf-8") as f:
            snap = json.load(f)
        for agent in snap.get("agents", []):
            if agent["sp_id"] == sp_id:
                history.append({
                    "narrative": agent.get("narrative", ""),
                    "day": snap["day"],
                    "state": agent["state"],
                    "illness_day": agent["illness_day"],
                })
                break
    return sorted(history, key=lambda x: x["day"])
