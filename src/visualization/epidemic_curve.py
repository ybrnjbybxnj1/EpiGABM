from __future__ import annotations

import plotly.graph_objects as go

_PHASE_COLORS = {
    "BASELINE": "rgba(200,200,200,0.15)",
    "GROWTH": "rgba(255,235,59,0.15)",
    "PEAK": "rgba(244,67,54,0.15)",
    "DECLINE": "rgba(76,175,80,0.15)",
}

_SEIR_COLORS = {
    "S": "#4caf50",
    "E": "#ffc107",
    "I": "#f44336",
    "R": "#9e9e9e",
}

def render_epidemic_curve(
    day_results: list[dict],
    current_day: int | None = None,
    switch_day: int | None = None,
    strain: str | None = None,
) -> go.Figure:
    days = [dr["day"] for dr in day_results]
    if strain:
        s_vals = [dr["S"].get(strain, 0) for dr in day_results]
        e_vals = [dr["E"].get(strain, 0) for dr in day_results]
        i_vals = [dr["I"].get(strain, 0) for dr in day_results]
        r_vals = [dr["R"].get(strain, 0) for dr in day_results]
    else:
        s_vals = [sum(dr["S"].values()) for dr in day_results]
        e_vals = [sum(dr["E"].values()) for dr in day_results]
        i_vals = [sum(dr["I"].values()) for dr in day_results]
        r_vals = [sum(dr["R"].values()) for dr in day_results]
    fig = go.Figure()
    for label, vals, color in [
        ("S", s_vals, _SEIR_COLORS["S"]),
        ("E", e_vals, _SEIR_COLORS["E"]),
        ("I", i_vals, _SEIR_COLORS["I"]),
        ("R", r_vals, _SEIR_COLORS["R"]),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=vals, mode="lines",
            name=label, line=dict(color=color, width=2),
        ))
    _add_phase_bands(fig, day_results)
    if current_day is not None:
        fig.add_vline(
            x=current_day, line_dash="dash",
            line_color="black", line_width=1,
            annotation_text=f"day {current_day}",
        )
    if switch_day is not None:
        fig.add_vline(
            x=switch_day, line_dash="dot",
            line_color="purple", line_width=2,
            annotation_text="ABM->SEIR",
        )
    fig.update_layout(
        title="SEIR epidemic curve",
        xaxis_title="day",
        yaxis_title="count",
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

def _add_phase_bands(fig: go.Figure, day_results: list[dict]) -> None:
    if not day_results:
        return
    runs: list[tuple[int, int, str]] = []
    current_phase = day_results[0].get("phase")
    start_day = day_results[0]["day"]
    for dr in day_results[1:]:
        phase = dr.get("phase")
        if phase != current_phase:
            if current_phase:
                runs.append((start_day, dr["day"] - 1, current_phase))
            current_phase = phase
            start_day = dr["day"]
    if current_phase:
        runs.append((start_day, day_results[-1]["day"], current_phase))
    for start, end, phase in runs:
        color = _PHASE_COLORS.get(phase, "rgba(0,0,0,0)")
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, layer="below", line_width=0,
        )
