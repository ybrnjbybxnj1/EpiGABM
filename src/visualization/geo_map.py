from __future__ import annotations

import json
from pathlib import Path

import folium
import pandas as pd
from loguru import logger

_INFECTION_COLORS = [
    (0.0, "gray"),
    (0.01, "#f0e130"),
    (0.11, "#ff8c00"),
    (0.31, "#e60000"),
]

_STATE_COLORS = {
    "S": "#4caf50",
    "E": "#ffc107",
    "I": "#f44336",
    "R": "#9e9e9e",
}

def _infection_color(pct_infected: float) -> str:
    color = "gray"
    for threshold, c in _INFECTION_COLORS:
        if pct_infected >= threshold:
            color = c
    return color

def load_households(config: dict) -> pd.DataFrame:
    pop_dir = Path(config["data"]["population_dir"])
    hh_path = pop_dir / "households.txt"
    if not hh_path.exists():
        logger.warning("households.txt not found at {}", hh_path)
        return pd.DataFrame(columns=["sp_id", "latitude", "longitude"])
    hh = pd.read_csv(hh_path, sep="\t")
    hh = hh[["sp_id", "latitude", "longitude"]]
    hh.index = hh.sp_id
    return hh

def load_geojson(config: dict) -> dict | None:
    geo_path = Path(config["data"]["geo_file"])
    if not geo_path.exists():
        logger.warning("geojson not found at {}", geo_path)
        return None
    with open(geo_path, encoding="utf-8") as f:
        return json.load(f)

def compute_building_stats(
    snapshot_agents: list[dict],
    households: pd.DataFrame,
) -> dict[int, dict]:
    buildings: dict[int, dict] = {}
    for agent in snapshot_agents:
        hh_id = agent["sp_hh_id"]
        if hh_id not in buildings:
            lat = lon = 0.0
            if hh_id in households.index:
                lat = float(households.loc[hh_id, "latitude"])
                lon = float(households.loc[hh_id, "longitude"])
            buildings[hh_id] = {
                "total": 0, "infected": 0, "lat": lat, "lon": lon, "agents": [],
            }
        buildings[hh_id]["total"] += 1
        if agent["state"] in ("E", "I"):
            buildings[hh_id]["infected"] += 1
        buildings[hh_id]["agents"].append(agent)
    for b in buildings.values():
        b["pct_infected"] = b["infected"] / max(b["total"], 1)
    return buildings

def render_map(
    snapshot_agents: list[dict],
    households: pd.DataFrame,
    config: dict,
    geojson: dict | None = None,
    center: tuple[float, float] | None = None,
    zoom_override: int | None = None,
    highlight_hh: int | None = None,
) -> folium.Map:
    viz_cfg = config.get("visualization", {})
    zoom = zoom_override or viz_cfg.get("map_zoom", 15)
    center_lat, center_lon = 59.9415, 30.2728
    if center:
        center_lat, center_lon = center
    elif not households.empty:
        center_lat = float(households.latitude.mean())
        center_lon = float(households.longitude.mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        attr="",
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
             '&copy; <a href="https://carto.com/">CARTO</a>',
        name="CartoDB Positron",
    ).add_to(m)
    building_stats = compute_building_stats(snapshot_agents, households)
    if geojson:
        _add_geojson_buildings(m, geojson, building_stats, households)
    for hh_id, stats in building_stats.items():
        if stats["lat"] == 0.0 and stats["lon"] == 0.0:
            continue
        color = _infection_color(stats["pct_infected"])
        has_infection = stats["infected"] > 0
        is_highlighted = hh_id == highlight_hh
        tooltip = (
            f"house #{hh_id} - {stats['total']} people, "
            f"{stats['infected']} infected"
        )
        border_color = "#2196f3" if is_highlighted else color
        border_weight = 3 if is_highlighted else (1 if has_infection else 0.5)
        opacity = 0.8 if has_infection or is_highlighted else 0.2
        radius = max(3, min(stats["total"], 8)) if has_infection else 2
        folium.CircleMarker(
            location=[stats["lat"], stats["lon"]],
            radius=radius,
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=tooltip,
            popup=_build_popup(hh_id, stats),
        ).add_to(m)
    return m

def _add_geojson_buildings(
    m: folium.Map,
    geojson: dict,
    building_stats: dict[int, dict],
    households: pd.DataFrame,
) -> None:
    folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            "fillColor": "gray",
            "color": "gray",
            "weight": 0.5,
            "fillOpacity": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name"] if "name" in _get_geojson_props(geojson) else [],
            aliases=[""],
            localize=True,
        ),
    ).add_to(m)

def _get_geojson_props(geojson: dict) -> set[str]:
    props: set[str] = set()
    for feature in geojson.get("features", [])[:1]:
        props.update(feature.get("properties", {}).keys())
    return props

def _add_agent_dots(m: folium.Map, stats: dict) -> None:
    base_lat = stats["lat"]
    base_lon = stats["lon"]
    n = len(stats["agents"])
    for i, agent in enumerate(stats["agents"]):
        offset_lat = (i % 3 - 1) * 0.0001
        offset_lon = (i // 3 - n // 6) * 0.0001
        state = agent["state"]
        color = _STATE_COLORS.get(state, "gray")
        folium.CircleMarker(
            location=[base_lat + offset_lat, base_lon + offset_lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"#{agent['sp_id']} {agent['archetype']} [{state}]",
        ).add_to(m)

def _build_popup(hh_id: int, stats: dict) -> str:
    header = (
        f"<b>house #{hh_id}</b> - {stats['total']} people, "
        f"{stats['infected']} infected<br><br>"
    )
    rows = [header]
    for agent in stats["agents"]:
        state = agent["state"]
        color = _STATE_COLORS.get(state, "gray")
        flags = []
        if agent.get("will_isolate"):
            flags.append("изол.")
        if agent.get("wears_mask"):
            flags.append("маска")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        rows.append(
            f'<span style="color:{color}">'
            f'#{agent["sp_id"]} {agent["archetype"]} [{state}]{flag_str}'
            f"</span>"
        )
    return "<br>".join(rows)
