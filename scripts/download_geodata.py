from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

def download_buildings(config_path: str = "config/default.yaml") -> Path:
    import osmnx as ox

    with open(config_path) as f:
        config = yaml.safe_load(f)

    district = config["visualization"]["geo_district"]
    geo_path = Path(config["data"]["geo_file"])
    geo_path.parent.mkdir(parents=True, exist_ok=True)

    place = f"{district} District, Saint Petersburg, Russia"
    logger.info("downloading building footprints for {}", place)

    buildings = ox.features_from_place(place, tags={"building": True})
    buildings.to_file(str(geo_path), driver="GeoJSON")

    logger.info("saved {} buildings to {}", len(buildings), geo_path)
    return geo_path

if __name__ == "__main__":
    download_buildings()
