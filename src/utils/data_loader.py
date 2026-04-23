from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

def _sample_whole_households(
    data: pd.DataFrame,
    target_n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hh_sizes = data.groupby("sp_hh_id").size()
    hh_ids = hh_sizes.index.values.copy()
    rng.shuffle(hh_ids)
    selected_hh: list[int] = []
    total = 0
    for hh_id in hh_ids:
        selected_hh.append(hh_id)
        total += hh_sizes[hh_id]
        if total >= target_n:
            break
    result = data[data["sp_hh_id"].isin(selected_hh)].copy()
    result = result.reset_index(drop=True)
    logger.info(
        "subsampled {} agents in {} households (target was {})",
        len(result), len(selected_hh), target_n,
    )
    return result

def load_data(
    population_dir: str | Path,
    frac: float = 1.0,
    n_agents: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    population_dir = Path(population_dir)
    data = pd.read_csv(population_dir / "people.txt", sep="\t")
    data = data[["sp_id", "sp_hh_id", "age", "sex", "work_id"]]
    if n_agents is not None and n_agents < len(data):
        data = _sample_whole_households(data, n_agents, seed)
    elif frac < 1.0:
        data = _sample_whole_households(data, int(len(data) * frac), seed)
    households = pd.read_csv(population_dir / "households.txt", sep="\t")
    households = households[["sp_id", "latitude", "longitude"]]
    school_mask = (data.age < 18) & (data.work_id != "X")
    dict_school_id = {
        str(wid): list(group.index)
        for wid, group in data[school_mask].groupby("work_id")
    }
    logger.info(
        "loaded {} people, {} households, {} schools",
        len(data), len(households), len(dict_school_id),
    )
    return data, households, dict_school_id

def preprocess_data(
    data: pd.DataFrame,
    households: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    households = households.copy()
    data[["sp_id", "sp_hh_id", "age"]] = data[["sp_id", "sp_hh_id", "age"]].astype(int)
    data[["work_id"]] = data[["work_id"]].astype(str)
    households[["sp_id"]] = households[["sp_id"]].astype(int)
    households[["latitude", "longitude"]] = households[["latitude", "longitude"]].astype(float)
    households.index = households.sp_id
    return data, households
