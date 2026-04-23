from __future__ import annotations

import copy
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from src.models.data_structures import DayResult, EpidemicContext, SimulationResult

STRAINS_KEYS = ["H1N1", "H3N2", "B"]

def _strain_for_inf_index(idx: int) -> str:
    return STRAINS_KEYS[idx - 1]

def _inf_index_for_strain(strain: str) -> int:
    return STRAINS_KEYS.index(strain) + 1

def _func_b_r(inf_day: int) -> float:
    a = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
    if inf_day < 9:
        return a[inf_day - 1]
    return 0.0

_vfunc_b_r = np.vectorize(_func_b_r)

class ABMSimulation:
    def __init__(
        self,
        config: dict,
        data: pd.DataFrame,
        households: pd.DataFrame,
    ) -> None:
        self.config = config
        model_cfg = config["model"]
        self.lmbd: float = model_cfg["lmbd"]
        self.strains = model_cfg["strains"]
        self.infected_init: dict[str, int] = model_cfg["infected_init"]
        self.alpha: dict[str, float] = model_cfg["alpha"]
        self.data = data.copy()
        self.households = households.copy()
        self.dict_school_id_all: dict[str, list[int]] = {}
        self.dict_school_len: list[int] = []
        self.dict_hh_id: dict[str, dict[int, list[int]]] = {}
        self.dict_work_id: dict[str, dict[int, list[int]]] = {}
        self.dict_school_id: dict[str, dict[str, list[int]]] = {}
        self._s_prev: dict[str, int] = {}
        self._prev_total_infected: int = 0

    def set_initial_conditions(self) -> None:
        data = self.data
        for key in self.strains:
            data["susceptible_" + key] = 0
        data["infected"] = 0
        data["illness_day"] = 0
        for key in self.strains:
            n_susceptible = round(len(data) * self.alpha[key])
            idx = np.random.choice(data.index, n_susceptible, replace=False)
            data.loc[idx, "susceptible_" + key] = 1
        for key in self.strains:
            if self.infected_init[key] == 0:
                continue
            susceptible_ids = data[data["susceptible_" + key] == 1].sp_id.values
            chosen = np.random.choice(susceptible_ids, self.infected_init[key], replace=False)
            mask = data.sp_id.isin(chosen)
            data.loc[mask, ["infected", "susceptible_" + key, "illness_day"]] = [
                _inf_index_for_strain(key),
                0,
                3,
            ]
        school_mask = (data.age < 18) & (data.work_id != "X")
        self.dict_school_id_all = {
            str(wid): list(group.index) for wid, group in data[school_mask].groupby("work_id")
        }
        self.dict_school_len = [
            len(self.dict_school_id_all[k]) for k in self.dict_school_id_all
        ]
        infected_children = data[
            (data.infected > 0) & (data.age <= 17) & (data.work_id != "X")
        ]
        for _, row in infected_children.iterrows():
            wid = str(row.work_id)
            if wid in self.dict_school_id_all and row.name in self.dict_school_id_all[wid]:
                self.dict_school_id_all[wid].remove(row.name)
        self.dict_hh_id = {}
        for key in self.strains:
            inf_idx = _inf_index_for_strain(key)
            infected_hh_ids = data.loc[data.infected == inf_idx, "sp_hh_id"].unique()
            self.dict_hh_id[key] = {
                hh_id: list(
                    data[
                        (data.sp_hh_id == hh_id) & (data["susceptible_" + key] == 1)
                    ].index
                )
                for hh_id in infected_hh_ids
            }
        self.dict_work_id = {}
        for key in self.strains:
            inf_idx = _inf_index_for_strain(key)
            infected_workers = data[
                (data.infected == inf_idx) & (data.age > 17) & (data.work_id != "X")
            ]
            self.dict_work_id[key] = {
                int(wid): list(
                    data[
                        (data.age > 17)
                        & (data.work_id == wid)
                        & (data["susceptible_" + key] == 1)
                    ].index
                )
                for wid in infected_workers.work_id.unique()
            }
        self.dict_school_id = {}
        for key in self.strains:
            self.dict_school_id[key] = copy.deepcopy(self.dict_school_id_all)
            immune_children = data[
                (data["susceptible_" + key] == 0)
                & (data.infected == 0)
                & (data.age <= 17)
                & (data.work_id != "X")
            ]
            for _, row in immune_children.iterrows():
                wid = str(row.work_id)
                if wid in self.dict_school_id[key] and row.name in self.dict_school_id[key][wid]:
                    self.dict_school_id[key][wid].remove(row.name)
        for key in self.strains:
            self._s_prev[key] = 0
        logger.info("initial conditions set, {} agents", len(data))

    def behavior_step(self, day: int, context: EpidemicContext) -> None:
        pass

    def get_effective_lmbd(self, agent_idx: int, context: str = "household") -> float:
        return self.lmbd

    def _collect_infected_places(
        self,
    ) -> tuple[
        dict[tuple, list[int]],
        dict[tuple, list[int]],
        dict[tuple, list[int]],
    ]:
        data = self.data
        hh_inf: dict[tuple, list[int]] = defaultdict(list)
        work_inf: dict[tuple, list[int]] = defaultdict(list)
        school_inf: dict[tuple, list[int]] = defaultdict(list)
        virulent = data[(data.infected > 0) & (data.illness_day > 2)]
        for _, row in virulent.iterrows():
            cur_key = _strain_for_inf_index(row.infected)
            hh_inf[row.sp_hh_id, cur_key].append(row.illness_day)
            if row.work_id != "X":
                if row.age > 17:
                    work_inf[row.work_id, cur_key].append(row.illness_day)
                else:
                    school_inf[row.work_id, cur_key].append(row.illness_day)
        return hh_inf, work_inf, school_inf

    def infect_households(self, day: int) -> dict[str, list[int]]:
        data = self.data
        hh_inf, _, _ = self._collected_inf
        result: dict[str, list[int]] = {k: [] for k in self.strains}
        keys_shuffled = list(hh_inf.keys())
        random.shuffle(keys_shuffled)
        for hh_key in keys_shuffled:
            hh_id = hh_key[0]
            cur_strain = hh_key[1]
            if hh_id not in self.dict_hh_id[cur_strain]:
                self.dict_hh_id[cur_strain][hh_id] = list(
                    data[
                        (data.sp_hh_id == hh_id) & (data["susceptible_" + cur_strain] == 1)
                    ].index
                )
            pool = self.dict_hh_id[cur_strain][hh_id]
            hh_len = len(pool)
            if hh_len == 0:
                continue
            temp = _vfunc_b_r(hh_inf[hh_key])
            lmbds = np.array([self.get_effective_lmbd(idx, "household") for idx in pool])
            prob = np.repeat(temp, hh_len) * np.tile(lmbds, len(temp))
            rand_vals = self._x_rand[self._rand_idx : self._rand_idx + len(prob)]
            self._rand_idx += len(prob)
            real_inf = int(np.sum(rand_vals < prob))
            real_inf = min(real_inf, hh_len)
            if real_inf > 0:
                inf_ids = np.random.choice(np.array(pool), real_inf, replace=False)
                result[cur_strain].extend(inf_ids.tolist())
        return result

    def infect_workplaces(self, day: int) -> dict[str, list[int]]:
        data = self.data
        _, work_inf, _ = self._collected_inf
        result: dict[str, list[int]] = {k: [] for k in self.strains}
        some_current = data[(data.work_id != "X") & (data.age > 17)].copy()
        some_current[["work_id"]] = some_current[["work_id"]].astype(int)
        keys_shuffled = list(work_inf.keys())
        random.shuffle(keys_shuffled)
        for work_key in keys_shuffled:
            work_id_raw = work_key[0]
            cur_strain = work_key[1]
            work_id = int(work_id_raw)
            if work_id not in self.dict_work_id[cur_strain]:
                self.dict_work_id[cur_strain][work_id] = list(
                    some_current[
                        (some_current.work_id == work_id)
                        & (some_current["susceptible_" + cur_strain] == 1)
                    ].index
                )
            pool = self.dict_work_id[cur_strain][work_id]
            work_len = len(pool)
            if work_len == 0:
                continue
            temp = _vfunc_b_r(work_inf[work_key])
            lmbds = np.array([self.get_effective_lmbd(idx, "workplace") for idx in pool])
            prob = np.repeat(temp, work_len) * np.tile(lmbds, len(temp))
            rand_vals = self._x_rand[self._rand_idx : self._rand_idx + len(prob)]
            self._rand_idx += len(prob)
            real_inf = int(np.sum(rand_vals < prob))
            real_inf = min(real_inf, work_len)
            if real_inf > 0:
                inf_ids = np.random.choice(np.array(pool), real_inf, replace=False)
                result[cur_strain].extend(inf_ids.tolist())
        return result

    def infect_schools(self, day: int) -> dict[str, list[int]]:
        _, _, school_inf = self._collected_inf
        result: dict[str, list[int]] = {k: [] for k in self.strains}
        keys_shuffled = list(school_inf.keys())
        random.shuffle(keys_shuffled)
        for school_key in keys_shuffled:
            school_id = school_key[0]
            cur_strain = school_key[1]
            pool = self.dict_school_id[cur_strain].get(str(school_id), [])
            school_len = len(pool)
            if school_len == 0:
                continue
            sid = str(school_id)
            length = len(self.dict_school_id_all.get(sid, pool))
            temp = _vfunc_b_r(school_inf[school_key])
            prob_cont = 8.5 / (length - 1) if (8.5 + 1) < length else 1.0
            if pool:
                mean_lmbd = np.mean(
                    [self.get_effective_lmbd(idx, "school") for idx in pool]
                )
            else:
                mean_lmbd = self.lmbd
            res = np.prod(1 - prob_cont * mean_lmbd * temp)
            real_inf = np.random.binomial(length - 1, 1 - res)
            real_inf = min(real_inf, school_len)
            if real_inf > 0:
                inf_ids = np.random.choice(np.array(pool), real_inf, replace=False)
                result[cur_strain].extend(inf_ids.tolist())
        return result

    def update_states(self, day: int) -> dict[str, np.ndarray]:
        data = self.data
        hh_new = self._hh_new
        work_new = self._work_new
        school_new = self._school_new
        real_inf_results: dict[str, np.ndarray] = {}
        for key in self.strains:
            combined = np.array(hh_new[key] + school_new[key] + work_new[key])
            if len(combined) > 0:
                combined = np.unique(combined.astype(int))
            real_inf_results[key] = combined
        for key in self.strains:
            if len(real_inf_results[key]) > 0:
                data.loc[
                    real_inf_results[key],
                    ["infected", "illness_day", "susceptible_" + key],
                ] = [_inf_index_for_strain(key), 1, 0]
        for key in self.strains:
            if len(real_inf_results[key]) == 0:
                continue
            current_hh_members: set[int] = set()
            for members in self.dict_hh_id[key].values():
                current_hh_members.update(members)
            for idx in real_inf_results[key]:
                if idx in current_hh_members:
                    hh_id = data.loc[idx, "sp_hh_id"]
                    if hh_id in self.dict_hh_id[key] and idx in self.dict_hh_id[key][hh_id]:
                        self.dict_hh_id[key][hh_id].remove(idx)
            current_wp_members: set[int] = set()
            for members in self.dict_work_id[key].values():
                current_wp_members.update(members)
            for idx in real_inf_results[key]:
                if idx in current_wp_members:
                    wid = data.loc[idx, "work_id"]
                    try:
                        wid_int = int(wid)
                    except (ValueError, TypeError):
                        continue
                    if wid_int in self.dict_work_id[key] and idx in self.dict_work_id[key][wid_int]:
                        self.dict_work_id[key][wid_int].remove(idx)
            school_infected = real_inf_results[key][
                (data.loc[real_inf_results[key], "work_id"] != "X").values
                & (data.loc[real_inf_results[key], "age"] <= 17).values
            ]
            for idx in school_infected:
                wid = str(data.loc[idx, "work_id"])
                if (
                    wid in self.dict_school_id[key]
                    and idx in self.dict_school_id[key][wid]
                ):
                    self.dict_school_id[key][wid].remove(idx)
        return real_inf_results

    def compute_seir(self, day: int) -> DayResult:
        data = self.data
        n_days_exposed = 2
        s_dict: dict[str, int] = {}
        e_dict: dict[str, int] = {}
        i_dict: dict[str, int] = {}
        r_dict: dict[str, int] = {}
        beta_dict: dict[str, float] = {}
        new_inf_dict: dict[str, int] = {}
        prev_dict: dict[str, int] = {}
        for key in self.strains:
            strain_idx = _inf_index_for_strain(key)
            s_val = int(data["susceptible_" + key].sum())
            e_val = int(
                data[
                    (data.infected == strain_idx) & (data.illness_day <= n_days_exposed)
                ].shape[0]
            )
            i_val = int(
                data[
                    (data.infected == strain_idx) & (data.illness_day > n_days_exposed)
                ].shape[0]
            )
            r_val = int(
                data[
                    (data["susceptible_" + key] == 0) & (data.infected == 0)
                ].shape[0]
            )
            delta_s = s_val - self._s_prev[key]
            if s_val > 0 and i_val > 0:
                beta_val = -delta_s / (s_val * i_val)
            else:
                beta_val = 0.0
            newly_infected = int(
                data[
                    (data.illness_day == 1) & (data.infected == strain_idx)
                ].shape[0]
            )
            prevalence = int(data[data.infected == strain_idx].shape[0])
            s_dict[key] = s_val
            e_dict[key] = e_val
            i_dict[key] = i_val
            r_dict[key] = r_val
            beta_dict[key] = beta_val
            new_inf_dict[key] = newly_infected
            prev_dict[key] = prevalence
            self._s_prev[key] = s_val
        return DayResult(
            day=day,
            S=s_dict,
            E=e_dict,
            I=i_dict,
            R=r_dict,
            beta=beta_dict,
            new_infections=new_inf_dict,
            prevalence=prev_dict,
        )

    def recover(self) -> None:
        data = self.data
        data.loc[data.infected > 0, "illness_day"] += 1
        recovered_mask = data.illness_day > 8
        for key in self.strains:
            data.loc[recovered_mask, "susceptible_" + key] = 0
        data.loc[recovered_mask, ["infected", "illness_day"]] = 0

    def run_day(self, day: int) -> DayResult:
        data = self.data
        total_infected = int((data.infected > 0).sum())
        total_susceptible = int(
            sum(data["susceptible_" + k].sum() for k in self.strains)
        )
        total_recovered = int(
            sum(
                ((data["susceptible_" + k] == 0) & (data.infected == 0)).sum()
                for k in self.strains
            )
        )
        new_today = int((data.illness_day == 1).sum())
        if self._prev_total_infected > 0:
            growth_rate = (total_infected - self._prev_total_infected) / self._prev_total_infected
        else:
            growth_rate = 0.0
        official_measures = self.config.get("model", {}).get("official_measures", [])
        context = EpidemicContext(
            day=day,
            total_infected=total_infected,
            total_susceptible=total_susceptible,
            total_recovered=total_recovered,
            growth_rate=growth_rate,
            new_infections_today=new_today,
            phase=None,
            official_measures=list(official_measures),
        )
        self.behavior_step(day, context)
        has_infectious = len(data[data.illness_day > 2]) > 0
        if has_infectious:
            self._x_rand = np.random.rand(max(len(data) * 100, 100_000))
            self._rand_idx = 0
            self._collected_inf = self._collect_infected_places()
            self._hh_new = self.infect_households(day)
            self._work_new = self.infect_workplaces(day)
            self._school_new = self.infect_schools(day)
            self.update_states(day)
        else:
            for key in self.strains:
                pass
        day_result = self.compute_seir(day)
        self.recover()
        self._prev_total_infected = int((data.infected > 0).sum())
        return day_result

    def run(
        self,
        days: range,
        seed: int,
        early_stop_window: int = 5,
    ) -> SimulationResult:
        np.random.seed(seed)
        random.seed(seed)
        start_time = time.perf_counter()
        self.set_initial_conditions()
        day_list = list(days)
        results: list[DayResult] = []
        epidemic_started = False
        zero_streak = 0
        early_stopped_at: int | None = None
        for day in day_list:
            day_result = self.run_day(day)
            results.append(day_result)
            total_ei = sum(day_result.E.values()) + sum(day_result.I.values())
            if total_ei > 0:
                epidemic_started = True
                zero_streak = 0
            else:
                zero_streak += 1
            total_inf = sum(day_result.prevalence.values())
            if day % 10 == 0 or day == day_list[0]:
                logger.debug("day {}: total infected={}", day, total_inf)
            if (
                epidemic_started
                and zero_streak >= early_stop_window
            ):
                early_stopped_at = day
                logger.info(
                    "early stop at day {} (E=I=0 for {} days)",
                    day, early_stop_window,
                )
                break
        if early_stopped_at is not None and len(results) < len(day_list):
            last = results[-1]
            for day in day_list[len(results):]:
                pad = DayResult(
                    day=day,
                    S=dict(last.S),
                    E={k: 0 for k in last.E},
                    I={k: 0 for k in last.I},
                    R=dict(last.R),
                    beta={k: 0.0 for k in last.beta},
                    new_infections={k: 0 for k in last.new_infections},
                    prevalence={k: 0 for k in last.prevalence},
                    phase=last.phase,
                    n_isolating=0 if last.n_isolating is not None else None,
                    n_masked=0 if last.n_masked is not None else None,
                    n_reducing_contacts=(
                        0 if last.n_reducing_contacts is not None else None
                    ),
                    n_seeing_doctor=0 if last.n_seeing_doctor is not None else None,
                    archetype_rules_updated=False,
                )
                results.append(pad)
        elapsed = time.perf_counter() - start_time
        logger.info(
            "simulation finished, {} days in {:.1f}s (early_stopped_at={})",
            len(results), elapsed, early_stopped_at,
        )
        config = dict(self.config)
        config["early_stopped_at"] = early_stopped_at
        return SimulationResult(
            days=results,
            config=config,
            seed=seed,
            backend=None,
            runtime_seconds=elapsed,
        )
