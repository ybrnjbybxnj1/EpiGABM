
from __future__ import annotations

import time

import numpy as np
from loguru import logger

from src.models.abm import ABMSimulation
from src.models.data_structures import DayResult, EpidemicContext, SEIRState, SimulationResult
from src.models.seir import SEIRModel
from src.regime.base import BaseDetector


class HybridModel:
    def __init__(
        self,
        abm: ABMSimulation,
        seir: SEIRModel,
        detector: BaseDetector,
    ) -> None:
        self.abm = abm
        self.seir = seir
        self.detector = detector

    def aggregate_abm_to_seir(self, strain: str) -> SEIRState:
        data = self.abm.data
        n = len(data)
        from src.models.abm import _inf_index_for_strain
        strain_idx = _inf_index_for_strain(strain)
        s = int(data["susceptible_" + strain].sum())
        e = int(
            ((data.infected == strain_idx) & (data.illness_day <= 2)).sum()
        )
        i = int(
            ((data.infected == strain_idx) & (data.illness_day > 2)).sum()
        )
        r = int(
            ((data["susceptible_" + strain] == 0) & (data.infected == 0)).sum()
        )
        return SEIRState(S=s, E=e, I=i, R=r, N=n)

    def run(self, days: range, seed: int) -> SimulationResult:
        import random
        np.random.seed(seed)
        random.seed(seed)
        start_time = time.perf_counter()
        self.abm.set_initial_conditions()
        results: list[DayResult] = []
        switch_day: int | None = None
        day_list = list(days)
        for day in day_list:
            day_result = self.abm.run_day(day)
            results.append(day_result)
            data = self.abm.data
            total_inf = int((data.infected > 0).sum())
            new_today = int((data.illness_day == 1).sum())
            prev_inf = getattr(self, "_prev_inf", 0)
            gr = (total_inf - prev_inf) / prev_inf if prev_inf > 0 else 0.0
            self._prev_inf = total_inf
            context = EpidemicContext(
                day=day,
                total_infected=total_inf,
                total_susceptible=0,
                total_recovered=0,
                growth_rate=gr,
                new_infections_today=new_today,
                phase=None,
            )
            if self.detector.should_switch(context):
                switch_day = day
                logger.info("hybrid switch at day {}", day)
                break
        if switch_day is not None:
            remaining_start = day_list.index(switch_day) + 1
            remaining_days = day_list[remaining_start:]
            if remaining_days:
                seir_by_strain: dict[str, list[SEIRState]] = {}
                avg_betas: dict[str, float] = {}
                for strain in self.abm.strains:
                    seir_init = self.aggregate_abm_to_seir(strain)
                    recent_betas = [r.beta.get(strain, 0.0) for r in results[-5:]]
                    positive_betas = [b for b in recent_betas if b > 0]
                    avg_beta = float(np.mean(positive_betas)) if positive_betas else 0.3
                    avg_betas[strain] = avg_beta
                    seir_states = self.seir.run(
                        days=range(len(remaining_days) + 1),
                        initial=seir_init,
                        beta=avg_beta,
                        stochastic=False,
                    )
                    seir_by_strain[strain] = seir_states
                for i, day_num in enumerate(remaining_days):
                    s_dict, e_dict, i_dict, r_dict = {}, {}, {}, {}
                    beta_dict, new_inf_dict, prev_dict = {}, {}, {}
                    for strain in self.abm.strains:
                        states = seir_by_strain[strain]
                        st = states[i + 1] if i + 1 < len(states) else states[-1]
                        s_dict[strain] = st.S
                        e_dict[strain] = st.E
                        i_dict[strain] = st.I
                        r_dict[strain] = st.R
                        beta_dict[strain] = avg_betas[strain]
                        new_inf_dict[strain] = 0
                        prev_dict[strain] = st.I
                    results.append(DayResult(
                        day=day_num,
                        S=s_dict,
                        E=e_dict,
                        I=i_dict,
                        R=r_dict,
                        beta=beta_dict,
                        new_infections=new_inf_dict,
                        prevalence=prev_dict,
                    ))
        elapsed = time.perf_counter() - start_time
        config = dict(self.abm.config)
        config["switch_day"] = switch_day
        logger.info(
            "hybrid run finished, {} days, switch_day={}, {:.1f}s",
            len(results), switch_day, elapsed,
        )
        return SimulationResult(
            days=results,
            config=config,
            seed=seed,
            runtime_seconds=elapsed,
        )
