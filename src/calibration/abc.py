from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.models.data_structures import PosteriorResult

class ABCSampler:
    def __init__(self, distance_fn: Callable | None = None) -> None:
        self._distance_fn = distance_fn or self._mse

    @staticmethod
    def _mse(sim: pd.Series, target: pd.Series) -> float:
        min_len = min(len(sim), len(target))
        return float(np.mean((sim.values[:min_len] - target.values[:min_len]) ** 2))

    def rejection(
        self,
        run_model: Callable,
        target: pd.Series,
        prior: pd.DataFrame,
        n_samples: int,
        threshold: float,
    ) -> PosteriorResult:
        start = time.perf_counter()
        accepted = []
        n_runs = 0

        samples = prior.sample(n=min(n_samples, len(prior)), replace=True)
        for _, row in samples.iterrows():
            params = row.to_dict()
            sim = run_model(params)
            n_runs += 1
            d = self._distance_fn(sim, target)
            if d < threshold:
                params["_distance"] = d
                accepted.append(params)

        elapsed = time.perf_counter() - start
        accepted_df = pd.DataFrame(accepted) if accepted else pd.DataFrame()

        logger.info("abc rejection: {}/{} accepted in {:.1f}s", len(accepted), n_runs, elapsed)
        return self._build_result(accepted_df, n_runs, elapsed)

    def annealing(
        self,
        run_model: Callable,
        target: pd.Series,
        prior: pd.DataFrame,
        n_samples: int,
        cooling_steps: int = 3,
    ) -> PosteriorResult:
        start = time.perf_counter()
        n_runs = 0
        param_cols = [c for c in prior.columns if not c.startswith("_")]

        initial_particles = prior.sample(n=min(n_samples, len(prior)), replace=True).copy()
        distances = []
        for _, row in initial_particles.iterrows():
            sim = run_model(row.to_dict())
            n_runs += 1
            distances.append(self._distance_fn(sim, target))
        initial_particles["_distance"] = distances

        eps_start = float(initial_particles["_distance"].quantile(0.5))
        eps_end = float(initial_particles["_distance"].quantile(0.1))
        if eps_end <= 0:
            eps_end = eps_start * 0.1
        epsilons = np.geomspace(eps_start, max(eps_end, 1e-10), cooling_steps)

        current = initial_particles
        for step, epsilon in enumerate(epsilons):
            accepted = current[current["_distance"] < epsilon].copy()
            if len(accepted) == 0:
                break
            new_particles = []
            for _ in range(n_samples):
                base = accepted.sample(1).iloc[0]
                perturbed = {}
                for col in param_cols:
                    perturbed[col] = np.clip(
                        base[col] + np.random.uniform(-0.02, 0.02),
                        prior[col].min(), prior[col].max(),
                    )
                sim = run_model(perturbed)
                n_runs += 1
                perturbed["_distance"] = self._distance_fn(sim, target)
                new_particles.append(perturbed)

            current = pd.DataFrame(new_particles)
            logger.debug(
                "annealing step {}: epsilon={:.4f}, {} evaluated",
                step, epsilon, len(current),
            )

        elapsed = time.perf_counter() - start
        logger.info("abc annealing: {} runs in {:.1f}s", n_runs, elapsed)
        return self._build_result(current, n_runs, elapsed)

    def smc(
        self,
        run_model: Callable,
        target: pd.Series,
        prior: pd.DataFrame,
        n_particles: int,
        n_steps: int = 5,
    ) -> PosteriorResult:
        start = time.perf_counter()
        n_runs = 0
        param_cols = [c for c in prior.columns if not c.startswith("_")]

        particles = prior.sample(n=min(n_particles, len(prior)), replace=True).copy()
        distances = []
        for _, row in particles.iterrows():
            sim = run_model(row.to_dict())
            n_runs += 1
            distances.append(self._distance_fn(sim, target))
        particles["_distance"] = distances
        weights = np.ones(len(particles)) / len(particles)

        for step in range(n_steps):
            q = 0.7 - step * (0.6 / max(n_steps - 1, 1))
            epsilon = float(particles["_distance"].quantile(max(q, 0.05)))

            mask = particles["_distance"] < epsilon
            if mask.sum() == 0:
                break
            accepted = particles[mask].copy()
            weights_accepted = weights[mask.values]
            weights_accepted = weights_accepted / weights_accepted.sum()

            param_vals = accepted[param_cols].values
            if len(param_vals) > 1:
                cov = np.cov(param_vals, rowvar=False) * 2.0
                cov += np.eye(len(param_cols)) * 1e-8
            else:
                cov = np.eye(len(param_cols)) * 0.01

            new_particles = []
            new_weights = []
            for _ in range(n_particles):
                idx = np.random.choice(len(accepted), p=weights_accepted)
                base = accepted.iloc[idx]
                base_vals = base[param_cols].values.astype(float)
                perturbed_vals = np.random.multivariate_normal(base_vals, cov)

                perturbed = {}
                for i, col in enumerate(param_cols):
                    perturbed[col] = np.clip(
                        perturbed_vals[i], prior[col].min(), prior[col].max(),
                    )

                sim = run_model(perturbed)
                n_runs += 1
                d = self._distance_fn(sim, target)
                perturbed["_distance"] = d
                new_particles.append(perturbed)
                new_weights.append(1.0)

            particles = pd.DataFrame(new_particles)
            weights = np.array(new_weights)
            weights = weights / weights.sum()

            logger.debug(
                "smc step {}: epsilon={:.4f}, {} particles",
                step, epsilon, len(particles),
            )

        elapsed = time.perf_counter() - start
        logger.info("abc smc: {} runs in {:.1f}s", n_runs, elapsed)
        return self._build_result(particles, n_runs, elapsed)

    def _build_result(
        self, samples: pd.DataFrame, n_runs: int, elapsed: float,
    ) -> PosteriorResult:
        param_cols = [c for c in samples.columns if not c.startswith("_")]

        if len(samples) == 0:
            return PosteriorResult(
                parameters={},
                posterior_samples=pd.DataFrame(),
                metrics={"n_accepted": 0},
                n_model_runs=n_runs,
                runtime_seconds=elapsed,
            )

        point_estimates = {col: float(samples[col].median()) for col in param_cols}
        metrics = {
            "n_accepted": len(samples),
            "mean_distance": float(samples["_distance"].mean()) if "_distance" in samples else 0.0,
        }

        return PosteriorResult(
            parameters=point_estimates,
            posterior_samples=samples[param_cols].copy(),
            metrics=metrics,
            n_model_runs=n_runs,
            runtime_seconds=elapsed,
        )
