"""Canonical particle swarm optimization for the Rastrigin function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyswarms.single import GlobalBestPSO

from rastrigin import GLOBAL_MINIMUM
from rcga import (
    DIMENSION,
    LOWER_BOUND,
    POPULATION_SIZE,
    UPPER_BOUND,
    ConvergencePoint,
)


CANONICAL_INERTIA = 0.7298
CANONICAL_COGNITIVE = 1.49618
CANONICAL_SOCIAL = 1.49618


@dataclass(frozen=True)
class PSOResult:
    """Optimization summary for one PSO run."""

    best_individual: list[float]
    best_fitness: float
    generations: int
    converged: bool
    convergence: list[ConvergencePoint]


def rastrigin_swarm(positions: np.ndarray) -> np.ndarray:
    """Evaluate Rastrigin for a swarm position matrix."""
    dimensions = positions.shape[1]
    return 10.0 * dimensions + np.sum(
        positions**2 - 10.0 * np.cos(2.0 * np.pi * positions),
        axis=1,
    )


def run_pso(
    *,
    dimension: int = DIMENSION,
    population_size: int = POPULATION_SIZE,
    max_generations: int,
    zero_tolerance: float,
    checkpoints: set[int] | None = None,
) -> PSOResult:
    """Optimize Rastrigin with canonical global-best PSO."""
    checkpoints = checkpoints or set()
    bounds = (
        np.full(dimension, LOWER_BOUND),
        np.full(dimension, UPPER_BOUND),
    )
    options = {
        "c1": CANONICAL_COGNITIVE,
        "c2": CANONICAL_SOCIAL,
        "w": CANONICAL_INERTIA,
    }

    optimizer = GlobalBestPSO(
        n_particles=population_size,
        dimensions=dimension,
        options=options,
        bounds=bounds,
    )
    best_fitness, best_individual = optimizer.optimize(
        rastrigin_swarm,
        iters=max_generations,
        verbose=False,
    )

    convergence = []
    target_fitness = GLOBAL_MINIMUM + zero_tolerance
    best_generation = max_generations
    for index, fitness in enumerate(optimizer.cost_history, start=1):
        if index in checkpoints:
            convergence.append(
                ConvergencePoint(generation=index, best_fitness=float(fitness))
            )
        if fitness <= target_fitness:
            best_generation = index
            break

    if not convergence or convergence[-1].generation != best_generation:
        convergence.append(
            ConvergencePoint(generation=best_generation, best_fitness=float(best_fitness))
        )

    return PSOResult(
        best_individual=best_individual.tolist(),
        best_fitness=float(best_fitness),
        generations=int(best_generation),
        converged=bool(best_fitness <= target_fitness),
        convergence=convergence,
    )
