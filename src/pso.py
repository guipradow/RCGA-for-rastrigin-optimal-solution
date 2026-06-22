"""Canonical particle swarm optimization for benchmark functions."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass

import numpy as np
from pyswarms.single import GlobalBestPSO

from objectives import DEFAULT_OBJECTIVE, ObjectiveFunction
from rcga import (
    DIMENSION,
    POPULATION_SIZE,
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


def run_pso(
    *,
    dimension: int = DIMENSION,
    population_size: int = POPULATION_SIZE,
    max_generations: int,
    zero_tolerance: float,
    seed: int | None = None,
    objective: ObjectiveFunction = DEFAULT_OBJECTIVE,
    processes: int = 1,
    checkpoints: set[int] | None = None,
) -> PSOResult:
    """Optimize an objective with canonical global-best PSO."""
    if seed is not None:
        np.random.seed(seed)

    checkpoints = checkpoints or set()
    bounds = (
        np.full(dimension, objective.lower_bound),
        np.full(dimension, objective.upper_bound),
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
    n_processes = None
    if processes != 1:
        n_processes = multiprocessing.cpu_count() if processes == 0 else processes

    best_fitness, best_individual = optimizer.optimize(
        objective.swarm,
        iters=max_generations,
        n_processes=n_processes,
        verbose=False,
    )

    convergence = []
    target_fitness = objective.global_minimum + zero_tolerance
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
