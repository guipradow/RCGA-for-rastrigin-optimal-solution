"""Benchmark objective functions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


ScalarObjective = Callable[[Sequence[float]], float]
SwarmObjective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ObjectiveFunction:
    """Objective metadata used by RCGA and PSO."""

    key: str
    label: str
    lower_bound: float
    upper_bound: float
    global_minimum: float
    scalar: ScalarObjective
    swarm: SwarmObjective


def rastrigin(individual: Sequence[float]) -> float:
    """Return the Rastrigin function value."""
    coefficient = 10.0
    dimension = len(individual)
    return coefficient * dimension + sum(
        x**2 - coefficient * math.cos(2.0 * math.pi * x)
        for x in individual
    )


def rastrigin_swarm(positions: np.ndarray) -> np.ndarray:
    """Evaluate Rastrigin for a swarm position matrix."""
    dimensions = positions.shape[1]
    return 10.0 * dimensions + np.sum(
        positions**2 - 10.0 * np.cos(2.0 * np.pi * positions),
        axis=1,
    )


def rosenbrock(individual: Sequence[float]) -> float:
    """Return the Rosenbrock function value."""
    return sum(
        100.0 * (individual[index + 1] - individual[index] ** 2) ** 2
        + (1.0 - individual[index]) ** 2
        for index in range(len(individual) - 1)
    )


def rosenbrock_swarm(positions: np.ndarray) -> np.ndarray:
    """Evaluate Rosenbrock for a swarm position matrix."""
    return np.sum(
        100.0 * (positions[:, 1:] - positions[:, :-1] ** 2) ** 2
        + (1.0 - positions[:, :-1]) ** 2,
        axis=1,
    )


def de_jong(individual: Sequence[float]) -> float:
    """Return the De Jong sphere function value."""
    return sum(x**2 for x in individual)


def de_jong_swarm(positions: np.ndarray) -> np.ndarray:
    """Evaluate De Jong sphere for a swarm position matrix."""
    return np.sum(positions**2, axis=1)


OBJECTIVES = {
    "rastrigin": ObjectiveFunction(
        key="rastrigin",
        label="Rastrigin",
        lower_bound=-5.12,
        upper_bound=5.12,
        global_minimum=0.0,
        scalar=rastrigin,
        swarm=rastrigin_swarm,
    ),
    "rosenbrock": ObjectiveFunction(
        key="rosenbrock",
        label="Rosenbrock",
        lower_bound=-5.0,
        upper_bound=10.0,
        global_minimum=0.0,
        scalar=rosenbrock,
        swarm=rosenbrock_swarm,
    ),
    "de_jong": ObjectiveFunction(
        key="de_jong",
        label="De Jong",
        lower_bound=-5.12,
        upper_bound=5.12,
        global_minimum=0.0,
        scalar=de_jong,
        swarm=de_jong_swarm,
    ),
}


DEFAULT_OBJECTIVE = OBJECTIVES["rastrigin"]
