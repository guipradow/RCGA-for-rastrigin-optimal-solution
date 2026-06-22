"""Rastrigin objective function."""

from __future__ import annotations

import math
from collections.abc import Sequence


A_COEFFICIENT = 10.0
GLOBAL_MINIMUM = 0.0


def rastrigin(individual: Sequence[float]) -> float:
    """Return the Rastrigin function value for a real-coded individual."""
    dimension = len(individual)
    return A_COEFFICIENT * dimension + sum(
        x**2 - A_COEFFICIENT * math.cos(2.0 * math.pi * x)
        for x in individual
    )


def evaluate(individual: Sequence[float]) -> tuple[float]:
    """Return a DEAP-compatible single-objective fitness tuple."""
    return (rastrigin(individual),)
