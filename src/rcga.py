"""Real-coded genetic algorithm for the Rastrigin function."""

from __future__ import annotations

import argparse
import multiprocessing
import random
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

from deap import base, creator, tools

from objectives import DEFAULT_OBJECTIVE, OBJECTIVES, ObjectiveFunction


DIMENSION = 10
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0
POPULATION_SIZE = 30
MAX_GENERATIONS = 100_000
ZERO_TOLERANCE = 1e-8

CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.3

# Bibliographic parameters:
# eta_c follows the SBX distribution index used in icannga.pdf.
# eta_m follows the polynomial mutation index discussed in
# 978-3-642-35380-2_1.pdf.
SBX_ETA = 1.0
POLYNOMIAL_MUTATION_ETA = 20.0
TOURNAMENT_SIZE = 3


def evaluate_objective(individual: list[float], objective: ObjectiveFunction) -> tuple[float]:
    """Return a DEAP-compatible single-objective fitness tuple."""
    return (objective.scalar(individual),)


@dataclass(frozen=True)
class RCGAResult:
    """Optimization summary for one RCGA run."""

    best_individual: list[float]
    best_fitness: float
    generations: int
    converged: bool
    convergence: list["ConvergencePoint"]


class ConvergencePoint(NamedTuple):
    """Best fitness observed at one generation."""

    generation: int
    best_fitness: float


def ensure_deap_types() -> None:
    """Create DEAP classes once, even when the module is reloaded."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def build_toolbox(
    dimension: int,
    objective: ObjectiveFunction = DEFAULT_OBJECTIVE,
    eta_c: float = SBX_ETA,
    eta_m: float = POLYNOMIAL_MUTATION_ETA,
) -> base.Toolbox:
    """Configure DEAP primitives for a bounded real-coded GA."""
    ensure_deap_types()

    toolbox = base.Toolbox()
    bounds_low = [objective.lower_bound] * dimension
    bounds_up = [objective.upper_bound] * dimension

    toolbox.register("gene", random.uniform, objective.lower_bound, objective.upper_bound)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=dimension,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate_objective, objective=objective))
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        eta=eta_c,
        low=bounds_low,
        up=bounds_up,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta_m,
        low=bounds_low,
        up=bounds_up,
        indpb=1.0 / dimension,
    )
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    return toolbox


def evaluate_invalid_individuals(population: list, toolbox: base.Toolbox) -> int:
    """Evaluate every individual with invalid fitness and return that count."""
    invalid_individuals = [
        individual for individual in population if not individual.fitness.valid
    ]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)

    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = fitness

    return len(invalid_individuals)


def make_offspring(
    population: list,
    toolbox: base.Toolbox,
    crossover_probability: float = CROSSOVER_PROBABILITY,
    mutation_probability: float = MUTATION_PROBABILITY,
) -> list:
    """Select, clone, recombine, and mutate a new offspring population."""
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for first_child, second_child in zip(offspring[::2], offspring[1::2]):
        if random.random() <= crossover_probability:
            toolbox.mate(first_child, second_child)
            del first_child.fitness.values
            del second_child.fitness.values

    for mutant in offspring:
        if random.random() <= mutation_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    return offspring


def run_rcga(
    *,
    dimension: int = DIMENSION,
    population_size: int = POPULATION_SIZE,
    max_generations: int = MAX_GENERATIONS,
    zero_tolerance: float = ZERO_TOLERANCE,
    seed: int | None = None,
    objective: ObjectiveFunction = DEFAULT_OBJECTIVE,
    processes: int = 1,
    checkpoints: set[int] | None = None,
    crossover_probability: float = CROSSOVER_PROBABILITY,
    mutation_probability: float = MUTATION_PROBABILITY,
    eta_c: float = SBX_ETA,
    eta_m: float = POLYNOMIAL_MUTATION_ETA,
) -> RCGAResult:
    """Optimize an objective until max generations or zero tolerance is reached."""
    if seed is not None:
        random.seed(seed)

    checkpoints = checkpoints or set()
    toolbox = build_toolbox(dimension, objective, eta_c=eta_c, eta_m=eta_m)
    pool = None
    if processes != 1:
        pool = multiprocessing.Pool(processes=processes if processes > 0 else None)
        toolbox.register("map", pool.map)

    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(1)

    try:
        evaluate_invalid_individuals(population, toolbox)
        hall_of_fame.update(population)

        generation = 0
        target_fitness = objective.global_minimum + zero_tolerance
        convergence = [
            ConvergencePoint(
                generation=generation,
                best_fitness=hall_of_fame[0].fitness.values[0],
            )
        ]

        while generation < max_generations:
            best_fitness = hall_of_fame[0].fitness.values[0]
            if best_fitness <= target_fitness:
                break

            generation += 1
            offspring = make_offspring(
                population,
                toolbox,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            evaluate_invalid_individuals(offspring, toolbox)

            population[:] = tools.selBest(population + offspring, population_size)
            hall_of_fame.update(population)

            if generation in checkpoints:
                convergence.append(
                    ConvergencePoint(
                        generation=generation,
                        best_fitness=hall_of_fame[0].fitness.values[0],
                    )
                )

        best = hall_of_fame[0]
        best_fitness = best.fitness.values[0]
        if convergence[-1].generation != generation:
            convergence.append(
                ConvergencePoint(generation=generation, best_fitness=best_fitness)
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return RCGAResult(
        best_individual=list(best),
        best_fitness=best_fitness,
        generations=generation,
        converged=best_fitness <= target_fitness,
        convergence=convergence,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize the Rastrigin function with an RCGA."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--zero", type=float, default=ZERO_TOLERANCE)
    parser.add_argument("--cxpb", type=float, default=CROSSOVER_PROBABILITY)
    parser.add_argument("--mutpb", type=float, default=MUTATION_PROBABILITY)
    parser.add_argument("--eta-c", type=float, default=SBX_ETA)
    parser.add_argument("--eta-m", type=float, default=POLYNOMIAL_MUTATION_ETA)
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes for fitness evaluation; 1 disables multiprocessing, 0 uses all available cores.",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default=DEFAULT_OBJECTIVE.key,
    )
    return parser.parse_args()


def main() -> None:
    """Run one RCGA optimization and print the best result."""
    args = parse_args()
    result = run_rcga(
        dimension=args.dimension,
        population_size=args.population_size,
        max_generations=args.max_generations,
        zero_tolerance=args.zero,
        seed=args.seed,
        objective=OBJECTIVES[args.objective],
        processes=args.processes,
        crossover_probability=args.cxpb,
        mutation_probability=args.mutpb,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
    )

    print(f"Best fitness: {result.best_fitness:.12g}")
    print(f"Generations: {result.generations}")
    print(f"Converged: {result.converged}")
    print(f"Best individual: {result.best_individual}")


if __name__ == "__main__":
    main()
