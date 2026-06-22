"""Run the RCGA experiment required by the MDIOC 2026 exercise."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from rastrigin import GLOBAL_MINIMUM
from pso import run_pso
from rcga import (
    DIMENSION,
    MAX_GENERATIONS,
    POPULATION_SIZE,
    ZERO_TOLERANCE,
    run_rcga,
)


DEFAULT_RUNS = 21
DEFAULT_REPORTS_DIR = Path("reports")
CHECKPOINTS = (1, 50, 100, 200, 500, 1_000, 5_000, 10_000, 50_000, 100_000)
MIN_LOG_VALUE = 1e-12


def write_convergence_report(results: list[dict], reports_dir: Path) -> None:
    """Save checkpoint convergence data for every run."""
    path = reports_dir / "benchmark_convergence.csv"
    fieldnames = [
        "algorithm",
        "run",
        "checkpoint_generation",
        "recorded_generation",
        "best_fitness",
        "error",
        "converged",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            convergence_by_generation = {
                point.generation: point.best_fitness
                for point in result["convergence"]
            }
            final_generation = result["generations"]
            final_fitness = result["best_fitness"]

            for checkpoint in CHECKPOINTS:
                if checkpoint in convergence_by_generation:
                    recorded_generation = checkpoint
                    best_fitness = convergence_by_generation[checkpoint]
                elif final_generation < checkpoint:
                    recorded_generation = final_generation
                    best_fitness = final_fitness
                else:
                    continue

                writer.writerow(
                    {
                        "algorithm": result["algorithm"],
                        "run": result["run"],
                        "checkpoint_generation": checkpoint,
                        "recorded_generation": recorded_generation,
                        "best_fitness": f"{best_fitness:.16g}",
                        "error": f"{best_fitness - GLOBAL_MINIMUM:.16g}",
                        "converged": result["converged"],
                    }
                )


def write_final_results(results: list[dict], reports_dir: Path) -> None:
    """Save final objective-function errors for the violin plot and metrics."""
    path = reports_dir / "benchmark_final_results.csv"
    fieldnames = [
        "algorithm",
        "run",
        "best_fitness",
        "error",
        "generations",
        "converged",
        "best_individual",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "algorithm": result["algorithm"],
                    "run": result["run"],
                    "best_fitness": f"{result['best_fitness']:.16g}",
                    "error": f"{result['best_fitness'] - GLOBAL_MINIMUM:.16g}",
                    "generations": result["generations"],
                    "converged": result["converged"],
                    "best_individual": json.dumps(result["best_individual"]),
                }
            )


def write_summary(results: list[dict], reports_dir: Path) -> None:
    """Save the item-2 summary statistics."""
    summary = {}
    for algorithm in sorted({result["algorithm"] for result in results}):
        algorithm_results = [
            result for result in results if result["algorithm"] == algorithm
        ]
        best_fitness_values = [
            result["best_fitness"] for result in algorithm_results
        ]
        errors = [fitness - GLOBAL_MINIMUM for fitness in best_fitness_values]
        summary[algorithm] = {
            "runs": len(algorithm_results),
            "dimension": DIMENSION,
            "population_size": POPULATION_SIZE,
            "max_generations": algorithm_results[0]["max_generations"],
            "zero_tolerance": algorithm_results[0]["zero_tolerance"],
            "best": min(best_fitness_values),
            "mean": statistics.fmean(best_fitness_values),
            "max": max(best_fitness_values),
            "std": statistics.stdev(best_fitness_values)
            if len(best_fitness_values) > 1
            else 0.0,
            "error_best": min(errors),
            "error_mean": statistics.fmean(errors),
            "error_max": max(errors),
            "error_std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "converged_runs": int(
                sum(bool(result["converged"]) for result in algorithm_results)
            ),
        }

    path = reports_dir / "benchmark_summary.json"
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")


def write_convergence_plot(results: list[dict], reports_dir: Path) -> None:
    """Save the convergence graph required by item 1."""
    path = reports_dir / "benchmark_convergence.svg"
    plot_data = {
        "algorithm": [],
        "run": [],
        "generation": [],
        "best_fitness": [],
    }
    for result in results:
        for point in result["convergence"]:
            plot_data["algorithm"].append(result["algorithm"])
            plot_data["run"].append(f"{result['algorithm']} {result['run']}")
            plot_data["generation"].append(max(point.generation, 1))
            plot_data["best_fitness"].append(max(point.best_fitness, MIN_LOG_VALUE))

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=plot_data,
        x="generation",
        y="best_fitness",
        hue="algorithm",
        units="run",
        estimator=None,
        linewidth=1.3,
        alpha=0.75,
        legend=True,
        ax=axis,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("RCGA x PSO convergence")
    axis.set_xlabel("Generation (log scale)")
    axis.set_ylabel("Best objective value (log scale)")
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)


def write_error_violin_plot(results: list[dict], reports_dir: Path) -> None:
    """Save the violin plot required by item 2."""
    path = reports_dir / "benchmark_error_violin.svg"
    plot_data = {"algorithm": [], "error": []}
    stats_lines = []
    for algorithm in sorted({result["algorithm"] for result in results}):
        errors = [
            result["best_fitness"] - GLOBAL_MINIMUM
            for result in results
            if result["algorithm"] == algorithm
        ]
        plot_data["algorithm"].extend([algorithm] * len(errors))
        plot_data["error"].extend(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
        stats_lines.append(
            f"{algorithm}: mean={statistics.fmean(errors):.4g}, "
            f"std={std_error:.4g}, best={min(errors):.4g}"
        )

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.violinplot(
        data=plot_data,
        x="algorithm",
        y="error",
        inner=None,
        palette="Set2",
        linewidth=1.5,
        cut=0,
        hue="algorithm",
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="algorithm",
        y="error",
        color="#1f2937",
        jitter=0.12,
        size=5,
        alpha=0.75,
        ax=axis,
    )
    axis.set_title("Objective error violin plot: RCGA x PSO")
    axis.set_xlabel("")
    axis.set_ylabel("Objective error")
    axis.text(
        0.98,
        0.95,
        "\n".join(stats_lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)


def run_experiments(
    runs: int,
    reports_dir: Path,
    max_generations: int,
    zero_tolerance: float,
) -> None:
    """Execute repeated RCGA runs and persist assessment reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for run_number in range(1, runs + 1):
        result = run_rcga(
            max_generations=max_generations,
            zero_tolerance=zero_tolerance,
            checkpoints=set(CHECKPOINTS),
        )
        results.append(
            {
                "run": run_number,
                "algorithm": "RCGA",
                "best_individual": result.best_individual,
                "best_fitness": result.best_fitness,
                "generations": result.generations,
                "converged": result.converged,
                "convergence": result.convergence,
                "max_generations": max_generations,
                "zero_tolerance": zero_tolerance,
            }
        )
        print(
            f"RCGA run {run_number:02d}/{runs}: "
            f"best={result.best_fitness:.12g}, "
            f"generations={result.generations}, "
            f"converged={result.converged}"
        )

    for run_number in range(1, runs + 1):
        result = run_pso(
            max_generations=max_generations,
            zero_tolerance=zero_tolerance,
            checkpoints=set(CHECKPOINTS),
        )
        results.append(
            {
                "run": run_number,
                "algorithm": "PSO",
                "best_individual": result.best_individual,
                "best_fitness": result.best_fitness,
                "generations": result.generations,
                "converged": result.converged,
                "convergence": result.convergence,
                "max_generations": max_generations,
                "zero_tolerance": zero_tolerance,
            }
        )
        print(
            f"PSO run {run_number:02d}/{runs}: "
            f"best={result.best_fitness:.12g}, "
            f"generations={result.generations}, "
            f"converged={result.converged}"
        )

    write_convergence_report(results, reports_dir)
    write_final_results(results, reports_dir)
    write_summary(results, reports_dir)
    write_convergence_plot(results, reports_dir)
    write_error_violin_plot(results, reports_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 21 RCGA repetitions and save reports."
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--zero", type=float, default=ZERO_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    """Run the experiment from the command line."""
    args = parse_args()
    run_experiments(
        runs=args.runs,
        reports_dir=args.reports_dir,
        max_generations=args.max_generations,
        zero_tolerance=args.zero,
    )


if __name__ == "__main__":
    main()
