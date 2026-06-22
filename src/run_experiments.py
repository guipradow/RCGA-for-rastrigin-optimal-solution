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

from objectives import OBJECTIVES, ObjectiveFunction
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
        "objective",
        "algorithm",
        "run",
        "seed",
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
                        "objective": result["objective"],
                        "algorithm": result["algorithm"],
                        "run": result["run"],
                        "seed": result["seed"],
                        "checkpoint_generation": checkpoint,
                        "recorded_generation": recorded_generation,
                        "best_fitness": f"{best_fitness:.16g}",
                        "error": f"{best_fitness - result['global_minimum']:.16g}",
                        "converged": result["converged"],
                    }
                )


def write_final_results(results: list[dict], reports_dir: Path) -> None:
    """Save final objective-function errors for the violin plot and metrics."""
    path = reports_dir / "benchmark_final_results.csv"
    fieldnames = [
        "objective",
        "algorithm",
        "run",
        "seed",
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
                    "objective": result["objective"],
                    "algorithm": result["algorithm"],
                    "run": result["run"],
                    "seed": result["seed"],
                    "best_fitness": f"{result['best_fitness']:.16g}",
                    "error": f"{result['best_fitness'] - result['global_minimum']:.16g}",
                    "generations": result["generations"],
                    "converged": result["converged"],
                    "best_individual": json.dumps(result["best_individual"]),
                }
            )


def write_summary(results: list[dict], reports_dir: Path) -> None:
    """Save the item-2 summary statistics."""
    summary = {}
    for objective in sorted({result["objective"] for result in results}):
        objective_results = [
            result for result in results if result["objective"] == objective
        ]
        summary[objective] = {}
        for algorithm in sorted({result["algorithm"] for result in objective_results}):
            algorithm_results = [
                result
                for result in objective_results
                if result["algorithm"] == algorithm
            ]
            best_fitness_values = [
                result["best_fitness"] for result in algorithm_results
            ]
            global_minimum = algorithm_results[0]["global_minimum"]
            errors = [fitness - global_minimum for fitness in best_fitness_values]
            summary[objective][algorithm] = {
                "runs": len(algorithm_results),
                "dimension": DIMENSION,
                "population_size": POPULATION_SIZE,
                "max_generations": algorithm_results[0]["max_generations"],
                "zero_tolerance": algorithm_results[0]["zero_tolerance"],
                "best": min(best_fitness_values),
                "worst": max(best_fitness_values),
                "mean": statistics.fmean(best_fitness_values),
                "std": statistics.stdev(best_fitness_values)
                if len(best_fitness_values) > 1
                else 0.0,
                "error_best": min(errors),
                "error_worst": max(errors),
                "error_mean": statistics.fmean(errors),
                "error_std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "converged_runs": int(
                    sum(bool(result["converged"]) for result in algorithm_results)
                ),
            }

    path = reports_dir / "benchmark_summary.json"
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")


def write_convergence_plot(
    results: list[dict],
    reports_dir: Path,
    objective: ObjectiveFunction,
) -> None:
    """Save the convergence graph required by item 1."""
    path = reports_dir / f"{objective.key}_benchmark_convergence.svg"
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
    axis.set_title(f"{objective.label}: RCGA x PSO convergence")
    axis.set_xlabel("Generation (log scale)")
    axis.set_ylabel("Best objective value (log scale)")
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)


def write_error_violin_plot(
    results: list[dict],
    reports_dir: Path,
    objective: ObjectiveFunction,
) -> None:
    """Save the violin plot required by item 2."""
    path = reports_dir / f"{objective.key}_benchmark_error_violin.svg"
    plot_data = {"algorithm": [], "error": []}
    stats_lines = []
    for algorithm in sorted({result["algorithm"] for result in results}):
        errors = [
            result["best_fitness"] - objective.global_minimum
            for result in results
            if result["algorithm"] == algorithm
        ]
        plot_data["algorithm"].extend([algorithm] * len(errors))
        plot_data["error"].extend(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
        stats_lines.append(
            f"{algorithm}: best={min(errors):.4g}, worst={max(errors):.4g}, "
            f"mean={statistics.fmean(errors):.4g}, std={std_error:.4g}"
        )

    all_errors = sorted(plot_data["error"])
    error_span = all_errors[-1] - all_errors[0]
    gap_index = max(
        range(len(all_errors) - 1),
        key=lambda index: all_errors[index + 1] - all_errors[index],
        default=0,
    )
    largest_gap = (
        all_errors[gap_index + 1] - all_errors[gap_index]
        if len(all_errors) > 1
        else 0.0
    )
    padding = max(error_span * 0.05, MIN_LOG_VALUE)

    if error_span == 0.0:
        lower_ylim = (max(0.0, all_errors[0] - padding), all_errors[0] + padding)
        upper_ylim = lower_ylim
    elif largest_gap > error_span * 0.2:
        lower_ylim = (max(0.0, all_errors[0] - padding), all_errors[gap_index] + padding)
        upper_ylim = (
            max(0.0, all_errors[gap_index + 1] - padding),
            all_errors[-1] + padding,
        )
    else:
        midpoint = all_errors[0] + error_span * 0.55
        lower_ylim = (max(0.0, all_errors[0] - padding), midpoint)
        upper_ylim = (max(0.0, midpoint - padding), all_errors[-1] + padding)

    sns.set_theme(style="whitegrid", context="talk")
    figure, (upper_axis, lower_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.08},
    )
    for axis in (upper_axis, lower_axis):
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
        axis.set_xlabel("")

    upper_axis.set_ylim(*upper_ylim)
    lower_axis.set_ylim(*lower_ylim)
    upper_axis.set_title(f"{objective.label}: objective error violin plot")
    upper_axis.set_ylabel("Objective error")
    lower_axis.set_ylabel("Objective error (zoom)")
    lower_axis.set_xlabel("")
    upper_axis.text(
        0.98,
        0.95,
        "\n".join(stats_lines),
        transform=upper_axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    upper_axis.spines["bottom"].set_visible(False)
    lower_axis.spines["top"].set_visible(False)
    upper_axis.tick_params(labelbottom=False, bottom=False)
    lower_axis.tick_params(top=False)
    break_marker_size = 0.012
    break_kwargs = dict(color="k", clip_on=False, linewidth=1.1)
    upper_axis.plot(
        (-break_marker_size, +break_marker_size),
        (-break_marker_size, +break_marker_size),
        transform=upper_axis.transAxes,
        **break_kwargs,
    )
    upper_axis.plot(
        (1 - break_marker_size, 1 + break_marker_size),
        (-break_marker_size, +break_marker_size),
        transform=upper_axis.transAxes,
        **break_kwargs,
    )
    lower_axis.plot(
        (-break_marker_size, +break_marker_size),
        (1 - break_marker_size, 1 + break_marker_size),
        transform=lower_axis.transAxes,
        **break_kwargs,
    )
    lower_axis.plot(
        (1 - break_marker_size, 1 + break_marker_size),
        (1 - break_marker_size, 1 + break_marker_size),
        transform=lower_axis.transAxes,
        **break_kwargs,
    )

    figure.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.08)
    figure.savefig(path, format="svg")
    plt.close(figure)


def run_experiments(
    runs: int,
    base_seed: int,
    reports_dir: Path,
    max_generations: int,
    zero_tolerance: float,
    objective_keys: list[str],
    processes: int,
) -> None:
    """Execute repeated benchmark runs and persist assessment reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for objective_key in objective_keys:
        objective = OBJECTIVES[objective_key]
        for run_number in range(1, runs + 1):
            seed = base_seed + run_number - 1
            result = run_rcga(
                max_generations=max_generations,
                zero_tolerance=zero_tolerance,
                seed=seed,
                objective=objective,
                processes=processes,
                checkpoints=set(CHECKPOINTS),
            )
            results.append(
                {
                    "objective": objective.key,
                    "objective_label": objective.label,
                    "global_minimum": objective.global_minimum,
                    "run": run_number,
                    "seed": seed,
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
                f"{objective.label} RCGA run {run_number:02d}/{runs}: "
                f"seed={seed}, "
                f"best={result.best_fitness:.12g}, "
                f"generations={result.generations}, "
                f"converged={result.converged}"
            )

        for run_number in range(1, runs + 1):
            seed = base_seed + run_number - 1
            result = run_pso(
                max_generations=max_generations,
                zero_tolerance=zero_tolerance,
                seed=seed,
                objective=objective,
                processes=processes,
                checkpoints=set(CHECKPOINTS),
            )
            results.append(
                {
                    "objective": objective.key,
                    "objective_label": objective.label,
                    "global_minimum": objective.global_minimum,
                    "run": run_number,
                    "seed": seed,
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
                f"{objective.label} PSO run {run_number:02d}/{runs}: "
                f"seed={seed}, "
                f"best={result.best_fitness:.12g}, "
                f"generations={result.generations}, "
                f"converged={result.converged}"
            )

    write_convergence_report(results, reports_dir)
    write_final_results(results, reports_dir)
    write_summary(results, reports_dir)
    for objective_key in objective_keys:
        objective = OBJECTIVES[objective_key]
        objective_results = [
            result for result in results if result["objective"] == objective.key
        ]
        write_convergence_plot(objective_results, reports_dir, objective)
        write_error_violin_plot(objective_results, reports_dir, objective)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 21 RCGA repetitions and save reports."
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--zero", type=float, default=ZERO_TOLERANCE)
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of RCGA/PSO worker processes; 1 disables multiprocessing, 0 uses all available cores.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=sorted(OBJECTIVES),
        default=list(OBJECTIVES),
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment from the command line."""
    args = parse_args()
    run_experiments(
        runs=args.runs,
        base_seed=args.base_seed,
        reports_dir=args.reports_dir,
        max_generations=args.max_generations,
        zero_tolerance=args.zero,
        objective_keys=args.objectives,
        processes=args.processes,
    )


if __name__ == "__main__":
    main()
