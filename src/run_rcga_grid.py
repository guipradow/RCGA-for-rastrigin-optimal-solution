"""Grid-search RCGA parameters and compare the best setting with PSO."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from objectives import OBJECTIVES
from pso import run_pso
from rcga import (
    DIMENSION,
    MAX_GENERATIONS,
    POPULATION_SIZE,
    ZERO_TOLERANCE,
    ConvergencePoint,
    run_rcga,
)


REPORTS_DIR = Path("reports")
CHECKPOINT_GENERATIONS = (1, 50, 100, 200, 500, 1_000, 5_000, 10_000, 50_000, 100_000)
MIN_LOG_VALUE = 1e-12


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def usable_checkpoints(max_generations: int) -> tuple[int, ...]:
    """Return log checkpoints bounded by the configured generation budget."""
    checkpoints = [point for point in CHECKPOINT_GENERATIONS if point <= max_generations]
    if max_generations not in checkpoints:
        checkpoints.append(max_generations)
    return tuple(checkpoints)


def value_at_generation(
    convergence: list[ConvergencePoint],
    generation: int,
    final_fitness: float,
) -> float:
    """Return the best known value at a generation checkpoint."""
    best_value = final_fitness
    for point in convergence:
        if point.generation <= generation:
            best_value = point.best_fitness
        else:
            break
    return best_value


def write_grid_csv(rows: list[dict], reports_dir: Path) -> None:
    """Persist raw grid-search executions."""
    path = reports_dir / "rcga_parameter_grid_runs.csv"
    fieldnames = [
        "cxpb",
        "mutpb",
        "eta_c",
        "eta_m",
        "run",
        "seed",
        "best_fitness",
        "error",
        "generations",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_grid(rows: list[dict]) -> list[dict]:
    """Aggregate the three RCGA runs for every parameter combination."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["cxpb"], row["mutpb"], row["eta_c"], row["eta_m"])
        grouped[key].append(row)

    summary = []
    for (cxpb, mutpb, eta_c, eta_m), combo_rows in grouped.items():
        errors = [row["error"] for row in combo_rows]
        fitnesses = [row["best_fitness"] for row in combo_rows]
        generations = [row["generations"] for row in combo_rows]
        summary.append(
            {
                "cxpb": cxpb,
                "mutpb": mutpb,
                "eta_c": eta_c,
                "eta_m": eta_m,
                "runs": len(combo_rows),
                "best_error": min(errors),
                "median_error": statistics.median(errors),
                "mean_error": statistics.fmean(errors),
                "std_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "worst_error": max(errors),
                "best_fitness": min(fitnesses),
                "mean_generations": statistics.fmean(generations),
                "converged_runs": sum(bool(row["converged"]) for row in combo_rows),
            }
        )

    summary.sort(key=lambda row: (row["median_error"], row["mean_error"], row["best_error"]))
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def write_summary_csv(summary: list[dict], reports_dir: Path) -> None:
    """Persist the aggregated grid-search table as CSV."""
    path = reports_dir / "rcga_parameter_grid_summary.csv"
    fieldnames = [
        "rank",
        "cxpb",
        "mutpb",
        "eta_c",
        "eta_m",
        "runs",
        "best_error",
        "median_error",
        "mean_error",
        "std_error",
        "worst_error",
        "best_fitness",
        "mean_generations",
        "converged_runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def format_number(value: float) -> str:
    """Format compact numeric values for the HTML report."""
    if abs(value) >= 1_000 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def write_html_table(summary: list[dict], reports_dir: Path) -> None:
    """Create an HTML table with the RCGA parameter-grid results."""
    path = reports_dir / "rcga_parameter_grid_results.html"
    headers = [
        "Rank",
        "CXPB",
        "MUTPB",
        "eta_c",
        "eta_m",
        "Runs",
        "Best error",
        "Median error",
        "Mean error",
        "Std error",
        "Worst error",
        "Mean generations",
        "Converged",
    ]
    numeric_keys = [
        "best_error",
        "median_error",
        "mean_error",
        "std_error",
        "worst_error",
        "mean_generations",
    ]

    rows_html = []
    for row in summary:
        values = [
            row["rank"],
            row["cxpb"],
            row["mutpb"],
            row["eta_c"],
            row["eta_m"],
            row["runs"],
            *(format_number(row[key]) for key in numeric_keys),
            f"{row['converged_runs']}/{row['runs']}",
        ]
        class_name = "best" if row["rank"] == 1 else ""
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows_html.append(f"<tr class=\"{class_name}\">{cells}</tr>")

    document = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RCGA parameter grid</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #1f2937;
      background: #f8fafc;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
    }}
    caption {{
      caption-side: top;
      text-align: left;
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    th, td {{
      border: 1px solid #dbe3ef;
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      background: #e8eef7;
      color: #111827;
    }}
    tr.best td {{
      background: #ecfdf5;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <table>
    <caption>Grid de parametros do RCGA para Rastrigin 10D</caption>
    <thead><tr>{''.join(f'<th>{header}</th>' for header in headers)}</tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_comparison_csv(results: list[dict], reports_dir: Path) -> None:
    """Save final RCGA/PSO comparison runs."""
    path = reports_dir / "best_rcga_vs_pso_final_results.csv"
    fieldnames = [
        "algorithm",
        "run",
        "seed",
        "best_fitness",
        "error",
        "generations",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "algorithm": row["algorithm"],
                "run": row["run"],
                "seed": row["seed"],
                "best_fitness": row["best_fitness"],
                "error": row["error"],
                "generations": row["generations"],
                "converged": row["converged"],
            }
            for row in results
        )


def convergence_series(results: list[dict], checkpoints: tuple[int, ...]) -> dict[str, dict]:
    """Build median and min-max convergence series by algorithm."""
    series = {}
    for algorithm in sorted({row["algorithm"] for row in results}):
        algorithm_rows = [row for row in results if row["algorithm"] == algorithm]
        evaluations = []
        medians = []
        lows = []
        highs = []
        for generation in checkpoints:
            values = [
                max(
                    value_at_generation(
                        row["convergence"],
                        generation,
                        row["best_fitness"],
                    ),
                    MIN_LOG_VALUE,
                )
                for row in algorithm_rows
            ]
            evaluations.append(max(generation * POPULATION_SIZE, 1))
            medians.append(statistics.median(values))
            lows.append(min(values))
            highs.append(max(values))
        series[algorithm] = {
            "evaluations": evaluations,
            "median": medians,
            "min": lows,
            "max": highs,
        }
    return series


def draw_algorithm_panel(axis, data: dict, title: str, color: str, marker: str) -> None:
    """Draw one median/envelope convergence panel."""
    axis.fill_between(
        data["evaluations"],
        data["min"],
        data["max"],
        color=color,
        alpha=0.18,
        label="Envelope (min-max)",
    )
    axis.plot(
        data["evaluations"],
        data["median"],
        color=color,
        marker=marker,
        linewidth=2,
        markersize=5,
        label="Mediana",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("Avaliacoes da FO")
    axis.set_ylabel("Melhor f_obj")
    axis.legend(loc="best", fontsize=8, frameon=True)


def write_convergence_plot(
    results: list[dict],
    reports_dir: Path,
    checkpoints: tuple[int, ...],
) -> None:
    """Create a comparison figure with median and min-max envelopes."""
    series = convergence_series(results, checkpoints)
    colors = {"RCGA": "#1f77b4", "PSO": "#f2b134"}
    markers = {"RCGA": "o", "PSO": "^"}

    sns.set_theme(style="whitegrid", context="notebook")
    figure = plt.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 2)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]

    draw_algorithm_panel(axes[0], series["RCGA"], "RCGA", colors["RCGA"], markers["RCGA"])
    draw_algorithm_panel(axes[1], series["PSO"], "PSO", colors["PSO"], markers["PSO"])

    for algorithm in ("RCGA", "PSO"):
        axes[2].plot(
            series[algorithm]["evaluations"],
            series[algorithm]["median"],
            color=colors[algorithm],
            marker=markers[algorithm],
            linewidth=2,
            markersize=5,
            label=algorithm,
        )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Comparacao", fontweight="bold")
    axes[2].set_xlabel("Avaliacoes da FO")
    axes[2].set_ylabel("Melhor f_obj")
    axes[2].legend(loc="best", frameon=True)

    figure.suptitle("Convergencia - Rastrigin 10D (mediana e envelope das runs)", fontweight="bold")
    figure.tight_layout()
    for suffix, fmt in (("svg", "svg"), ("png", "png")):
        figure.savefig(reports_dir / f"best_rcga_vs_pso_convergence.{suffix}", format=fmt, dpi=180)
    plt.close(figure)


def write_error_violin(results: list[dict], algorithm: str, reports_dir: Path) -> None:
    """Create one violin plot for an algorithm's final errors."""
    rows = [row for row in results if row["algorithm"] == algorithm]
    plot_data = {
        "algorithm": [algorithm] * len(rows),
        "error": [row["error"] for row in rows],
    }

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(5.5, 6))
    sns.violinplot(
        data=plot_data,
        x="algorithm",
        y="error",
        inner="quart",
        cut=0,
        color="#8ecae6" if algorithm == "RCGA" else "#ffca3a",
        linewidth=1.4,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="algorithm",
        y="error",
        color="#1f2937",
        jitter=0.08,
        size=4,
        alpha=0.75,
        ax=axis,
    )
    axis.set_title(f"Erro final - {algorithm}")
    axis.set_xlabel("")
    axis.set_ylabel("Erro da funcao objetivo")
    figure.tight_layout()
    safe_algorithm = f"best_{algorithm.lower()}"
    for suffix, fmt in (("svg", "svg"), ("png", "png")):
        figure.savefig(reports_dir / f"{safe_algorithm}_error_violin.{suffix}", format=fmt, dpi=180)
    plt.close(figure)


def run_grid(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Run every RCGA parameter combination three times."""
    objective = OBJECTIVES["rastrigin"]
    grid_rows = []
    combinations = list(product(args.cxpb, args.mutpb, args.eta_c, args.eta_m))
    total_runs = len(combinations) * args.grid_runs
    completed = 0

    for combo_index, (cxpb, mutpb, eta_c, eta_m) in enumerate(combinations, start=1):
        for run_number in range(1, args.grid_runs + 1):
            completed += 1
            seed = args.base_seed + (combo_index - 1) * args.grid_runs + run_number - 1
            result = run_rcga(
                max_generations=args.max_generations,
                zero_tolerance=args.zero,
                seed=seed,
                objective=objective,
                processes=args.processes,
                checkpoints=set(args.checkpoints),
                crossover_probability=cxpb,
                mutation_probability=mutpb,
                eta_c=eta_c,
                eta_m=eta_m,
            )
            error = result.best_fitness - objective.global_minimum
            grid_rows.append(
                {
                    "cxpb": cxpb,
                    "mutpb": mutpb,
                    "eta_c": eta_c,
                    "eta_m": eta_m,
                    "run": run_number,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "error": error,
                    "generations": result.generations,
                    "converged": result.converged,
                }
            )
            print(
                f"Grid {completed:03d}/{total_runs}: "
                f"cxpb={cxpb}, mutpb={mutpb}, eta_c={eta_c}, eta_m={eta_m}, "
                f"run={run_number}, best={result.best_fitness:.6g}"
            )

    return grid_rows, objective


def run_comparison(args: argparse.Namespace, best_params: dict) -> list[dict]:
    """Compare the best RCGA parameter set against canonical PSO."""
    objective = OBJECTIVES["rastrigin"]
    results = []

    for algorithm in ("RCGA", "PSO"):
        for run_number in range(1, args.comparison_runs + 1):
            seed = args.comparison_seed + run_number - 1
            if algorithm == "RCGA":
                result = run_rcga(
                    max_generations=args.max_generations,
                    zero_tolerance=args.zero,
                    seed=seed,
                    objective=objective,
                    processes=args.processes,
                    checkpoints=set(args.checkpoints),
                    crossover_probability=best_params["cxpb"],
                    mutation_probability=best_params["mutpb"],
                    eta_c=best_params["eta_c"],
                    eta_m=best_params["eta_m"],
                )
            else:
                result = run_pso(
                    max_generations=args.max_generations,
                    zero_tolerance=args.zero,
                    seed=seed,
                    objective=objective,
                    processes=args.processes,
                    checkpoints=set(args.checkpoints),
                )
            error = result.best_fitness - objective.global_minimum
            results.append(
                {
                    "algorithm": algorithm,
                    "run": run_number,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "error": error,
                    "generations": result.generations,
                    "converged": result.converged,
                    "convergence": result.convergence,
                }
            )
            print(
                f"{algorithm} comparison run {run_number:02d}/{args.comparison_runs}: "
                f"seed={seed}, best={result.best_fitness:.6g}"
            )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an RCGA parameter grid and compare the best RCGA with PSO."
    )
    parser.add_argument("--grid-runs", type=int, default=3)
    parser.add_argument("--comparison-runs", type=int, default=21)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--comparison-seed", type=int, default=5000)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--zero", type=float, default=ZERO_TOLERANCE)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--cxpb", type=parse_float_list, default=parse_float_list("0.6,0.8"))
    parser.add_argument("--mutpb", type=parse_float_list, default=parse_float_list("0.2,0.4"))
    parser.add_argument("--eta-c", type=parse_float_list, default=parse_float_list("1.0,5.0"))
    parser.add_argument("--eta-m", type=parse_float_list, default=parse_float_list("20.0,50.0"))
    args = parser.parse_args()
    args.checkpoints = usable_checkpoints(args.max_generations)
    return args


def main() -> None:
    """Run reports end to end."""
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    grid_rows, _objective = run_grid(args)
    summary = summarize_grid(grid_rows)
    best_params = summary[0]

    write_grid_csv(grid_rows, args.reports_dir)
    write_summary_csv(summary, args.reports_dir)
    write_html_table(summary, args.reports_dir)

    comparison_results = run_comparison(args, best_params)
    write_comparison_csv(comparison_results, args.reports_dir)
    write_convergence_plot(comparison_results, args.reports_dir, args.checkpoints)
    write_error_violin(comparison_results, "RCGA", args.reports_dir)
    write_error_violin(comparison_results, "PSO", args.reports_dir)

    metadata = {
        "best_rcga_parameters": {
            "cxpb": best_params["cxpb"],
            "mutpb": best_params["mutpb"],
            "eta_c": best_params["eta_c"],
            "eta_m": best_params["eta_m"],
        },
        "grid_runs_per_combination": args.grid_runs,
        "comparison_runs": args.comparison_runs,
        "max_generations": args.max_generations,
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
    }
    (args.reports_dir / "rcga_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best RCGA parameters: {metadata['best_rcga_parameters']}")


if __name__ == "__main__":
    main()
