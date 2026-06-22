"""Run the RCGA experiment required by the MDIOC 2026 exercise."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from rastrigin import GLOBAL_MINIMUM
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
SVG_WIDTH = 960
SVG_HEIGHT = 560
PLOT_MARGIN = 70
MIN_LOG_VALUE = 1e-12


def scale_linear(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    """Scale a numeric value between two intervals."""
    if source_max == source_min:
        return (target_min + target_max) / 2.0
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def svg_header(width: int = SVG_WIDTH, height: int = SVG_HEIGHT) -> list[str]:
    """Return common SVG opening elements."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#222;stroke-width:1.2}'
        '.grid{stroke:#ddd;stroke-width:1}'
        '</style>',
    ]


def save_svg(lines: list[str], path: Path) -> None:
    """Persist SVG markup."""
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def write_convergence_report(results: list[dict], reports_dir: Path) -> None:
    """Save checkpoint convergence data for every run."""
    path = reports_dir / "rcga_convergence.csv"
    fieldnames = [
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
                        "run": result["run"],
                        "seed": result["seed"],
                        "checkpoint_generation": checkpoint,
                        "recorded_generation": recorded_generation,
                        "best_fitness": f"{best_fitness:.16g}",
                        "error": f"{best_fitness - GLOBAL_MINIMUM:.16g}",
                        "converged": result["converged"],
                    }
                )


def write_final_results(results: list[dict], reports_dir: Path) -> None:
    """Save final objective-function errors for the violin plot and metrics."""
    path = reports_dir / "rcga_final_results.csv"
    fieldnames = [
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
                    "run": result["run"],
                    "seed": result["seed"],
                    "best_fitness": f"{result['best_fitness']:.16g}",
                    "error": f"{result['best_fitness'] - GLOBAL_MINIMUM:.16g}",
                    "generations": result["generations"],
                    "converged": result["converged"],
                    "best_individual": json.dumps(result["best_individual"]),
                }
            )


def write_summary(results: list[dict], reports_dir: Path) -> None:
    """Save the item-2 summary statistics."""
    best_fitness_values = [result["best_fitness"] for result in results]
    errors = [fitness - GLOBAL_MINIMUM for fitness in best_fitness_values]
    summary = {
        "runs": len(results),
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "max_generations": results[0]["max_generations"],
        "zero_tolerance": results[0]["zero_tolerance"],
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
        "converged_runs": sum(result["converged"] for result in results),
    }

    path = reports_dir / "rcga_summary.json"
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")


def write_convergence_plot(results: list[dict], reports_dir: Path) -> None:
    """Save the convergence graph required by item 1."""
    path = reports_dir / "rcga_convergence.svg"
    left = PLOT_MARGIN
    right = SVG_WIDTH - 30
    top = 45
    bottom = SVG_HEIGHT - PLOT_MARGIN

    generations = [
        point.generation
        for result in results
        for point in result["convergence"]
        if point.generation > 0
    ]
    fitness_values = [
        max(point.best_fitness, MIN_LOG_VALUE)
        for result in results
        for point in result["convergence"]
    ]
    min_generation = 1
    max_generation = max(generations, default=MAX_GENERATIONS)
    min_log_generation = math.log10(min_generation)
    max_log_generation = math.log10(max_generation)
    min_log_fitness = math.log10(max(min(fitness_values), MIN_LOG_VALUE))
    max_log_fitness = math.log10(max(fitness_values))
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    svg = svg_header()
    svg.extend(
        [
            '<text x="480" y="28" text-anchor="middle" '
            'font-size="20" font-weight="700">RCGA convergence</text>',
            f'<line class="axis" x1="{left}" y1="{bottom}" '
            f'x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{bottom}"/>',
        ]
    )

    x_ticks = [1, 10, 100, 1_000, 10_000, 100_000]
    for tick in x_ticks:
        if tick > max_generation:
            continue
        x = scale_linear(
            math.log10(tick),
            min_log_generation,
            max_log_generation,
            left,
            right,
        )
        svg.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                f'x2="{x:.2f}" y2="{bottom}"/>',
                f'<text x="{x:.2f}" y="{bottom + 24}" '
                f'text-anchor="middle" font-size="12">{tick:g}</text>',
            ]
        )

    y_tick_count = 6
    for index in range(y_tick_count):
        log_value = scale_linear(
            index,
            0,
            y_tick_count - 1,
            min_log_fitness,
            max_log_fitness,
        )
        y = scale_linear(log_value, min_log_fitness, max_log_fitness, bottom, top)
        value = 10**log_value
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end" font-size="12">{value:.1e}</text>',
            ]
        )

    for index, result in enumerate(results):
        points = []
        for point in result["convergence"]:
            generation = max(point.generation, 1)
            x = scale_linear(
                math.log10(generation),
                min_log_generation,
                max_log_generation,
                left,
                right,
            )
            y = scale_linear(
                math.log10(max(point.best_fitness, MIN_LOG_VALUE)),
                min_log_fitness,
                max_log_fitness,
                bottom,
                top,
            )
            points.append(f"{x:.2f},{y:.2f}")

        color = colors[index % len(colors)]
        svg.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="1.6" opacity="0.75"/>'
        )

    svg.extend(
        [
            f'<text x="{(left + right) / 2:.2f}" y="{SVG_HEIGHT - 20}" '
            'text-anchor="middle" font-size="14">Generation (log scale)</text>',
            '<text x="18" y="280" transform="rotate(-90 18 280)" '
            'text-anchor="middle" font-size="14">Best objective value '
            '(log scale)</text>',
        ]
    )
    save_svg(svg, path)


def gaussian_kernel_density(
    values: list[float],
    samples: list[float],
    bandwidth: float,
) -> list[float]:
    """Estimate a one-dimensional Gaussian KDE."""
    normalization = len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    densities = []
    for sample in samples:
        kernel_sum = sum(
            math.exp(-0.5 * ((sample - value) / bandwidth) ** 2)
            for value in values
        )
        densities.append(kernel_sum / normalization)
    return densities


def write_error_violin_plot(results: list[dict], reports_dir: Path) -> None:
    """Save the violin plot required by item 2."""
    path = reports_dir / "rcga_error_violin.svg"
    left = PLOT_MARGIN
    right = SVG_WIDTH - 120
    top = 45
    bottom = SVG_HEIGHT - PLOT_MARGIN
    center_x = (left + right) / 2.0
    max_half_width = 170.0

    errors = [result["best_fitness"] - GLOBAL_MINIMUM for result in results]
    min_error = min(errors)
    max_error = max(errors)
    padding = max((max_error - min_error) * 0.08, MIN_LOG_VALUE)
    y_min = max(0.0, min_error - padding)
    y_max = max_error + padding

    sample_count = 120
    samples = [
        scale_linear(index, 0, sample_count - 1, y_min, y_max)
        for index in range(sample_count)
    ]
    std_dev = statistics.stdev(errors) if len(errors) > 1 else 0.0
    bandwidth = 1.06 * std_dev * (len(errors) ** (-1 / 5)) if std_dev else 1.0
    bandwidth = max(bandwidth, (y_max - y_min) / 100.0, MIN_LOG_VALUE)
    densities = gaussian_kernel_density(errors, samples, bandwidth)
    max_density = max(densities) if densities else 1.0

    right_points = []
    left_points = []
    for sample, density in zip(samples, densities):
        y = scale_linear(sample, y_min, y_max, bottom, top)
        half_width = scale_linear(density, 0.0, max_density, 0.0, max_half_width)
        right_points.append(f"{center_x + half_width:.2f},{y:.2f}")
        left_points.append(f"{center_x - half_width:.2f},{y:.2f}")

    polygon_points = " ".join(right_points + list(reversed(left_points)))
    svg = svg_header()
    svg.extend(
        [
            '<text x="480" y="28" text-anchor="middle" '
            'font-size="20" font-weight="700">Objective error violin plot</text>',
            f'<line class="axis" x1="{left}" y1="{bottom}" '
            f'x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{bottom}"/>',
            f'<polygon points="{polygon_points}" fill="#9ecae1" '
            'stroke="#2171b5" stroke-width="1.5" opacity="0.9"/>',
        ]
    )

    for index, error in enumerate(errors):
        y = scale_linear(error, y_min, y_max, bottom, top)
        jitter = ((index % 7) - 3) * 5.5
        svg.append(
            f'<circle cx="{center_x + jitter:.2f}" cy="{y:.2f}" r="3.5" '
            'fill="#08306b" opacity="0.75"/>'
        )

    mean_error = statistics.fmean(errors)
    y_mean = scale_linear(mean_error, y_min, y_max, bottom, top)
    svg.append(
        f'<line x1="{center_x - max_half_width:.2f}" y1="{y_mean:.2f}" '
        f'x2="{center_x + max_half_width:.2f}" y2="{y_mean:.2f}" '
        'stroke="#cb181d" stroke-width="2"/>'
    )

    for index in range(6):
        value = scale_linear(index, 0, 5, y_min, y_max)
        y = scale_linear(value, y_min, y_max, bottom, top)
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end" font-size="12">{value:.3g}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{center_x:.2f}" y="{bottom + 28}" '
            'text-anchor="middle" font-size="14">RCGA runs</text>',
            '<text x="18" y="280" transform="rotate(-90 18 280)" '
            'text-anchor="middle" font-size="14">Objective error</text>',
            f'<text x="{right + 20}" y="{top + 20}" font-size="12">'
            f'Mean: {mean_error:.4g}</text>',
            f'<text x="{right + 20}" y="{top + 40}" font-size="12">'
            f'Best: {min_error:.4g}</text>',
            f'<text x="{right + 20}" y="{top + 60}" font-size="12">'
            f'Max: {max_error:.4g}</text>',
        ]
    )
    save_svg(svg, path)


def run_experiments(
    runs: int,
    base_seed: int,
    reports_dir: Path,
    max_generations: int,
    zero_tolerance: float,
) -> None:
    """Execute repeated RCGA runs and persist assessment reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for run_number in range(1, runs + 1):
        seed = base_seed + run_number - 1
        result = run_rcga(
            seed=seed,
            max_generations=max_generations,
            zero_tolerance=zero_tolerance,
            checkpoints=set(CHECKPOINTS),
        )
        results.append(
            {
                "run": run_number,
                "seed": seed,
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
            f"Run {run_number:02d}/{runs}: "
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
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--zero", type=float, default=ZERO_TOLERANCE)
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
    )


if __name__ == "__main__":
    main()
