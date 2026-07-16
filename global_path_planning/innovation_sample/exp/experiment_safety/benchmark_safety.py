"""车辆安全裕度实验：4个裕度×5张地图×10个搜索种子×3种算法。"""
import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiment_rrt_star import run_once


SAFETY_MARGINS = (0.00, 0.15, 0.30, 0.45)
MAP_SEEDS = range(5)
SEARCH_SEEDS = range(10)
METHODS = ("RRT*", "GoalBias", "Hybrid")
COLORS = {"RRT*": "#9AA0A6", "GoalBias": "#4C78A8", "Hybrid": "#E07A5F"}
METRICS = ("planning_time", "first_solution_iteration", "node_count", "path_length")


def trial(task):
    safety_margin, map_seed, search_seed, method, max_iterations = task
    metrics, *_ = run_once(
        method=method,
        seed=search_seed,
        env_type=f"density_30_{map_seed}",
        environment_path=None,
        rectangle_length=30.0,
        rectangle_width=20.0,
        allow_reverse=False,
        safety_margin=safety_margin,
        max_iterations=max_iterations,
    )
    row = {
        "safety_margin": safety_margin,
        "map_seed": map_seed,
        "search_seed": search_seed,
        "method": method,
    }
    row.update(metrics)
    return row


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    for safety_margin in SAFETY_MARGINS:
        for method in METHODS:
            group = [
                row for row in rows
                if float(row["safety_margin"]) == safety_margin and row["method"] == method
            ]
            item = {"safety_margin": safety_margin, "method": method, "trials": len(group)}
            item["success_rate_percent"] = (
                100.0 * sum(int(row.get("success", 0)) for row in group) / len(group)
            )
            for metric in METRICS:
                values = [float(row[metric]) for row in group if int(row.get("success", 0))]
                item[f"mean_{metric}"] = statistics.mean(values) if values else float("nan")
                item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else float("nan")
                item[f"median_{metric}"] = statistics.median(values) if values else float("nan")
            result.append(item)
    return result


def draw(summary, stem):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    labels = {
        "planning_time": "Planning time (s)",
        "first_solution_iteration": "First-solution iteration",
        "node_count": "Number of nodes",
        "path_length": "Path length (m)",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            group = [row for row in summary if row["method"] == method]
            axis.errorbar(
                [row["safety_margin"] for row in group],
                [row[f"mean_{metric}"] for row in group],
                yerr=[row[f"std_{metric}"] for row in group],
                marker="o", capsize=3, linewidth=1.5,
                color=COLORS[method], label=method,
            )
        axis.set_xlabel("Vehicle safety margin (m)")
        axis.set_ylabel(labels[metric])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    figure.suptitle("Safety-margin sensitivity of orchard path planners", fontsize=10)
    for ext, options in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        figure.savefig(stem + "." + ext, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = os.path.join(os.path.dirname(__file__), "safety_results")
    checkpoint = os.path.join(output, "safety_detail_checkpoint.csv")
    fields = ["safety_margin", "map_seed", "search_seed", "method", "success", *METRICS]
    rows = []
    if os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    completed = {
        (float(row["safety_margin"]), int(row["map_seed"]), int(row["search_seed"]), row["method"])
        for row in rows
    }
    tasks = [
        (margin, map_seed, search_seed, method, args.max_iterations)
        for margin in SAFETY_MARGINS for map_seed in MAP_SEEDS for search_seed in SEARCH_SEEDS
        for method in METHODS
        if (margin, map_seed, search_seed, method) not in completed
    ]
    total = len(SAFETY_MARGINS) * len(MAP_SEEDS) * len(SEARCH_SEEDS) * len(METHODS)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(trial, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            write_csv(checkpoint, rows, fields)
            print(
                f"[{len(completed) + index}/{total}] margin={row['safety_margin']} map={row['map_seed']} "
                f"search={row['search_seed']} {row['method']} success={row.get('success', 0)}",
                flush=True,
            )
    rows.sort(key=lambda row: (
        float(row["safety_margin"]), int(row["map_seed"]), int(row["search_seed"]),
        METHODS.index(row["method"]),
    ))
    summary = summarize(rows)
    write_csv(os.path.join(output, "safety_detail.csv"), rows, fields)
    write_csv(os.path.join(output, "safety_summary.csv"), summary, list(summary[0]))
    draw(summary, os.path.join(output, "safety_sensitivity"))
    print("结果目录:", output)


if __name__ == "__main__":
    main()
