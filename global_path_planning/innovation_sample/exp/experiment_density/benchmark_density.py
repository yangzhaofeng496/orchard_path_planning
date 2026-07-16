"""障碍物数量密度实验：4个密度×5张地图×10个搜索种子×3种算法。"""
import csv
import os
import statistics
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment_rrt_star import run_once


DENSITIES = (10, 20, 30, 40)
MAP_SEEDS = range(5)
SEARCH_SEEDS = range(10)
METHODS = ("RRT*", "GoalBias", "Hybrid")
COLORS = {"RRT*": "#9AA0A6", "GoalBias": "#4C78A8", "Hybrid": "#E07A5F"}
METRICS = ("planning_time", "first_solution_iteration", "node_count", "path_length")


def run_trial(task):
    density, map_seed, search_seed, method, max_iterations = task
    # 密度地图的搜索矩形保持相同，避免矩形尺寸成为密度实验的混杂因素。
    metrics, *_ = run_once(
        method=method,
        seed=search_seed,
        env_type=f"density_{density}_{map_seed}",
        environment_path=None,
        rectangle_length=30.0,
        rectangle_width=20.0,
        allow_reverse=False,
        max_iterations=max_iterations,
    )
    row = {
        "density": density,
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
    for density in DENSITIES:
        for method in METHODS:
            group = [r for r in rows if r["density"] == density and r["method"] == method]
            item = {"density": density, "method": method, "trials": len(group)}
            item["success_rate_percent"] = 100 * sum(int(r.get("success", 0)) for r in group) / len(group)
            for metric in METRICS:
                values = [float(r[metric]) for r in group if int(r.get("success", 0))]
                item[f"mean_{metric}"] = statistics.mean(values) if values else float("nan")
                item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else float("nan")
                item[f"median_{metric}"] = statistics.median(values) if values else float("nan")
            result.append(item)
    return result


def draw(summary, output_stem):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    labels = {
        "planning_time": "Planning time (s)",
        "first_solution_iteration": "First-solution iteration",
        "node_count": "Number of nodes",
        "path_length": "Path length (m)",
    }
    for ax, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            group = [r for r in summary if r["method"] == method]
            x = [r["density"] for r in group]
            y = [r[f"mean_{metric}"] for r in group]
            e = [r[f"std_{metric}"] for r in group]
            ax.errorbar(x, y, yerr=e, marker="o", linewidth=1.5, capsize=3,
                        color=COLORS[method], label=method)
        ax.set_xlabel("Number of circular obstacles")
        ax.set_ylabel(labels[metric])
        ax.grid(axis="y", color="#D9D9D9", linewidth=.6, alpha=.7)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Density sensitivity of orchard path planners", fontsize=10)
    fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=1500)
    args = parser.parse_args()
    out = os.path.join(os.path.dirname(__file__), "density_results")
    os.makedirs(out, exist_ok=True)
    detail_path = os.path.join(out, "density_detail_checkpoint.csv")
    detail_fields = ["density", "map_seed", "search_seed", "method", "success", *METRICS]
    rows = []
    if os.path.exists(detail_path):
        with open(detail_path, encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for key in ("density", "map_seed", "search_seed"):
                row[key] = int(row[key])
    completed = {
        (row["density"], row["map_seed"], row["search_seed"], row["method"])
        for row in rows
    }
    tasks = [
        (density, map_seed, search_seed, method, args.max_iterations)
        for density in DENSITIES for map_seed in MAP_SEEDS
        for search_seed in SEARCH_SEEDS for method in METHODS
        if (density, map_seed, search_seed, method) not in completed
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_trial, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            write_csv(detail_path, rows, detail_fields)
            print(f"[{index}/{len(tasks)}] density={row['density']} map={row['map_seed']} "
                  f"search={row['search_seed']} {row['method']} success={row.get('success', 0)}",
                  flush=True)
    rows.sort(key=lambda r: (r["density"], r["map_seed"], r["search_seed"], METHODS.index(r["method"])))
    summary = summarize(rows)
    summary_path = os.path.join(out, "density_summary.csv")
    stem = os.path.join(out, "density_sensitivity")
    write_csv(os.path.join(out, "density_detail.csv"), rows, detail_fields)
    write_csv(summary_path, summary, list(summary[0]))
    draw(summary, stem)
    print("逐次结果:", detail_path)
    print("汇总结果:", summary_path)
    print("图表:", stem + ".png/.pdf/.svg")


if __name__ == "__main__":
    main()
