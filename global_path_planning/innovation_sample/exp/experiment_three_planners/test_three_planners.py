"""RRT*、GoalBias、Hybrid在三个圆形果园场景上的30种子测试与绘图。"""
import argparse
import csv
import math
import os
import sys
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment_rrt_star import run_once


SCENARIOS = {
    "hybrid_single_blocker": "Single blocker",
    "hybrid_staggered_trees": "Staggered trees",
    "hybrid_row_corridor": "Row corridor",
}
METHODS = ("RRT*", "GoalBias", "Hybrid")
COLORS = {"RRT*": "#9AA0A6", "GoalBias": "#4C78A8", "Hybrid": "#E07A5F"}
METRICS = {
    "planning_time": "Planning time (s)",
    "first_solution_iteration": "First-solution iteration",
    "node_count": "Number of nodes",
    "path_length": "Path length (m)",
}


def run_all_trials(seed_count=30):
    rows = []
    total = len(SCENARIOS) * len(METHODS) * seed_count
    completed = 0
    for scenario in SCENARIOS:
        rectangle_width = 12.0 if scenario == "hybrid_row_corridor" else 20.0
        for seed in range(seed_count):
            # 相同场景、相同种子下依次运行三种方法，便于进行配对比较。
            for method in METHODS:
                metrics, *_ = run_once(
                    method=method,
                    seed=seed,
                    env_type=scenario,
                    environment_path=None,
                    rectangle_length=30.0,
                    rectangle_width=rectangle_width,
                    allow_reverse=False,
                )
                row = {"scenario": scenario, "method": method, "seed": seed}
                row.update(metrics)
                rows.append(row)
                completed += 1
                print(
                    f"[{completed:3d}/{total}] {scenario:<24} "
                    f"{method:<8} seed={seed:02d} success={metrics.get('success', 0)}",
                    flush=True,
                )
    return rows


def successful_values(rows, scenario, method, metric):
    return [
        float(row[metric]) for row in rows
        if row["scenario"] == scenario
        and row["method"] == method
        and int(row.get("success", 0)) == 1
        and metric in row
    ]


def build_summary(rows):
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            successes = sum(int(row.get("success", 0)) for row in group)
            item = {
                "scenario": scenario,
                "method": method,
                "trials": len(group),
                "successes": successes,
                "success_rate_percent": 100.0 * successes / len(group),
            }
            for metric in METRICS:
                values = successful_values(rows, scenario, method, metric)
                item[f"mean_{metric}"] = statistics.mean(values) if values else math.nan
                item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else math.nan
                item[f"median_{metric}"] = statistics.median(values) if values else math.nan
            summary.append(item)
    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def configure_figure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def draw_results(rows, output_stem):
    """箱线图展示30种子的完整分布，散点保留每次实验信息。"""
    configure_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    rng = np.random.default_rng(2026)
    scenario_positions = np.arange(len(SCENARIOS)) * 4.2
    offsets = {"RRT*": -0.85, "GoalBias": 0.0, "Hybrid": 0.85}

    for panel_index, (ax, (metric, ylabel)) in enumerate(zip(axes.flat, METRICS.items())):
        for method in METHODS:
            datasets = [
                successful_values(rows, scenario, method, metric)
                for scenario in SCENARIOS
            ]
            positions = scenario_positions + offsets[method]
            box = ax.boxplot(
                datasets,
                positions=positions,
                widths=0.68,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.1},
                whiskerprops={"color": COLORS[method], "linewidth": 0.9},
                capprops={"color": COLORS[method], "linewidth": 0.9},
                boxprops={"edgecolor": COLORS[method], "linewidth": 1.0},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.48)
            for position, values in zip(positions, datasets):
                jitter = rng.normal(0.0, 0.075, len(values))
                ax.scatter(
                    position + jitter, values, s=7, color=COLORS[method],
                    alpha=0.38, linewidths=0, rasterized=True,
                )

        ax.set_xticks(scenario_positions)
        ax.set_xticklabels(SCENARIOS.values())
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
        ax.text(-0.10, 1.04, chr(ord("a") + panel_index), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    handles = [
        mpl.patches.Patch(facecolor=COLORS[method], edgecolor=COLORS[method],
                          alpha=0.6, label=method)
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Planner performance across circular-obstacle orchard scenarios",
                 fontsize=10, y=1.065)
    fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(output_stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "three_planner_results"),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds必须大于0")

    rows = run_all_trials(args.seeds)
    summary = build_summary(rows)
    detail_path = os.path.join(args.output_dir, "three_planners_detail.csv")
    summary_path = os.path.join(args.output_dir, "three_planners_summary.csv")
    figure_stem = os.path.join(args.output_dir, "three_planners_comparison")
    detail_fields = ["scenario", "method", "seed", "success", *METRICS]
    write_csv(detail_path, rows, detail_fields)
    write_csv(summary_path, summary, list(summary[0]))
    draw_results(rows, figure_stem)
    print(f"原始数据: {detail_path}")
    print(f"汇总数据: {summary_path}")
    print(f"对比图: {figure_stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
