"""对三个Hybrid果园场景执行可复现的30随机种子对比实验。"""
import argparse
import csv
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_rrt_star import run_once


SCENARIOS = (
    "hybrid_single_blocker",
    "hybrid_staggered_trees",
    "hybrid_row_corridor",
)
METHODS = ("GoalBias", "Hybrid")
METRICS = (
    "planning_time",
    "first_solution_iteration",
    "node_count",
    "path_length",
    "reverse_length",
    "switch_count",
)


def run_trial(task):
    scenario, method, seed = task
    # 行间通道宽度来源于前面的参数敏感性实验，其余场景保持20 m。
    rectangle_width = 12.0 if scenario == "hybrid_row_corridor" else 20.0
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
    return row


def mean_successful(rows, metric):
    values = [float(row[metric]) for row in rows if row.get("success") and metric in row]
    return statistics.mean(values) if values else float("nan")


def summarize(rows):
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            successful = sum(int(row.get("success", 0)) for row in group)
            item = {
                "scenario": scenario,
                "method": method,
                "trials": len(group),
                "successes": successful,
                "success_rate_percent": 100.0 * successful / len(group),
            }
            for metric in METRICS:
                item[f"mean_{metric}"] = mean_successful(group, metric)
            summary.append(item)
    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(__file__), "benchmark_results"
    ))
    args = parser.parse_args()
    tasks = [
        (scenario, method, seed)
        for scenario in SCENARIOS
        for method in METHODS
        for seed in range(30)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_trial, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{completed:3d}/{len(tasks)}] {row['scenario']} "
                f"{row['method']} seed={row['seed']} success={row.get('success', 0)}",
                flush=True,
            )

    rows.sort(key=lambda row: (
        SCENARIOS.index(row["scenario"]), METHODS.index(row["method"]), row["seed"]
    ))
    detail_fields = ["scenario", "method", "seed", "success", *METRICS]
    detail_path = os.path.join(args.output_dir, "hybrid_30_seeds_detail.csv")
    summary_path = os.path.join(args.output_dir, "hybrid_30_seeds_summary.csv")
    write_csv(detail_path, rows, detail_fields)
    summary = summarize(rows)
    write_csv(summary_path, summary, list(summary[0]))
    print(f"逐次结果: {detail_path}")
    print(f"汇总结果: {summary_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
