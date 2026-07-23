#!/usr/bin/env python3
"""Batch benchmark global planners on every NPZ below experiment/map."""
from __future__ import annotations

import argparse
import csv
import io
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from _planner_common import (
    CollisionChecker, minimum_clearance, path_length, postprocess, select_pairs,
)
from a_star_planner import plan as plan_astar
from bit_star_planner import plan as plan_bit_star
from improved_rrt_star_planner import plan as plan_gaskpo_rrt
from informed_rrt_star_planner import plan as plan_informed_rrt_star
from rrt_star_planner import plan as plan_rrt_star
from theta_star_planner import plan as plan_theta_star
from global_path_planning.innovation_sample.orchard_environment import load_environment
from vehicle.vehicle_collision import VehicleGeometry


METHODS = (
    ("A*", plan_astar),
    ("Theta*", plan_theta_star),
    ("RRT*", plan_rrt_star),
    ("GASKPO-RRT* (ours)", plan_gaskpo_rrt),
    ("Informed RRT*", plan_informed_rrt_star),
    ("BIT*", plan_bit_star),
)
METRICS = (
    "planning_time_s", "total_time_s", "path_length_m", "node_count",
    "first_solution_iteration", "path_point_count", "detour_ratio",
    "min_obstacle_distance_m",
)


def load_environment_quiet(path):
    """Load a map without letting per-task loader messages break tqdm output."""
    with redirect_stdout(io.StringIO()):
        return load_environment(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trial_seed(base_seed, map_index, start_index, goal_index, repeat):
    """Match Ablation's random_seed = base_seed + trial_index protocol."""
    del map_index, start_index, goal_index
    return int(base_seed + repeat)


def run_task(task):
    method_index, map_index, map_path, density, start_i, goal_i, start, goal, repeat, config = task
    method, planner = METHODS[method_index]
    seed = trial_seed(config.seed, map_index, start_i, goal_i, repeat)
    row = {
        "density": density, "map": Path(map_path).name, "map_path": str(Path(map_path).resolve()),
        "method": method, "is_ours": int(method.startswith("GASKPO-RRT*")),
        "start_index": start_i, "goal_index": goal_i, "repeat": repeat, "seed": seed,
    }
    try:
        env = load_environment_quiet(map_path)
        vehicle = VehicleGeometry(config.front_length, config.rear_length,
                                  config.vehicle_width, config.safety_distance)
        checker = CollisionChecker(vehicle, env.obstacles, env.bounds, config.collision_resolution)
        begin = time.perf_counter()
        planning_begin = time.perf_counter()
        if checker.point_free(start) and checker.point_free(goal):
            result = planner(start, goal, env.bounds, checker, env.obstacles, seed, config)
        else:
            result = SimpleNamespace(path=None, nodes=0, first_iteration=-1)
        measured_planning_time = time.perf_counter() - planning_begin
        internal_planning_time = getattr(result, "planning_time_s", None)
        planning_time = (internal_planning_time
                         if internal_planning_time is not None
                         else measured_planning_time)
        final_path = result.path
        if final_path and config.postprocess:
            final_path = postprocess(final_path, checker, seed, config.shortcut_iterations,
                                     config.smooth_spacing)
        total_time = time.perf_counter() - begin
        success = int(final_path is not None)
        length = path_length(final_path) if success else math.nan
        direct = math.dist(start, goal)
        row.update({
            "success": success, "planning_time_s": planning_time, "total_time_s": total_time,
            "path_length_m": length, "node_count": result.nodes,
            "first_solution_iteration": result.first_iteration,
            "path_point_count": len(final_path) if success else 0,
            "detour_ratio": length / direct if success and direct > 1e-12 else math.nan,
            "min_obstacle_distance_m": minimum_clearance(final_path, env.obstacles, vehicle) if success else math.nan,
            "error": "",
        })
    except Exception as exc:
        row.update({"success": 0, **{metric: math.nan for metric in METRICS},
                    "node_count": 0, "first_solution_iteration": -1,
                    "path_point_count": 0, "error": f"{type(exc).__name__}: {exc}"})
    return row


def summarize_group(rows, scope, density="ALL", map_name="ALL"):
    successful = [row for row in rows if row["success"]]
    result = {
        "scope": scope, "density": density, "map": map_name,
        "method": rows[0]["method"], "is_ours": rows[0]["is_ours"],
        "trials": len(rows), "successes": len(successful),
        "success_rate_percent": 100.0 * len(successful) / len(rows),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in successful if math.isfinite(float(row[metric]))]
        result[f"mean_{metric}"] = float(np.mean(values)) if values else math.nan
        result[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else math.nan
    return result


def build_summaries(rows):
    output = []
    for method, _ in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        output.append(summarize_group(method_rows, "overall"))
        for density in sorted({row["density"] for row in method_rows}):
            group = [row for row in method_rows if row["density"] == density]
            output.append(summarize_group(group, "density", density=density))
        for map_name in sorted({row["map"] for row in method_rows}):
            group = [row for row in method_rows if row["map"] == map_name]
            output.append(summarize_group(group, "map", density=group[0]["density"], map_name=map_name))
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=Path(__file__).resolve().parents[2] / "map")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("benchmark_results"))
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes (default: 1, matching Ablation timing conditions)")
    parser.add_argument("--ours-only", action="store_true",
                        help="Run only GASKPO-RRT* (ours), skipping baseline planners")
    parser.add_argument("--informed-only", action="store_true",
                        help="Run only Informed RRT*")
    parser.add_argument("--bitstar-only", action="store_true",
                        help="Run only BIT*")
    parser.add_argument("--pair-mode", choices=("paired", "cartesian"), default="paired")
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=50,
                        help="Independent runs per method and start-goal pair (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--front-length", type=float, default=3.0)
    parser.add_argument("--rear-length", type=float, default=1.0)
    parser.add_argument("--vehicle-width", type=float, default=1.6)
    parser.add_argument("--safety-distance", type=float, default=0.5)
    parser.add_argument("--collision-resolution", type=float, default=0.20)
    parser.add_argument("--shortcut-iterations", type=int, default=100)
    parser.add_argument("--smooth-spacing", type=float, default=0.20)
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply shortcut/smoothing; disabled by default to match Ablation")
    parser.add_argument("--grid-resolution", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--step-size", type=float, default=3.0)
    parser.add_argument("--integration-step-size", type=float, default=0.08,
                        help=argparse.SUPPRESS)
    parser.add_argument("--near-radius", type=float, default=7.0)
    parser.add_argument("--rewire-gamma", type=float, default=45.0)
    parser.add_argument("--goal-connect-distance", type=float, default=5.0)
    parser.add_argument("--goal-bias", type=float, default=0.20)
    parser.add_argument("--tangent-probability", type=float, default=0.10)
    parser.add_argument("--tangent-clearance", type=float, default=0.50)
    parser.add_argument("--ours-max-iterations", type=int, default=2500)
    parser.add_argument("--ours-expand-length", type=float, default=3.0)
    parser.add_argument("--ours-integration-step-size", type=float, default=0.08)
    parser.add_argument("--ours-near-radius", type=float, default=5.0)
    parser.add_argument("--ours-goal-connect-distance", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=200,
                        help="BIT* samples added per batch")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be >= 1")
    if args.goal_bias < 0 or args.tangent_probability < 0 or args.goal_bias + args.tangent_probability > 1:
        parser.error("goal-bias and tangent-probability must be non-negative and sum to <= 1")
    if sum((args.ours_only, args.informed_only, args.bitstar_only)) > 1:
        parser.error("ours-only, informed-only and bitstar-only are mutually exclusive")
    return args


def main():
    args = parse_args()
    maps = sorted(args.map_dir.rglob("*.npz"))
    if not maps:
        raise SystemExit(f"No NPZ maps found below: {args.map_dir}")
    config = SimpleNamespace(**vars(args))
    tasks = []
    if args.ours_only:
        selected_methods = [3]
    elif args.informed_only:
        selected_methods = [4]
    elif args.bitstar_only:
        selected_methods = [5]
    else:
        selected_methods = range(len(METHODS))
    for map_index, map_path in enumerate(maps):
        env = load_environment_quiet(str(map_path))
        density = map_path.parent.name
        for start_i, goal_i, start, goal in select_pairs(env, args.pair_mode, args.pair_limit):
            for method_index in selected_methods:
                for repeat in range(args.repeats):
                    tasks.append((method_index, map_index, str(map_path), density,
                                  start_i, goal_i, start, goal, repeat, config))
    print(f"Maps={len(maps)}, tasks={len(tasks)}, workers={args.workers}", flush=True)
    rows = []
    if args.workers == 1:
        progress = tqdm(map(run_task, tasks), total=len(tasks), desc="Benchmark",
                        unit="task", dynamic_ncols=True)
        for row in progress:
            rows.append(row)
            progress.set_postfix(map=row["map"], method=row["method"],
                                 success=row["success"], refresh=False)
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_task, task) for task in tasks]
                progress = tqdm(as_completed(futures), total=len(futures), desc="Benchmark",
                                unit="task", dynamic_ncols=True)
                for future in progress:
                    row = future.result()
                    rows.append(row)
                    progress.set_postfix(map=row["map"], method=row["method"],
                                         success=row["success"], refresh=False)
        except (PermissionError, OSError) as exc:
            # Some containers disable POSIX semaphores. Preserve functionality
            # by falling back to deterministic single-process execution.
            tqdm.write(f"Parallel execution unavailable ({exc}); falling back to one worker.")
            rows.clear()
            progress = tqdm(tasks, total=len(tasks), desc="Benchmark (1 worker)",
                            unit="task", dynamic_ncols=True)
            for task in progress:
                row = run_task(task)
                rows.append(row)
                progress.set_postfix(map=row["map"], method=row["method"],
                                     success=row["success"], refresh=False)
    method_order = {name: i for i, (name, _) in enumerate(METHODS)}
    rows.sort(key=lambda row: (row["density"], row["map"], row["start_index"],
                               row["goal_index"], row["repeat"], method_order[row["method"]]))
    summaries = build_summaries(rows)
    single_names = {3: "gaskpo_rrtstar", 4: "informed_rrtstar", 5: "bitstar"}
    selected_list = list(selected_methods)
    prefix = single_names.get(selected_list[0]) if len(selected_list) == 1 else "all_maps"
    detail_path = args.output_dir / f"{prefix}_detail.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    paper_path = args.output_dir / f"{prefix}_paper_table.csv"
    write_csv(detail_path, rows)
    write_csv(summary_path, summaries)
    write_csv(paper_path, [row for row in summaries if row["scope"] == "overall"])
    print(f"\nDetail CSV: {detail_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Paper table: {paper_path.resolve()}")


if __name__ == "__main__":
    main()
