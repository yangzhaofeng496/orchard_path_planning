#!/usr/bin/env python3
"""Shared I/O, collision, post-processing and benchmark code for External planners."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from global_path_planning.innovation_sample.orchard_environment import load_environment
from path_optimizer.curvature_smoother import CurvatureSmoother
from path_optimizer.shortcut import ShortcutOptimizer
from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry, check_pose_collision

Point = tuple[float, float]


@dataclass
class SearchResult:
    path: Optional[list[Point]]
    nodes: int
    first_iteration: int
    planning_time_s: Optional[float] = None


@dataclass
class TrialMetric:
    algorithm: str
    map: str
    start_index: int
    goal_index: int
    repeat: int
    seed: int
    success: int
    planning_time_s: float
    total_time_s: float
    path_length_m: float
    node_count: int
    first_solution_iteration: int
    path_point_count: int
    detour_ratio: float
    min_obstacle_distance_m: float


class CollisionChecker:
    """Adapter over the repository's unchanged rectangular-vehicle interface."""

    def __init__(self, vehicle, obstacles, bounds, resolution=0.20):
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.bounds = tuple(float(v) for v in bounds)
        self.resolution = float(resolution)

    def point_free(self, point: Point, yaw: float = 0.0) -> bool:
        x, y = point
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        return not check_pose_collision(Pose(float(x), float(y), float(yaw)), self.vehicle, self.obstacles)

    def check_line(self, p1: Point, p2: Point) -> bool:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx) if distance > 1e-12 else 0.0
        count = max(1, int(math.ceil(distance / self.resolution)))
        return all(self.point_free((p1[0] + dx * i / count, p1[1] + dy * i / count), yaw)
                   for i in range(count + 1))


def path_length(path: Iterable[Point]) -> float:
    points = list(path)
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def postprocess(path: list[Point], checker: CollisionChecker, seed: int,
                shortcut_iterations: int, smooth_spacing: float) -> list[Point]:
    shortcut = ShortcutOptimizer(checker, max_iterations=shortcut_iterations,
                                 random_seed=seed, verbose=False).optimize(path)
    smoothed = CurvatureSmoother(
        collision_checker=checker, interpolation_spacing=smooth_spacing,
        max_curvature=0.22, verbose=False,
    ).smooth(shortcut).points
    # A final whole-path guard keeps reported paths collision-free even if future
    # smoother implementations change their local acceptance behavior.
    if len(smoothed) >= 2 and all(checker.check_line(a, b) for a, b in zip(smoothed[:-1], smoothed[1:])):
        return smoothed
    return shortcut


def minimum_clearance(path: list[Point], obstacles, vehicle, spacing=0.10) -> float:
    """Minimum centreline-to-obstacle-boundary distance along the final path."""
    if not obstacles or not path:
        return math.inf
    del vehicle
    best = math.inf
    for a, b in zip(path[:-1], path[1:]):
        length = math.dist(a, b)
        count = max(1, int(math.ceil(length / spacing)))
        for i in range(count + 1):
            t = i / count
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            best = min(best, *(math.hypot(x - o.x, y - o.y) - o.radius for o in obstacles))
    return best


def select_pairs(env, mode: str, limit: Optional[int]):
    pairs = env.start_goal_pairs or [(env.start_pos, env.goal_pos)]
    if limit is not None:
        pairs = pairs[:limit]
    if mode == "paired":
        return [(i, i, tuple(s), tuple(g)) for i, (s, g) in enumerate(pairs)]
    starts, goals = [tuple(p[0]) for p in pairs], [tuple(p[1]) for p in pairs]
    return [(i, j, s, g) for i, s in enumerate(starts) for j, g in enumerate(goals)]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("map", type=Path, help="NPZ map containing obstacles, bounds and start_goal_pairs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--pair-mode", choices=("paired", "cartesian"), default="paired",
                        help="paired uses N stored pairs; cartesian evaluates N starts x N goals")
    parser.add_argument("--pair-limit", type=int, default=None, help="Use only the first N stored pairs")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--front-length", type=float, default=3.0)
    parser.add_argument("--rear-length", type=float, default=1.0)
    parser.add_argument("--vehicle-width", type=float, default=1.6)
    parser.add_argument("--safety-distance", type=float, default=0.5)
    parser.add_argument("--collision-resolution", type=float, default=0.20)
    parser.add_argument("--shortcut-iterations", type=int, default=100)
    parser.add_argument("--smooth-spacing", type=float, default=0.20)


def run_benchmark(args, algorithm: str, planner_factory: Callable) -> tuple[Path, Path]:
    env = load_environment(str(args.map))
    vehicle = VehicleGeometry(args.front_length, args.rear_length,
                              args.vehicle_width, args.safety_distance)
    checker = CollisionChecker(vehicle, env.obstacles, env.bounds, args.collision_resolution)
    trials: list[TrialMetric] = []
    for start_i, goal_i, start, goal in select_pairs(env, args.pair_mode, args.pair_limit):
        for repeat in range(args.repeats):
            trial_seed = int(np.random.SeedSequence([args.seed, start_i, goal_i, repeat]).generate_state(1)[0])
            total_start = time.perf_counter()
            planning_start = time.perf_counter()
            if not checker.point_free(start) or not checker.point_free(goal):
                result = SearchResult(None, 0, -1)
            else:
                result = planner_factory(start, goal, env.bounds, checker, env.obstacles, trial_seed, args)
            planning_time = time.perf_counter() - planning_start
            final_path = None
            if result.path:
                final_path = postprocess(result.path, checker, trial_seed,
                                         args.shortcut_iterations, args.smooth_spacing)
            total_time = time.perf_counter() - total_start
            success = int(final_path is not None)
            direct = math.dist(start, goal)
            length = path_length(final_path) if final_path else math.nan
            trials.append(TrialMetric(
                algorithm, str(args.map.resolve()), start_i, goal_i, repeat, trial_seed, success,
                planning_time, total_time, length, result.nodes, result.first_iteration,
                len(final_path) if final_path else 0,
                length / direct if success and direct > 1e-12 else math.nan,
                minimum_clearance(final_path, env.obstacles, vehicle) if final_path else math.nan,
            ))
            print(f"[{algorithm}] start={start_i} goal={goal_i} repeat={repeat} "
                  f"success={success} planning={planning_time:.4f}s nodes={result.nodes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{algorithm.lower().replace('*', 'star')}_trials.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(trials[0]).keys()))
        writer.writeheader(); writer.writerows(asdict(row) for row in trials)

    successful = [row for row in trials if row.success]
    def mean(field):
        values = [getattr(row, field) for row in successful]
        return float(np.mean(values)) if values else math.nan
    summary = {
        "algorithm": algorithm, "map": str(args.map.resolve()), "pair_mode": args.pair_mode,
        "trial_count": len(trials), "success_count": len(successful),
        "success_rate": len(successful) / len(trials) if trials else 0.0,
        **{f"mean_{field}": mean(field) for field in (
            "planning_time_s", "total_time_s", "path_length_m", "node_count",
            "first_solution_iteration", "path_point_count", "detour_ratio", "min_obstacle_distance_m")},
    }
    summary_path = args.output_dir / f"{algorithm.lower().replace('*', 'star')}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    print(f"CSV: {detail_path}\nCSV: {summary_path}")
    return detail_path, summary_path
