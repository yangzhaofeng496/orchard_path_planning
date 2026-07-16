"""硕士论文补充实验：密度、重叠率、消融与参数敏感性（支持断点续跑）。"""
import argparse
import csv
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_rrt_star import run_once
from orchard_environment import (
    OrchardEnvironment,
    make_goal_rectangle,
    make_hybrid_benchmark_environment,
)
from vehicle.vehicle_collision_test import CircleObstacle


METHODS = ("RRT*", "GoalBias", "Hybrid")
FIELDS = [
    "experiment", "level", "map_seed", "planner_seed", "method", "variant",
    "success", "planning_time", "first_solution_iteration", "node_count",
    "path_length", "reverse_length", "switch_count",
]


def point_clear(x, y, radius, point, margin=6.0):
    return math.hypot(x - point[0], y - point[1]) > radius + margin


def random_environment(count, map_seed, overlap_probability=0.0):
    rng = random.Random(map_seed)
    bounds = (0.0, 90.0, 0.0, 90.0)
    start, goal = (8.0, 45.0), (82.0, 45.0)
    obstacles = []
    attempts = 0
    while len(obstacles) < count and attempts < count * 200:
        attempts += 1
        radius = rng.uniform(2.0, 4.0)
        use_overlap = obstacles and rng.random() < overlap_probability
        if use_overlap:
            base = rng.choice(obstacles)
            angle = rng.uniform(-math.pi, math.pi)
            distance = rng.uniform(0.35, 0.90) * (base.radius + radius)
            x = base.x + distance * math.cos(angle)
            y = base.y + distance * math.sin(angle)
        else:
            x = rng.uniform(7.0, 83.0)
            y = rng.uniform(7.0, 83.0)
        if not (radius < x < 90.0 - radius and radius < y < 90.0 - radius):
            continue
        if not point_clear(x, y, radius, start) or not point_clear(x, y, radius, goal):
            continue
        obstacles.append(CircleObstacle(x, y, radius))
    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=make_goal_rectangle(start, goal, 30.0, 20.0),
        start_pos=start,
        goal_pos=goal,
        bounds=bounds,
        description=(
            f"随机圆障碍: count={len(obstacles)}, overlap={overlap_probability:.2f}, "
            f"map_seed={map_seed}"
        ),
    )


def make_tasks():
    tasks = []
    # 4密度 × 5地图 × 10规划种子 × 3算法 = 600
    for count in (10, 20, 30, 40):
        for map_seed in range(5):
            for planner_seed in range(10):
                for method in METHODS:
                    tasks.append({"experiment": "density", "level": str(count),
                                  "map_seed": map_seed, "planner_seed": planner_seed,
                                  "method": method, "variant": method})
    # 4重叠率 × 5地图 × 10规划种子 × 3算法 = 600
    for overlap in (0.0, 0.2, 0.4, 0.6):
        for map_seed in range(5):
            for planner_seed in range(10):
                for method in METHODS:
                    tasks.append({"experiment": "overlap", "level": f"{overlap:.1f}",
                                  "map_seed": map_seed, "planner_seed": planner_seed,
                                  "method": method, "variant": method})
    # 5版本 × 2场景 × 30种子 = 300
    variants = (
        ("RRT*", "RRT*", {}),
        ("GoalBias", "GoalBias", {}),
        ("Rectangle", "Hybrid", {"use_tangent_guidance": False}),
        ("SingleTangent", "Hybrid", {"cluster_shape": "single_circle"}),
        ("EllipseCluster", "Hybrid", {"cluster_shape": "ellipse"}),
    )
    for scene in ("single_blocker", "staggered_trees"):
        for planner_seed in range(30):
            for variant, method, options in variants:
                tasks.append({"experiment": "ablation", "level": scene,
                              "map_seed": 0, "planner_seed": planner_seed,
                              "method": method, "variant": variant,
                              "options": options})
    # 三参数各5水平 × 20种子 = 300（交错果树场景）
    sensitivity = {
        "rectangle_width": (8.0, 12.0, 16.0, 20.0, 24.0),
        "rectangle_probability": (0.15, 0.25, 0.35, 0.45, 0.55),
        "tangent_extension": (2.0, 4.0, 6.0, 8.0, 10.0),
    }
    for parameter, values in sensitivity.items():
        for value in values:
            for planner_seed in range(20):
                tasks.append({"experiment": "sensitivity", "level": f"{parameter}={value:g}",
                              "map_seed": 0, "planner_seed": planner_seed,
                              "method": "Hybrid", "variant": parameter,
                              "parameter": parameter, "value": value})
    return tasks


def task_key(task):
    return tuple(str(task.get(name, "")) for name in
                 ("experiment", "level", "map_seed", "planner_seed", "method", "variant"))


def run_task(task):
    experiment = task["experiment"]
    rectangle_width = 20.0
    options = dict(task.get("options", {}))
    if experiment == "density":
        environment = random_environment(int(task["level"]), task["map_seed"] + 1000)
    elif experiment == "overlap":
        environment = random_environment(
            30, task["map_seed"] + 2000, float(task["level"])
        )
    elif experiment == "ablation":
        environment = make_hybrid_benchmark_environment(task["level"])
    else:
        environment = make_hybrid_benchmark_environment("staggered_trees")
        parameter, value = task["parameter"], task["value"]
        if parameter == "rectangle_width":
            rectangle_width = value
        else:
            options[parameter] = value

    metrics, *_ = run_once(
        method=task["method"], seed=task["planner_seed"],
        environment=environment, environment_path=None,
        rectangle_length=30.0, rectangle_width=rectangle_width,
        allow_reverse=False, sampling_options=options,
    )
    row = {name: task.get(name, "") for name in FIELDS}
    row.update(metrics)
    return row


def read_completed(path):
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return rows, {task_key(row) for row in rows}


def append_row(path, row):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "remaining_experiment_results", "detail.csv"
    ))
    args = parser.parse_args()
    tasks = make_tasks()
    _, completed = read_completed(args.output)
    pending = [task for task in tasks if task_key(task) not in completed]
    print(f"总任务={len(tasks)}, 已完成={len(completed)}, 待运行={len(pending)}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_task, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_row(args.output, row)
            print(f"[{len(completed)+index}/{len(tasks)}] {row['experiment']} "
                  f"{row['level']} {row['variant']} success={row.get('success', 0)}",
                  flush=True)


if __name__ == "__main__":
    main()
