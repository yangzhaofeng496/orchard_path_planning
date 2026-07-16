"""20 m x 20 m random dense-orchard demonstration using existing RRT*/TEB/DWA."""

import argparse
from dataclasses import fields
import math
import os
import random
import sys

import matplotlib
# 交互运行时确保弹出窗口；CI/无界面运行可用 MPLBACKEND=Agg 覆盖。
if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import yaml

PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
INNOVATION_DIR = os.path.join(PROJECT_ROOT, "global_path_planning", "innovation_sample")
for path in (PROJECT_ROOT, INNOVATION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from ackermann_rrt_star import AckermannRRTStar
from vehicle.vehicle_collision_test import (
    CircleObstacle as RRTCircleObstacle,
    Pose as RRTPose,
    VehicleGeometry,
)
from local_path_planning.base import CircleObstacle, Pose, VehicleState
from local_path_planning.config import DWAConfig, TEBConfig
from local_path_planning.adaptive_teb_dwa.adaptive_window import AdaptiveWindowConfig
from local_path_planning.adaptive_teb_dwa.parameter_manager import FeedbackConfig
from local_path_planning.adaptive_teb_dwa.planner import (
    AdaptivePlannerConfig,
    AdaptiveTEBDWAPlanner,
)


def dataclass_from_dict(cls, values):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def generate_orchard(seed, count):
    rng = random.Random(seed)
    obstacles = []
    while len(obstacles) < count:
        radius = rng.uniform(0.18, 0.32)
        x, y = rng.uniform(1.0, 19.0), rng.uniform(1.0, 19.0)
        if math.hypot(x - 1.0, y - 1.0) < 2.0 or math.hypot(x - 18.0, y - 18.0) < 2.0:
            continue
        if any(math.hypot(x - o.x, y - o.y) < radius + o.radius + 0.35 for o in obstacles):
            continue
        obstacles.append(CircleObstacle(x, y, radius))
    return obstacles


def plan_global_path(obstacles, seed, iterations):
    vehicle = VehicleGeometry(front_length=1.0, rear_length=0.5, width=0.8, safety_margin=0.15)
    rrt_obstacles = [RRTCircleObstacle(o.x, o.y, o.radius) for o in obstacles]
    planner = AckermannRRTStar(
        start=RRTPose(1.0, 1.0, math.pi / 4.0),
        goal=RRTPose(18.0, 18.0, math.pi / 4.0),
        bounds=(0.0, 20.0, 0.0, 20.0),
        vehicle=vehicle,
        obstacles=rrt_obstacles,
        curvature=math.tan(math.radians(30.0)) / 2.5,
        max_iterations=iterations,
        expand_length=2.5,
        step_size=0.08,
        goal_connect_distance=5.0,
        use_hybrid_sampling=True,
        allow_reverse=False,
        use_goal_connector=True,
        relax_goal_yaw=True,
        random_seed=seed,
    )
    result = planner.planning()
    if result is None:
        raise RuntimeError("现有 AckermannRRTStar 在限定迭代内未找到路径，请更换 seed 或增加迭代次数")
    xs, ys, yaws, _ = result
    return [Pose(float(x), float(y), float(yaw)) for x, y, yaw in zip(xs, ys, yaws)]


def trajectory_metrics(trajectory, obstacles):
    length, curvatures = 0.0, []
    for first, second in zip(trajectory[:-1], trajectory[1:]):
        ds = math.hypot(second.x - first.x, second.y - first.y)
        length += ds
        if ds > 1e-8:
            dyaw = (second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi
            curvatures.append(abs(dyaw / ds))
    changes = [abs(b - a) for a, b in zip(curvatures[:-1], curvatures[1:])]
    clearance = min(
        (math.hypot(p.x - o.x, p.y - o.y) - o.radius for p in trajectory for o in obstacles),
        default=math.inf,
    )
    return {
        "path_length": length,
        "max_curvature": max(curvatures, default=0.0),
        "max_curvature_rate": max(changes, default=0.0),
        "min_obstacle_clearance": clearance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PACKAGE_DIR, "config.yaml"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obstacles", type=int, default=40, choices=range(30, 51))
    parser.add_argument("--rrt-iterations", type=int, default=5000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    print(f"[Demo] 生成随机果园: seed={args.seed}, obstacles={args.obstacles}", flush=True)
    obstacles = generate_orchard(args.seed, args.obstacles)
    print(f"[Demo] 调用现有 AckermannRRTStar，最多迭代 {args.rrt_iterations} 次...", flush=True)
    global_path = plan_global_path(obstacles, args.seed, args.rrt_iterations)
    print(f"[Demo] RRT* 完成: {len(global_path)} 个路径点", flush=True)
    state = VehicleState(1.0, 1.0, math.pi / 4.0, 0.0, 0.0)
    planner = AdaptiveTEBDWAPlanner(
        dataclass_from_dict(TEBConfig, data["teb"]),
        dataclass_from_dict(DWAConfig, data["dwa"]),
        bounds=(0.0, 20.0, 0.0, 20.0),
        planner_config=dataclass_from_dict(AdaptivePlannerConfig, data["planner"]),
        window_config=dataclass_from_dict(AdaptiveWindowConfig, data["adaptive_window"]),
        feedback_config=dataclass_from_dict(FeedbackConfig, data["feedback"]),
    )
    planner.set_global_path(global_path)
    print("[Demo] 开始 Adaptive TEB-DWA 反馈优化...", flush=True)
    result = planner.plan(state, obstacles)
    metrics = trajectory_metrics(result.trajectory, obstacles)
    stats = planner.statistics

    print("\nAdaptive TEB-DWA metrics")
    print(f"success_rate: {stats.success_rate:.3f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"dwa_rejections: {stats.dwa_rejections}")
    print(f"teb_reoptimizations: {stats.teb_reoptimizations}")
    print(f"computation_time: {result.computation_time:.4f} s")
    print(f"evaluation: {result.evaluation}")

    fig, ax = plt.subplots(figsize=(10, 10))
    for index, obstacle in enumerate(obstacles):
        ax.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius,
                            color="forestgreen", alpha=0.65,
                            label="fruit tree" if index == 0 else None))
    ax.plot([p.x for p in global_path], [p.y for p in global_path],
            "--", color="darkorange", linewidth=1.5, label="RRT* global path")
    if result.window:
        ref = result.window.reference_path
        ax.plot([p.x for p in ref], [p.y for p in ref], "b-", linewidth=2,
                label=f"dynamic window Ld={result.window.lookahead_distance:.1f}m")
        ax.add_patch(Circle((state.x, state.y), result.window.lookahead_distance,
                            fill=False, linestyle=":", color="blue", alpha=0.5))
    if result.trajectory:
        ax.plot([p.x for p in result.trajectory], [p.y for p in result.trajectory],
                "r-", linewidth=2.5, label="TEB optimized trajectory")
    score = result.evaluation.score if result.evaluation else 0.0
    feasible = result.evaluation.feasible if result.evaluation else False
    ax.text(0.02, 0.98, f"DWA score={score:.3f}\nfeasible={feasible}",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
    ax.plot(1, 1, "ko", label="start")
    ax.plot(18, 18, "r*", markersize=14, label="goal")
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="X (m)", ylabel="Y (m)",
           title="Adaptive Ackermann-TEB with DWA feedback")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output = os.path.join(PACKAGE_DIR, "adaptive_teb_dwa_demo.png")
    fig.savefig(output, dpi=160)
    print(f"[Demo] 图像已保存: {output}", flush=True)
    if not args.no_show:
        print("[Demo] 正在打开 Matplotlib 窗口，关闭窗口后程序退出。", flush=True)
        plt.show()


if __name__ == "__main__":
    main()
