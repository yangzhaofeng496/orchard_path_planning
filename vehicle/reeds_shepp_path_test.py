import sys
import os
import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt

# 添加 vehicle 目录到 sys.path，以便导入同目录的模块
sys.path.insert(0, os.path.dirname(__file__))

from reeds_shepp_core import reeds_shepp_path_planning


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class ReedsSheppPath:
    x: list[float]
    y: list[float]
    yaw: list[float]
    directions: list[int]
    modes: list[str]
    lengths: list[float]
    total_length: float
    forward_length: float
    reverse_length: float
    gear_switch_count: int


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def count_gear_switches(lengths: list[float]) -> int:
    signs = [1 if length > 0.0 else -1 for length in lengths if abs(length) > 1e-10]
    return sum(a != b for a, b in zip(signs[:-1], signs[1:]))


def plan_reeds_shepp_path(
    start: Pose,
    goal: Pose,
    curvature: float,
    step_size: float = 0.05,
) -> Optional[ReedsSheppPath]:
    if curvature <= 0.0:
        raise ValueError("curvature 必须大于 0")
    if step_size <= 0.0:
        raise ValueError("step_size 必须大于 0")

    result = reeds_shepp_path_planning(
        start.x, start.y, start.yaw,
        goal.x, goal.y, goal.yaw,
        curvature, step_size,
    )

    path_x, path_y, path_yaw, directions, modes, lengths = result

    if path_x is None:
        return None

    total_length = sum(abs(length) for length in lengths)
    forward_length = sum(length for length in lengths if length > 0.0)
    reverse_length = sum(abs(length) for length in lengths if length < 0.0)

    return ReedsSheppPath(
        x=list(path_x),
        y=list(path_y),
        yaw=[normalize_angle(yaw) for yaw in path_yaw],
        directions=[int(direction) for direction in directions],
        modes=list(modes),
        lengths=list(lengths),
        total_length=total_length,
        forward_length=forward_length,
        reverse_length=reverse_length,
        gear_switch_count=count_gear_switches(lengths),
    )


def calculate_pose_error(path: ReedsSheppPath, goal: Pose) -> tuple[float, float]:
    position_error = math.hypot(path.x[-1] - goal.x, path.y[-1] - goal.y)
    yaw_error = abs(normalize_angle(path.yaw[-1] - goal.yaw))
    return position_error, yaw_error


def build_segment_description(modes: list[str], lengths: list[float]) -> str:
    parts = []
    for mode, length in zip(modes, lengths):
        sign = "+" if length >= 0.0 else "-"
        parts.append(f"{mode}{sign}{abs(length):.3f}")
    return " -> ".join(parts)


def split_path_by_direction(path: ReedsSheppPath):
    if not path.x:
        return []

    segments = []
    direction = path.directions[0]
    segment_x = [path.x[0]]
    segment_y = [path.y[0]]

    for i in range(1, len(path.x)):
        new_direction = path.directions[i]

        if new_direction != direction:
            segment_x.append(path.x[i])
            segment_y.append(path.y[i])
            segments.append((direction, segment_x, segment_y))

            direction = new_direction
            segment_x = [path.x[i - 1], path.x[i]]
            segment_y = [path.y[i - 1], path.y[i]]
        else:
            segment_x.append(path.x[i])
            segment_y.append(path.y[i])

    segments.append((direction, segment_x, segment_y))
    return segments


def draw_pose(ax, pose: Pose, label: str, arrow_length: float = 0.8):
    ax.scatter(pose.x, pose.y, s=70, label=label)
    ax.arrow(
        pose.x, pose.y,
        arrow_length * math.cos(pose.yaw),
        arrow_length * math.sin(pose.yaw),
        width=0.025,
        length_includes_head=True,
    )


def plot_path(start: Pose, goal: Pose, path: ReedsSheppPath):
    fig, ax = plt.subplots(figsize=(9, 8))

    forward_label_added = False
    reverse_label_added = False

    for direction, segment_x, segment_y in split_path_by_direction(path):
        if direction > 0:
            label = None if forward_label_added else "forward"
            ax.plot(segment_x, segment_y, "-", linewidth=2, label=label)
            forward_label_added = True
        else:
            label = None if reverse_label_added else "reverse"
            ax.plot(segment_x, segment_y, "--", linewidth=2, label=label)
            reverse_label_added = True

    draw_pose(ax, start, "start")
    draw_pose(ax, goal, "goal")

    interval = max(1, len(path.x) // 20)

    for i in range(0, len(path.x), interval):
        dx = 0.35 * math.cos(path.yaw[i])
        dy = 0.35 * math.sin(path.yaw[i])
        ax.plot([path.x[i], path.x[i] + dx], [path.y[i], path.y[i] + dy], linewidth=1)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Reeds-Shepp Local Planner Test")
    ax.legend()
    plt.show()


def main():
    wheel_base = 2.5
    max_steering_angle = math.radians(30.0)

    minimum_turning_radius = wheel_base / math.tan(max_steering_angle)
    maximum_curvature = 1.0 / minimum_turning_radius
    step_size = 0.05

    # start = Pose(0.0, 0.0, math.radians(0.0))
    # goal = Pose(-4.0, 0.0, math.radians(0.0))

    start = Pose(0.0, 0.0, math.radians(0.0))
    goal = Pose(1.0, 1.0, math.radians(180.0))

    path = plan_reeds_shepp_path(
        start, goal,
        curvature=maximum_curvature,
        step_size=step_size,
    )

    if path is None:
        print("没有找到 Reeds-Shepp 路径")
        return

    position_error, yaw_error = calculate_pose_error(path, goal)

    print("=" * 70)
    print("Reeds-Shepp 局部路径规划测试")
    print("=" * 70)
    print(f"轴距：{wheel_base:.3f} m")
    print(f"最大前轮转角：{math.degrees(max_steering_angle):.3f} deg")
    print(f"最小转弯半径：{minimum_turning_radius:.3f} m")
    print(f"最大曲率：{maximum_curvature:.6f} 1/m")
    print(f"路径模式：{path.modes}")
    print(f"带符号分段长度：{[round(v, 3) for v in path.lengths]}")
    print(f"路径段描述：{build_segment_description(path.modes, path.lengths)}")
    print(f"总路径长度：{path.total_length:.3f} m")
    print(f"前进距离：{path.forward_length:.3f} m")
    print(f"倒车距离：{path.reverse_length:.3f} m")
    print(f"换向次数：{path.gear_switch_count}")
    print(f"终点位置误差：{position_error:.10f} m")
    print(f"终点航向误差：{math.degrees(yaw_error):.10f} deg")
    print(f"离散采样点数量：{len(path.x)}")
    print(f"轨迹中包含的方向：{sorted(set(path.directions))}")

    plot_path(start, goal, path)


if __name__ == "__main__":
    main()