import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class DubinsPath:
    modes: list[str]
    lengths: list[float]
    total_length: float
    x: list[float]
    y: list[float]
    yaw: list[float]


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mod2pi(angle: float) -> float:
    """将角度归一化到 [0, 2*pi)。"""
    return angle % (2.0 * math.pi)


# ============================================================
# 六类 Dubins 路径的解析解
#
# 输入 alpha、beta、d 均处于归一化坐标系。
# 返回的 t、p、q 是以最小转弯半径为单位的无量纲长度。
# ============================================================

def dubins_lsl(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    p_squared = (
        2.0
        + d * d
        - 2.0 * cab
        + 2.0 * d * (sa - sb)
    )

    if p_squared < 0.0:
        return None

    tmp = math.atan2(cb - ca, d + sa - sb)

    t = mod2pi(-alpha + tmp)
    p = math.sqrt(max(0.0, p_squared))
    q = mod2pi(beta - tmp)

    return t, p, q


def dubins_rsr(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    p_squared = (
        2.0
        + d * d
        - 2.0 * cab
        + 2.0 * d * (sb - sa)
    )

    if p_squared < 0.0:
        return None

    tmp = math.atan2(ca - cb, d - sa + sb)

    t = mod2pi(alpha - tmp)
    p = math.sqrt(max(0.0, p_squared))
    q = mod2pi(-beta + tmp)

    return t, p, q


def dubins_lsr(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    p_squared = (
        -2.0
        + d * d
        + 2.0 * cab
        + 2.0 * d * (sa + sb)
    )

    if p_squared < 0.0:
        return None

    p = math.sqrt(max(0.0, p_squared))

    tmp = math.atan2(
        -ca - cb,
        d + sa + sb,
    ) - math.atan2(-2.0, p)

    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)

    return t, p, q


def dubins_rsl(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    p_squared = (
        d * d
        - 2.0
        + 2.0 * cab
        - 2.0 * d * (sa + sb)
    )

    if p_squared < 0.0:
        return None

    p = math.sqrt(max(0.0, p_squared))

    tmp = math.atan2(
        ca + cb,
        d - sa - sb,
    ) - math.atan2(2.0, p)

    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)

    return t, p, q


def dubins_rlr(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    tmp = (
        6.0
        - d * d
        + 2.0 * cab
        + 2.0 * d * (sa - sb)
    ) / 8.0

    if abs(tmp) > 1.0:
        return None

    p = mod2pi(2.0 * math.pi - math.acos(tmp))

    t = mod2pi(
        alpha
        - math.atan2(ca - cb, d - sa + sb)
        + p / 2.0
    )

    q = mod2pi(
        alpha
        - beta
        - t
        + p
    )

    return t, p, q


def dubins_lrl(
    alpha: float,
    beta: float,
    d: float,
) -> Optional[tuple[float, float, float]]:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cab = math.cos(alpha - beta)

    tmp = (
        6.0
        - d * d
        + 2.0 * cab
        + 2.0 * d * (-sa + sb)
    ) / 8.0

    if abs(tmp) > 1.0:
        return None

    p = mod2pi(2.0 * math.pi - math.acos(tmp))

    t = mod2pi(
        -alpha
        - math.atan2(ca - cb, d + sa - sb)
        + p / 2.0
    )

    q = mod2pi(
        beta
        - alpha
        - t
        + p
    )

    return t, p, q


DUBINS_CANDIDATES = [
    ("LSL", dubins_lsl),
    ("RSR", dubins_rsr),
    ("LSR", dubins_lsr),
    ("RSL", dubins_rsl),
    ("RLR", dubins_rlr),
    ("LRL", dubins_lrl),
]


def interpolate_segment(
    mode: str,
    length: float,
    curvature: float,
    step_size: float,
    x: float,
    y: float,
    yaw: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    从当前状态开始，对一段 L、R 或 S 轨迹进行插值。

    length 是实际长度，单位 m。
    curvature 是最大曲率，单位 1/m。
    """

    if length < 0.0:
        raise ValueError("Dubins 单段长度不能为负")

    if step_size <= 0.0:
        raise ValueError("step_size 必须大于 0")

    segment_x = []
    segment_y = []
    segment_yaw = []

    traveled = 0.0

    while traveled < length - 1e-10:
        ds = min(step_size, length - traveled)

        if mode == "S":
            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)

        elif mode == "L":
            new_yaw = yaw + curvature * ds

            x += (
                math.sin(new_yaw) - math.sin(yaw)
            ) / curvature

            y += (
                -math.cos(new_yaw) + math.cos(yaw)
            ) / curvature

            yaw = new_yaw

        elif mode == "R":
            new_yaw = yaw - curvature * ds

            x += (
                -math.sin(new_yaw) + math.sin(yaw)
            ) / curvature

            y += (
                math.cos(new_yaw) - math.cos(yaw)
            ) / curvature

            yaw = new_yaw

        else:
            raise ValueError(f"未知路径模式：{mode}")

        yaw = normalize_angle(yaw)

        segment_x.append(x)
        segment_y.append(y)
        segment_yaw.append(yaw)

        traveled += ds

    return segment_x, segment_y, segment_yaw


def plan_dubins_path(
    start: Pose,
    goal: Pose,
    curvature: float,
    step_size: float = 0.1,
) -> Optional[DubinsPath]:
    """
    规划从 start 到 goal 的最短 Dubins 路径。

    curvature:
        最大允许曲率，即 1 / 最小转弯半径。
    """

    if curvature <= 0.0:
        raise ValueError("curvature 必须大于 0")

    # --------------------------------------------------------
    # 将目标状态转换到以起点为原点、起点航向为 0 的局部坐标系。
    # --------------------------------------------------------
    dx = goal.x - start.x
    dy = goal.y - start.y

    cos_start = math.cos(start.yaw)
    sin_start = math.sin(start.yaw)

    local_goal_x = (
        cos_start * dx
        + sin_start * dy
    )

    local_goal_y = (
        -sin_start * dx
        + cos_start * dy
    )

    local_goal_yaw = normalize_angle(
        goal.yaw - start.yaw
    )

    distance = math.hypot(
        local_goal_x,
        local_goal_y,
    )

    normalized_distance = distance * curvature

    theta = mod2pi(
        math.atan2(local_goal_y, local_goal_x)
    )

    alpha = mod2pi(-theta)
    beta = mod2pi(local_goal_yaw - theta)

    best_modes = None
    best_normalized_lengths = None
    best_normalized_cost = math.inf

    # --------------------------------------------------------
    # 遍历六类 Dubins 路径，选择长度最短的可行路径。
    # --------------------------------------------------------
    for mode_string, solver in DUBINS_CANDIDATES:
        result = solver(
            alpha,
            beta,
            normalized_distance,
        )

        if result is None:
            continue

        t, p, q = result
        normalized_cost = t + p + q

        if normalized_cost < best_normalized_cost:
            best_normalized_cost = normalized_cost
            best_modes = list(mode_string)
            best_normalized_lengths = [t, p, q]

    if best_modes is None or best_normalized_lengths is None:
        return None

    # 将无量纲长度恢复为实际长度。
    actual_lengths = [
        value / curvature
        for value in best_normalized_lengths
    ]

    path_x = [start.x]
    path_y = [start.y]
    path_yaw = [normalize_angle(start.yaw)]

    current_x = start.x
    current_y = start.y
    current_yaw = start.yaw

    for mode, length in zip(best_modes, actual_lengths):
        segment_x, segment_y, segment_yaw = interpolate_segment(
            mode=mode,
            length=length,
            curvature=curvature,
            step_size=step_size,
            x=current_x,
            y=current_y,
            yaw=current_yaw,
        )

        path_x.extend(segment_x)
        path_y.extend(segment_y)
        path_yaw.extend(segment_yaw)

        if segment_x:
            current_x = segment_x[-1]
            current_y = segment_y[-1]
            current_yaw = segment_yaw[-1]

    total_length = sum(actual_lengths)

    return DubinsPath(
        modes=best_modes,
        lengths=actual_lengths,
        total_length=total_length,
        x=path_x,
        y=path_y,
        yaw=path_yaw,
    )


def calculate_pose_error(
    path: DubinsPath,
    goal: Pose,
) -> tuple[float, float]:
    position_error = math.hypot(
        path.x[-1] - goal.x,
        path.y[-1] - goal.y,
    )

    yaw_error = abs(
        normalize_angle(path.yaw[-1] - goal.yaw)
    )

    return position_error, yaw_error


def draw_pose(
    ax,
    pose: Pose,
    arrow_length: float = 1.0,
):
    ax.scatter(pose.x, pose.y, s=70)

    ax.arrow(
        pose.x,
        pose.y,
        arrow_length * math.cos(pose.yaw),
        arrow_length * math.sin(pose.yaw),
        width=0.03,
        length_includes_head=True,
    )


def main():
    # ========================================================
    # 与阶段 1 保持一致的车辆参数
    # ========================================================
    wheel_base = 2.5
    max_steering_angle = math.radians(30.0)

    minimum_turning_radius = (
        wheel_base / math.tan(max_steering_angle)
    )

    maximum_curvature = (
        1.0 / minimum_turning_radius
    )

    # 轨迹采样间隔
    step_size = 0.05

    start = Pose(
        x=0.0,
        y=0.0,
        yaw=math.radians(0.0),
    )

    goal = Pose(
        x=8.0,
        y=6.0,
        yaw=math.radians(90.0),
    )

    path = plan_dubins_path(
        start=start,
        goal=goal,
        curvature=maximum_curvature,
        step_size=step_size,
    )

    if path is None:
        print("未找到 Dubins 路径")
        return

    position_error, yaw_error = calculate_pose_error(
        path,
        goal,
    )

    print("=" * 60)
    print("Dubins 局部路径规划测试")
    print("=" * 60)

    print(f"轴距：{wheel_base:.3f} m")
    print(
        f"最大前轮转角："
        f"{math.degrees(max_steering_angle):.3f} deg"
    )

    print(
        f"最小转弯半径："
        f"{minimum_turning_radius:.3f} m"
    )

    print(
        f"最大曲率："
        f"{maximum_curvature:.6f} 1/m"
    )

    print(f"路径类型：{''.join(path.modes)}")

    print(
        "各段长度：",
        [
            round(length, 3)
            for length in path.lengths
        ],
    )

    print(f"总路径长度：{path.total_length:.3f} m")

    print(
        f"终点位置误差："
        f"{position_error:.8f} m"
    )

    print(
        f"终点航向误差："
        f"{math.degrees(yaw_error):.8f} deg"
    )

    print(
        f"路径采样点数量："
        f"{len(path.x)}"
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(
        path.x,
        path.y,
        linewidth=2,
        label=f"Dubins path: {''.join(path.modes)}",
    )

    draw_pose(ax, start)
    draw_pose(ax, goal)

    # 每隔若干点绘制一个航向
    interval = max(1, len(path.x) // 20)

    for i in range(0, len(path.x), interval):
        ax.arrow(
            path.x[i],
            path.y[i],
            0.4 * math.cos(path.yaw[i]),
            0.4 * math.sin(path.yaw[i]),
            width=0.01,
            length_includes_head=True,
        )

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Dubins Local Planner Test")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()