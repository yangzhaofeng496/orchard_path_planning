import sys
import os
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# 添加 vehicle 目录到 sys.path，以便导入同目录的模块
sys.path.insert(0, os.path.dirname(__file__))

from vehicle.reeds_shepp_path import (
    Pose,
    ReedsSheppPath,
    plan_reeds_shepp_path,
    split_path_by_direction,
)


@dataclass
class VehicleGeometry:
    front_length: float
    rear_length: float
    width: float
    safety_margin: float = 0.0


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def world_to_vehicle(
    point_x: float, point_y: float, pose: Pose
) -> tuple[float, float]:
    """将世界坐标点转换到车辆局部坐标系。"""
    dx = point_x - pose.x
    dy = point_y - pose.y

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    return local_x, local_y


def circle_collides_with_vehicle(
    pose: Pose,
    vehicle: VehicleGeometry,
    obstacle: CircleObstacle,
) -> bool:
    """
    检查圆形障碍物是否与车辆矩形相交。

    车辆局部坐标范围：
        x ∈ [-rear_length, front_length]
        y ∈ [-width/2, width/2]
    """
    local_x, local_y = world_to_vehicle(obstacle.x, obstacle.y, pose)

    margin = vehicle.safety_margin
    x_min = -vehicle.rear_length - margin
    x_max = vehicle.front_length + margin
    y_min = -vehicle.width / 2.0 - margin
    y_max = vehicle.width / 2.0 + margin

    closest_x = clamp(local_x, x_min, x_max)
    closest_y = clamp(local_y, y_min, y_max)

    dx = local_x - closest_x
    dy = local_y - closest_y

    return dx * dx + dy * dy <= obstacle.radius * obstacle.radius


def check_pose_collision(
    pose: Pose,
    vehicle: VehicleGeometry,
    obstacles: list[CircleObstacle],
) -> bool:
    return any(
        circle_collides_with_vehicle(pose, vehicle, obstacle)
        for obstacle in obstacles
    )


def check_path_collision(
    path: ReedsSheppPath,
    vehicle: VehicleGeometry,
    obstacles: list[CircleObstacle],
) -> tuple[bool, list[int]]:
    """检查整条轨迹，返回是否碰撞以及碰撞轨迹点下标。"""
    collision_indices = []

    for i, (x, y, yaw) in enumerate(zip(path.x, path.y, path.yaw)):
        pose = Pose(x, y, yaw)

        if check_pose_collision(pose, vehicle, obstacles):
            collision_indices.append(i)

    return len(collision_indices) > 0, collision_indices


def get_vehicle_corners(
    pose: Pose,
    vehicle: VehicleGeometry,
) -> list[tuple[float, float]]:
    """计算车辆矩形的世界坐标角点。"""
    margin = vehicle.safety_margin
    front = vehicle.front_length + margin
    rear = vehicle.rear_length + margin
    half_width = vehicle.width / 2.0 + margin

    local_corners = [
        (front, half_width),
        (front, -half_width),
        (-rear, -half_width),
        (-rear, half_width),
    ]

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    corners = []

    for local_x, local_y in local_corners:
        world_x = pose.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = pose.y + local_x * sin_yaw + local_y * cos_yaw
        corners.append((world_x, world_y))

    return corners


def plot_path_segments(ax, path: ReedsSheppPath):
    forward_added = False
    reverse_added = False

    for direction, segment_x, segment_y in split_path_by_direction(path):
        if direction > 0:
            label = None if forward_added else "forward"
            ax.plot(segment_x, segment_y, "-", linewidth=2, label=label)
            forward_added = True
        else:
            label = None if reverse_added else "reverse"
            ax.plot(segment_x, segment_y, "--", linewidth=2, label=label)
            reverse_added = True


def plot_collision_result(
    path: ReedsSheppPath,
    vehicle: VehicleGeometry,
    obstacles: list[CircleObstacle],
    collision_indices: list[int],
):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_path_segments(ax, path)

    for i, obstacle in enumerate(obstacles):
        label = "obstacle" if i == 0 else None
        patch = Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            fill=False,
            linewidth=2,
            label=label,
        )
        ax.add_patch(patch)

    # 每隔一定数量的轨迹点画一次车身轮廓
    interval = max(1, len(path.x) // 20)

    for i in range(0, len(path.x), interval):
        pose = Pose(path.x[i], path.y[i], path.yaw[i])
        polygon = Polygon(
            get_vehicle_corners(pose, vehicle),
            closed=True,
            fill=False,
            linewidth=0.8,
            alpha=0.5,
        )
        ax.add_patch(polygon)

    # 绘制第一个碰撞姿态
    if collision_indices:
        collision_index = collision_indices[0]
        collision_pose = Pose(
            path.x[collision_index],
            path.y[collision_index],
            path.yaw[collision_index],
        )

        collision_vehicle = Polygon(
            get_vehicle_corners(collision_pose, vehicle),
            closed=True,
            fill=False,
            linewidth=3,
            label="first collision pose",
        )

        ax.add_patch(collision_vehicle)
        ax.scatter(
            collision_pose.x,
            collision_pose.y,
            marker="x",
            s=100,
            label="collision reference point",
        )

    ax.scatter(path.x[0], path.y[0], s=70, label="start")
    ax.scatter(path.x[-1], path.y[-1], s=70, label="goal")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Vehicle Path Collision Test")
    ax.legend()
    plt.show()


def main():
    wheel_base = 2.5
    max_steering_angle = math.radians(30.0)

    minimum_turning_radius = wheel_base / math.tan(max_steering_angle)
    maximum_curvature = 1.0 / minimum_turning_radius

    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=0.1,
    )

    start = Pose(0.0, 0.0, math.radians(0.0))
    goal = Pose(1.0, 1.0, math.radians(180.0))

    path = plan_reeds_shepp_path(
        start, goal,
        curvature=maximum_curvature,#最小转弯半径
        step_size=0.05,#采样间隔单位是m
    )

    if path is None:
        print("没有找到 Reeds-Shepp 路径")
        return

    # 在轨迹中间放置一个障碍物，保证第一次测试发生碰撞
    middle_index = len(path.x) // 2

    obstacles = [
        CircleObstacle(
            x=path.x[middle_index],
            y=path.y[middle_index],
            radius=0.35,
        ),
        CircleObstacle(x=5.0, y=5.0, radius=0.5),
    ]

    has_collision, collision_indices = check_path_collision(
        path, vehicle, obstacles
    )

    print("=" * 70)
    print("矩形车辆轨迹碰撞检测")
    print("=" * 70)
    print(f"车辆前轴方向长度：{vehicle.front_length:.3f} m")
    print(f"车辆后悬长度：{vehicle.rear_length:.3f} m")
    print(f"车辆宽度：{vehicle.width:.3f} m")
    print(f"安全余量：{vehicle.safety_margin:.3f} m")
    print(f"轨迹点数量：{len(path.x)}")
    print(f"是否发生碰撞：{has_collision}")
    print(f"碰撞轨迹点数量：{len(collision_indices)}")

    if collision_indices:
        first_index = collision_indices[0]

        print(f"首次碰撞点下标：{first_index}")
        print(
            "首次碰撞车辆状态："
            f"x={path.x[first_index]:.3f}, "
            f"y={path.y[first_index]:.3f}, "
            f"yaw={math.degrees(path.yaw[first_index]):.3f} deg"
        )

    plot_collision_result(path, vehicle, obstacles, collision_indices)


if __name__ == "__main__":
    main()