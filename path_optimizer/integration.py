"""
RRT 路径优化集成示例

展示如何将 ShortcutOptimizer 与现有的 Ackermann RRT* 规划器集成
"""

import sys
import os
import math
import numpy as np

# 添加项目路径
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from path_optimizer import ShortcutOptimizer, CollisionChecker
from vehicle.vehicle_collision_test import (
    VehicleGeometry,
    CircleObstacle,
    Pose,
)
from vehicle.reeds_shepp_path_test import plan_reeds_shepp_path
from typing import List, Tuple


class VehicleCollisionChecker:
    """
    基于车辆几何形状的碰撞检测器

    将 2D 路径点之间的连接转换为车辆轨迹，检查是否碰撞
    """

    def __init__(
        self,
        vehicle: VehicleGeometry,
        obstacles: List[CircleObstacle],
        curvature: float,
        step_size: float = 0.1,
    ):
        """
        Args:
            vehicle: 车辆几何参数
            obstacles: 障碍物列表
            curvature: 最大曲率（用于规划 Reeds-Shepp 路径）
            step_size: 轨迹采样步长
        """
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.curvature = curvature
        self.step_size = step_size

    def check_line(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        检查从 p1 到 p2 的连接是否无碰撞

        注意：这里简化为直线检测。完整实现应该使用 Reeds-Shepp 或 Dubins 曲线
        """
        # 简化版本：采样直线路径
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_samples = max(2, int(distance / self.step_size))

        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

            pose = Pose(x, y, yaw)

            # 检查该位姿是否与障碍物碰撞
            from vehicle.vehicle_collision_test import check_pose_collision
            if check_pose_collision(pose, self.vehicle, self.obstacles):
                return False  # 有碰撞

        return True  # 无碰撞


def rrt_path_to_points(rrt_result) -> List[Tuple[float, float]]:
    """
    将 RRT 结果转换为 2D 点列表

    Args:
        rrt_result: RRT 规划结果，格式为 (path_x, path_y, path_yaw, directions)

    Returns:
        点列表 [(x0, y0), (x1, y1), ...]
    """
    path_x, path_y, _, _ = rrt_result
    return list(zip(path_x, path_y))


def points_to_rrt_path(points: List[Tuple[float, float]], original_result):
    """
    将优化后的点列表转换回 RRT 路径格式

    Args:
        points: 优化后的点列表
        original_result: 原始 RRT 结果，用于提取 yaw 和 directions

    Returns:
        优化后的路径，格式为 (path_x, path_y, path_yaw, directions)
    """
    path_x = [p[0] for p in points]
    path_y = [p[1] for p in points]

    # 重新计算 yaw
    path_yaw = []
    for i in range(len(points)):
        if i < len(points) - 1:
            yaw = math.atan2(points[i+1][1] - points[i][1],
                            points[i+1][0] - points[i][0])
        else:
            yaw = path_yaw[-1] if path_yaw else 0.0
        path_yaw.append(yaw)

    # 假设优化后的路径都是前进
    directions = [1] * len(points)

    return (path_x, path_y, path_yaw, directions)


def optimize_rrt_path(
    rrt_result,
    vehicle: VehicleGeometry,
    obstacles: List[CircleObstacle],
    curvature: float,
    max_iterations: int = 100,
    verbose: bool = True,
):
    """
    优化 RRT 生成的路径

    Args:
        rrt_result: RRT 规划结果
        vehicle: 车辆几何参数
        obstacles: 障碍物列表
        curvature: 最大曲率
        max_iterations: 优化迭代次数
        verbose: 是否打印详细信息

    Returns:
        优化后的路径
    """
    # 1. 将 RRT 路径转换为点列表
    path_points = rrt_path_to_points(rrt_result)

    if verbose:
        print(f"\n[RRT Optimization] Original path: {len(path_points)} points")

    # 2. 创建碰撞检测器
    checker = VehicleCollisionChecker(
        vehicle=vehicle,
        obstacles=obstacles,
        curvature=curvature,
        step_size=0.1,
    )

    # 3. 创建优化器
    optimizer = ShortcutOptimizer(
        collision_checker=checker,
        max_iterations=max_iterations,
        min_points_distance=0.2,  # 适合车辆尺度
        enable_angle_filter=True,
        angle_threshold=np.deg2rad(15),
        random_seed=42,
        verbose=verbose,
    )

    # 4. 执行优化
    optimized_points = optimizer.optimize(path_points)

    if verbose:
        print(f"[RRT Optimization] Optimized path: {len(optimized_points)} points")
        optimizer.print_stats()

    # 5. 转换回 RRT 路径格式
    optimized_result = points_to_rrt_path(optimized_points, rrt_result)

    return optimized_result


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """
    完整的使用示例
    """
    print("\n" + "=" * 70)
    print("RRT Path Optimization Integration Example")
    print("=" * 70)

    # 1. 定义车辆参数
    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=0.2,
    )

    # 2. 定义障碍物
    obstacles = [
        CircleObstacle(5.0, 5.0, 1.5),
        CircleObstacle(10.0, 8.0, 1.2),
        CircleObstacle(15.0, 6.0, 1.0),
    ]

    # 3. 计算曲率
    wheel_base = 2.5
    max_steer = math.radians(30.0)
    curvature = math.tan(max_steer) / wheel_base

    # 4. 模拟 RRT 规划结果（实际使用时从 RRT 获取）
    # 这里创建一个假的 RRT 路径用于演示
    print("\n[Example] Creating simulated RRT path...")

    # 生成一条模拟的冗余路径
    num_points = 50
    path_x = list(np.linspace(0, 20, num_points))
    path_y = [5 + 2 * np.sin(x * 0.5) for x in path_x]
    path_yaw = [0.0] * num_points
    directions = [1] * num_points

    rrt_result = (path_x, path_y, path_yaw, directions)

    print(f"[Example] Simulated RRT path: {len(path_x)} points")

    # 5. 优化路径
    print("\n[Example] Starting optimization...")
    optimized_result = optimize_rrt_path(
        rrt_result=rrt_result,
        vehicle=vehicle,
        obstacles=obstacles,
        curvature=curvature,
        max_iterations=100,
        verbose=True,
    )

    opt_x, opt_y, opt_yaw, opt_dir = optimized_result

    print(f"\n[Example] Optimization complete!")
    print(f"[Example] Original: {len(path_x)} points")
    print(f"[Example] Optimized: {len(opt_x)} points")
    print(f"[Example] Reduction: {(1 - len(opt_x)/len(path_x)) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")


# ============================================================================
# 与 ackermann_rrt_star.py 的集成方式
# ============================================================================

"""
在实际项目中的集成方式：

在 ackermann_rrt_star.py 或调用 RRT 的脚本中：

```python
from path_optimizer import ShortcutOptimizer
from path_optimizer.integration import optimize_rrt_path

# 原始代码：
result = planner.planning()

if result is not None:
    path_x, path_y, path_yaw, directions = result
    # 使用原始路径...

# 修改为：
result = planner.planning()

if result is not None:
    # 优化路径
    optimized_result = optimize_rrt_path(
        rrt_result=result,
        vehicle=planner.vehicle,
        obstacles=planner.obstacles,
        curvature=planner.curvature,
        max_iterations=100,
        verbose=True,
    )

    path_x, path_y, path_yaw, directions = optimized_result
    # 使用优化后的路径...
```

关键点：
1. 不修改 RRT 核心算法
2. 在得到 RRT 结果后调用优化器
3. 优化器返回相同格式的路径
4. 可以选择性启用/禁用优化
"""


if __name__ == "__main__":
    example_usage()
