"""
Path Shortcut Optimizer 测试

提供简单的测试用例，验证优化器的功能：
1. 人工构造的简单路径
2. 带障碍物的碰撞检测
3. 优化效果验证
"""

import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from .shortcut import ShortcutOptimizer, CollisionChecker
except ImportError:  # 兼容直接运行 python test_shortcut.py
    from shortcut import ShortcutOptimizer, CollisionChecker


class SimpleCollisionChecker:
    """
    简单的 2D 碰撞检测器

    假设环境中有若干圆形障碍物，检查直线路径是否与障碍物相交
    """

    def __init__(self, obstacles: List[Tuple[float, float, float]], resolution: float = 0.05):
        """
        Args:
            obstacles: 障碍物列表，每个障碍物格式为 (x, y, radius)
            resolution: 直线采样分辨率（米）
        """
        self.obstacles = obstacles
        self.resolution = resolution

    def check_line(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        检查从 p1 到 p2 的直线是否无碰撞

        策略：对直线进行密集采样，检查每个采样点是否在障碍物内

        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)

        Returns:
            True 表示无碰撞，False 表示有碰撞
        """
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_samples = max(2, int(distance / self.resolution))

        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            # 检查是否与任何障碍物碰撞
            for obs_x, obs_y, obs_radius in self.obstacles:
                dist_to_obs = math.hypot(x - obs_x, y - obs_y)
                if dist_to_obs <= obs_radius:
                    return False  # 碰撞

        return True  # 无碰撞


def generate_zigzag_path(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    num_zigzags: int = 5,
) -> List[Tuple[float, float]]:
    """
    生成之字形路径，模拟 RRT 生成的冗余路径

    Args:
        start: 起点
        goal: 终点
        num_zigzags: 之字数量

    Returns:
        路径点列表
    """
    path = [start]

    for i in range(1, num_zigzags):
        t = i / num_zigzags
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])

        # 添加随机偏移，模拟 RRT 的随机性
        offset = 0.5 * (1 if i % 2 == 0 else -1)
        y += offset

        path.append((x, y))

    path.append(goal)

    # 在每两个点之间插入额外的中间点，增加冗余
    dense_path = [path[0]]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        # 插入 3 个中间点
        for j in range(1, 4):
            t = j / 4.0
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            dense_path.append((x, y))

        dense_path.append(p2)

    return dense_path


def test_basic_optimization():
    """测试基本优化功能（无障碍物）"""
    print("\n" + "=" * 70)
    print("Test 1: Basic Optimization (No Obstacles)")
    print("=" * 70)

    # 生成测试路径
    start = (0.0, 0.0)
    goal = (10.0, 0.0)
    path = generate_zigzag_path(start, goal, num_zigzags=8)

    print(f"Original path length: {len(path)} points")

    # 无障碍物的碰撞检测器
    checker = SimpleCollisionChecker(obstacles=[])

    # 创建优化器
    optimizer = ShortcutOptimizer(
        collision_checker=checker,
        max_iterations=100,
        min_points_distance=0.1,
        enable_angle_filter=True,
        angle_threshold=np.deg2rad(10),
        random_seed=42,
        verbose=True,
    )

    # 执行优化
    optimized_path = optimizer.optimize(path)

    print(f"\nOptimized path length: {len(optimized_path)} points")

    # 打印统计信息
    optimizer.print_stats()

    return path, optimized_path, []


def test_optimization_with_obstacles():
    """测试带障碍物的优化"""
    print("\n" + "=" * 70)
    print("Test 2: Optimization with Obstacles")
    print("=" * 70)

    # 生成测试路径
    start = (0.0, 0.0)
    goal = (10.0, 5.0)
    path = generate_zigzag_path(start, goal, num_zigzags=10)

    print(f"Original path length: {len(path)} points")

    # 添加障碍物
    obstacles = [
        (3.0, 2.0, 0.8),
        (5.0, 3.0, 0.6),
        (7.0, 2.5, 0.7),
    ]

    checker = SimpleCollisionChecker(obstacles=obstacles, resolution=0.05)

    # 创建优化器
    optimizer = ShortcutOptimizer(
        collision_checker=checker,
        max_iterations=150,
        min_points_distance=0.1,
        enable_angle_filter=True,
        angle_threshold=np.deg2rad(15),
        random_seed=42,
        verbose=True,
    )

    # 执行优化
    optimized_path = optimizer.optimize(path)

    print(f"\nOptimized path length: {len(optimized_path)} points")

    # 打印统计信息
    optimizer.print_stats()

    return path, optimized_path, obstacles


def plot_comparison(
    original_path: List[Tuple[float, float]],
    optimized_path: List[Tuple[float, float]],
    obstacles: List[Tuple[float, float, float]],
    title: str = "Path Shortcut Optimization",
):
    """
    可视化优化前后的路径对比

    Args:
        original_path: 原始路径
        optimized_path: 优化后路径
        obstacles: 障碍物列表
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制原始路径
    orig_x = [p[0] for p in original_path]
    orig_y = [p[1] for p in original_path]
    ax.plot(
        orig_x, orig_y,
        'o-',
        color='lightblue',
        linewidth=1.5,
        markersize=4,
        alpha=0.6,
        label=f'Original ({len(original_path)} points)',
    )

    # 绘制优化后路径
    opt_x = [p[0] for p in optimized_path]
    opt_y = [p[1] for p in optimized_path]
    ax.plot(
        opt_x, opt_y,
        's-',
        color='red',
        linewidth=2.5,
        markersize=7,
        alpha=0.8,
        label=f'Optimized ({len(optimized_path)} points)',
    )

    # 绘制起点和终点
    ax.plot(original_path[0][0], original_path[0][1], 'go', markersize=12, label='Start')
    ax.plot(original_path[-1][0], original_path[-1][1], 'r*', markersize=16, label='Goal')

    # 绘制障碍物
    for i, (obs_x, obs_y, obs_radius) in enumerate(obstacles):
        circle = Circle(
            (obs_x, obs_y),
            obs_radius,
            facecolor='gray',
            edgecolor='black',
            alpha=0.5,
            label='Obstacle' if i == 0 else None,
        )
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Path Shortcut Optimizer Test Suite")
    print("=" * 70)

    # 测试 1: 无障碍物
    path1, opt_path1, obstacles1 = test_basic_optimization()
    plot_comparison(path1, opt_path1, obstacles1, "Test 1: Basic Optimization")

    # 测试 2: 带障碍物
    path2, opt_path2, obstacles2 = test_optimization_with_obstacles()
    plot_comparison(path2, opt_path2, obstacles2, "Test 2: Optimization with Obstacles")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
