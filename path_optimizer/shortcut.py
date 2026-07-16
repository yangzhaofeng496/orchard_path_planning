"""
Path Shortcut Optimizer

实现 RRT 路径的后处理优化，通过以下策略减少冗余节点：
1. Remove Close Points: 删除距离过近的节点
2. Collinear Point Removal: 删除近似共线的节点
3. Random Shortcut: 随机尝试连接远距离节点，跳过中间节点

核心思想：
- RRT 生成的路径往往包含大量冗余节点
- 通过直接连接可行的远距离节点，可以大幅简化路径
- 保持碰撞检测确保优化后路径仍然可行
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Protocol
from dataclasses import dataclass


class CollisionChecker(Protocol):
    """
    碰撞检测接口（抽象基类）

    这是一个协议类，用于解耦路径优化器和具体的碰撞检测实现。
    支持未来扩展：
    - 简单直线检测
    - Dubins 曲线检测
    - Reeds-Shepp 曲线检测
    - 阿克曼车辆模型约束
    """

    def check_line(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        检查从 p1 到 p2 的直线路径是否无碰撞

        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)

        Returns:
            True 表示无碰撞，False 表示有碰撞
        """
        ...


@dataclass
class OptimizationStats:
    """优化统计信息"""
    original_points: int
    optimized_points: int
    removed_points: int
    reduction_ratio: float
    close_points_removed: int
    collinear_points_removed: int
    shortcut_points_removed: int


class ShortcutOptimizer:
    """
    路径捷径优化器

    使用随机采样策略，尝试连接路径中的远距离节点，
    跳过中间的冗余节点，从而简化路径。

    算法流程：
    1. 预处理：删除距离过近的点
    2. 预处理：删除共线点
    3. 主循环：随机选择两个节点尝试直接连接
    4. 验证：通过碰撞检测确保连接可行
    5. 更新：删除中间节点，更新路径

    Example:
        >>> checker = MyCollisionChecker(obstacles)
        >>> optimizer = ShortcutOptimizer(
        ...     collision_checker=checker,
        ...     max_iterations=100,
        ...     random_seed=42
        ... )
        >>> simplified_path = optimizer.optimize(original_path)
        >>> print(f"Reduced from {len(original_path)} to {len(simplified_path)} points")
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        max_iterations: int = 100,
        min_points_distance: float = 0.1,
        enable_angle_filter: bool = True,
        angle_threshold: float = np.deg2rad(10),
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        初始化路径捷径优化器

        Args:
            collision_checker: 碰撞检测器，需实现 check_line 接口
            max_iterations: 随机捷径尝试的最大迭代次数
            min_points_distance: 最小点间距，小于此距离的点会被删除
            enable_angle_filter: 是否启用共线点过滤
            angle_threshold: 共线判定角度阈值（弧度），小于此角度认为共线
            random_seed: 随机种子，用于保证结果可复现
            verbose: 是否打印详细优化信息
        """
        self.collision_checker = collision_checker
        self.max_iterations = max_iterations
        self.min_points_distance = min_points_distance
        self.enable_angle_filter = enable_angle_filter
        self.angle_threshold = angle_threshold
        self.verbose = verbose

        # 设置随机种子以保证可复现性
        if random_seed is not None:
            np.random.seed(random_seed)

        # 统计信息
        self.stats: Optional[OptimizationStats] = None

    def optimize(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        优化路径，删除冗余节点

        Args:
            path: 原始路径，格式为 [(x0, y0), (x1, y1), ..., (xn, yn)]

        Returns:
            优化后的路径，保持起点和终点不变
        """
        if len(path) < 3:
            # 少于 3 个点无需优化
            self._create_stats(path, path, 0, 0, 0)
            return path.copy()

        original_count = len(path)

        # 步骤 1: 删除距离过近的点
        path = self._remove_close_points(path)
        close_removed = original_count - len(path)

        if self.verbose:
            print(f"[Shortcut] Step 1: Removed {close_removed} close points")
            print(f"           Remaining: {len(path)} points")

        # 步骤 2: 删除共线点
        collinear_removed = 0
        if self.enable_angle_filter and len(path) >= 3:
            before_collinear = len(path)
            path = self._remove_collinear_points(path)
            collinear_removed = before_collinear - len(path)

            if self.verbose:
                print(f"[Shortcut] Step 2: Removed {collinear_removed} collinear points")
                print(f"           Remaining: {len(path)} points")

        # 步骤 3: 随机捷径优化
        before_shortcut = len(path)
        path = self._random_shortcut(path)
        shortcut_removed = before_shortcut - len(path)

        if self.verbose:
            print(f"[Shortcut] Step 3: Removed {shortcut_removed} points via shortcut")
            print(f"           Remaining: {len(path)} points")

        # 创建统计信息
        self._create_stats(
            original_path=None,  # 传递计数
            optimized_path=path,
            close_removed=close_removed,
            collinear_removed=collinear_removed,
            shortcut_removed=shortcut_removed,
            original_count=original_count,
        )

        return path

    def _remove_close_points(
        self, path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        删除距离过近的节点

        策略：遍历路径，如果当前点与前一个点的距离小于阈值，则删除当前点
        注意：起点和终点永远保留

        Args:
            path: 输入路径

        Returns:
            过滤后的路径
        """
        if len(path) < 3:
            return path.copy()

        filtered = [path[0]]  # 保留起点

        for i in range(1, len(path) - 1):
            dist = self._euclidean_distance(filtered[-1], path[i])
            if dist >= self.min_points_distance:
                filtered.append(path[i])

        # 保留终点
        filtered.append(path[-1])

        return filtered

    def _remove_collinear_points(
        self, path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        删除近似共线的节点

        策略：对于连续三个点 A-B-C，计算向量 AB 和 BC 的夹角
        如果夹角小于阈值，认为 B 点冗余，可以删除

        几何意义：
        - 如果 A、B、C 几乎在一条直线上，B 点不提供额外的路径信息
        - 删除 B 后，A 直接连接 C，路径更简洁

        Args:
            path: 输入路径

        Returns:
            过滤后的路径
        """
        if len(path) < 3:
            return path.copy()

        filtered = [path[0]]  # 保留起点

        i = 1
        while i < len(path) - 1:
            p_prev = filtered[-1]
            p_curr = path[i]
            p_next = path[i + 1]

            angle = self._calculate_angle(p_prev, p_curr, p_next)

            # 如果夹角小于阈值，说明近似共线，跳过当前点
            if angle > self.angle_threshold:
                filtered.append(p_curr)

            i += 1

        # 保留终点
        filtered.append(path[-1])

        return filtered

    def _random_shortcut(
        self, path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        随机捷径优化（核心算法）

        策略：
        1. 随机选择路径中的两个索引 i < j
        2. 检查从 path[i] 到 path[j] 的直线是否无碰撞
        3. 如果无碰撞，删除 path[i+1] 到 path[j-1] 之间的所有点
        4. 重复 max_iterations 次

        为什么有效：
        - RRT 生成的路径经过随机采样，节点分布不均
        - 很多中间节点是为了避障而生成的，但实际可以直接连接
        - 通过随机尝试，可以发现这些捷径机会

        Args:
            path: 输入路径

        Returns:
            优化后的路径
        """
        if len(path) < 3:
            return path.copy()

        current_path = path.copy()

        for iteration in range(self.max_iterations):
            if len(current_path) < 3:
                break  # 无法继续简化

            # 随机选择两个索引，确保 i < j 且至少间隔 1 个点
            n = len(current_path)
            i = np.random.randint(0, n - 2)
            j = np.random.randint(i + 2, n)

            p_i = current_path[i]
            p_j = current_path[j]

            # 检查直线连接是否无碰撞
            if self.collision_checker.check_line(p_i, p_j):
                # 可以直接连接，删除中间节点
                current_path = current_path[:i+1] + current_path[j:]

                if self.verbose and (iteration + 1) % 20 == 0:
                    print(f"           Iteration {iteration + 1}: "
                          f"Connected point {i} to {j}, "
                          f"current length: {len(current_path)}")

        return current_path

    def _euclidean_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """计算两点之间的欧几里得距离"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _calculate_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> float:
        """
        计算三个点形成的夹角

        计算向量 p1->p2 和 p2->p3 的夹角

        Returns:
            夹角（弧度），范围 [0, π]
        """
        # 向量 p1 -> p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # 向量 p2 -> p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量长度
        len_v1 = math.hypot(v1[0], v1[1])
        len_v2 = math.hypot(v2[0], v2[1])

        # 避免除零
        if len_v1 < 1e-9 or len_v2 < 1e-9:
            return 0.0

        # 计算点积
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]

        # 计算夹角的余弦值
        cos_angle = dot_product / (len_v1 * len_v2)

        # 限制在 [-1, 1] 范围内，避免数值误差
        cos_angle = max(-1.0, min(1.0, cos_angle))

        # 返回夹角（弧度）
        angle = math.acos(cos_angle)

        return angle

    def _create_stats(
        self,
        original_path: Optional[List[Tuple[float, float]]],
        optimized_path: List[Tuple[float, float]],
        close_removed: int,
        collinear_removed: int,
        shortcut_removed: int,
        original_count: Optional[int] = None,
    ):
        """创建优化统计信息"""
        if original_count is None:
            original_count = len(original_path) if original_path else 0

        optimized_count = len(optimized_path)
        removed = original_count - optimized_count
        reduction = (removed / original_count * 100) if original_count > 0 else 0.0

        self.stats = OptimizationStats(
            original_points=original_count,
            optimized_points=optimized_count,
            removed_points=removed,
            reduction_ratio=reduction,
            close_points_removed=close_removed,
            collinear_points_removed=collinear_removed,
            shortcut_points_removed=shortcut_removed,
        )

    def print_stats(self):
        """打印优化统计信息"""
        if self.stats is None:
            print("No optimization has been performed yet.")
            return

        print("\n" + "=" * 70)
        print("Path Shortcut Optimization Statistics")
        print("=" * 70)
        print(f"Original points:              {self.stats.original_points}")
        print(f"Optimized points:             {self.stats.optimized_points}")
        print(f"Removed points:               {self.stats.removed_points}")
        print(f"Reduction ratio:              {self.stats.reduction_ratio:.2f}%")
        print(f"  - Close points removed:     {self.stats.close_points_removed}")
        print(f"  - Collinear points removed: {self.stats.collinear_points_removed}")
        print(f"  - Shortcut points removed:  {self.stats.shortcut_points_removed}")
        print("=" * 70 + "\n")
