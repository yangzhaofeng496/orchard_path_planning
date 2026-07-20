#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于圆形障碍物模型的狭窄通道中心引导全局路径优化模块

针对果园随机分布环境，障碍物采用圆形模型（树木）。
核心优化对象为"树圆之间形成的通道"，使车辆路径尽可能位于树木通道中心，
同时保持原路径连续性。

核心功能：
1. 狭窄通道检测（基于树间通道宽度）
2. 树间通道中心计算
3. 路径到通道中心的横向误差计算
4. 路径中心化优化（3项代价函数）
5. 路径约束检查

作者: Claude Code
日期: 2026-07-19
版本: v2.1.0（圆形障碍物模型）
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

# 尝试导入scipy用于优化
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, using gradient descent fallback")


@dataclass
class OptimizeParams:
    """优化参数"""
    safe_distance: float = 0.5         # 安全距离(m)
    narrow_threshold: float = 3.5      # 狭窄通道阈值(m)
    alpha: float = 0.5                 # 调整比例(0~1)
    w_center: float = 10.0             # 居中权重（增大以强化居中效果）
    w_smooth: float = 0.5              # 平滑权重
    w_original: float = 0.1            # 原路径权重（降低以允许更大调整）
    min_turning_radius: float = 3.0    # 最小转弯半径(m)
    max_iterations: int = 200          # 最大迭代次数（增加）
    tolerance: float = 1e-6            # 收敛容差（更严格）


class NarrowPathRefiner:
    """狭窄通道路径优化器（圆形障碍物模型）"""

    def __init__(self, params: Optional[OptimizeParams] = None):
        """初始化

        Args:
            params: 优化参数，None则使用默认值
        """
        self.params = params if params is not None else OptimizeParams()

    def detect_narrow_segments(
        self,
        path: np.ndarray,
        obstacles: List[Tuple[float, float, float]]
    ) -> Tuple[List[Tuple[int, int]], List[float], List[Tuple], List[Tuple]]:
        """检测狭窄区域

        基于树间通道宽度判断狭窄区域。

        Args:
            path: 路径点数组 (N, 3) [x, y, yaw]
            obstacles: 圆形障碍物列表 [(x, y, r), ...]

        Returns:
            narrow_segments: 狭窄区域列表 [(start_idx, end_idx), ...]
            corridor_widths: 每个点的通道宽度
            left_trees: 每个点的左侧树 [(x, y, r), ...]
            right_trees: 每个点的右侧树 [(x, y, r), ...]
        """
        N = len(path)
        corridor_widths = []
        left_trees = []
        right_trees = []
        is_narrow = np.zeros(N, dtype=bool)

        for i, (x, y, yaw) in enumerate(path):
            # 寻找左右两侧最近的树
            left_tree, right_tree = self._find_nearest_left_right_trees(
                x, y, yaw, obstacles
            )

            left_trees.append(left_tree)
            right_trees.append(right_tree)

            # 计算树间通道宽度
            if left_tree and right_tree:
                # 两树中心距离
                xL, yL, rL = left_tree
                xR, yR, rR = right_tree
                dist = np.hypot(xR - xL, yR - yL)

                # 有效通道宽度 = 距离 - 两半径 - 2*安全距离
                width = dist - rL - rR - 2 * self.params.safe_distance
                corridor_widths.append(width)

                # 判断是否狭窄
                if width < self.params.narrow_threshold:
                    is_narrow[i] = True
            else:
                corridor_widths.append(float('inf'))  # 没有两侧树，宽通道

        # 合并连续的狭窄区域
        narrow_segments = self._merge_narrow_regions(is_narrow)

        return narrow_segments, corridor_widths, left_trees, right_trees


    def _find_nearest_left_right_trees(
        self,
        x: float,
        y: float,
        yaw: float,
        obstacles: List[Tuple[float, float, float]]
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """寻找路径点左右两侧最近的树

        利用车辆坐标系判断左右：
        local_y = -sin(yaw)*dx + cos(yaw)*dy
        local_y > 0: 左侧
        local_y < 0: 右侧

        Args:
            x, y, yaw: 路径点位置和朝向
            obstacles: 障碍物列表

        Returns:
            left_tree: 左侧最近树 (x, y, r) 或 None
            right_tree: 右侧最近树 (x, y, r) 或 None
        """
        left_tree = None
        right_tree = None
        min_left_dist = float('inf')
        min_right_dist = float('inf')

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        for obs in obstacles:
            ox, oy, r = obs
            dx = ox - x
            dy = oy - y

            # 转换到车辆坐标系
            local_y = -sin_yaw * dx + cos_yaw * dy

            # 计算距离
            dist = np.hypot(dx, dy)

            if local_y > 0:  # 左侧
                if dist < min_left_dist:
                    min_left_dist = dist
                    left_tree = obs
            elif local_y < 0:  # 右侧
                if dist < min_right_dist:
                    min_right_dist = dist
                    right_tree = obs

        return left_tree, right_tree

    def _merge_narrow_regions(self, is_narrow: np.ndarray) -> List[Tuple[int, int]]:
        """合并连续的狭窄区域

        Args:
            is_narrow: 布尔数组，标记每个点是否狭窄

        Returns:
            狭窄区域列表 [(start_idx, end_idx), ...]
        """
        segments = []
        in_segment = False
        start_idx = 0

        for i, narrow in enumerate(is_narrow):
            if narrow and not in_segment:
                start_idx = i
                in_segment = True
            elif not narrow and in_segment:
                segments.append((start_idx, i - 1))
                in_segment = False

        # 处理最后一个区域
        if in_segment:
            segments.append((start_idx, len(is_narrow) - 1))

        return segments


    def find_corridor_center(
        self,
        path: np.ndarray,
        left_trees: List[Tuple],
        right_trees: List[Tuple]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """计算树间通道中心和横向误差

        对于左右树：C = (left_tree_center + right_tree_center) / 2
        通道方向：v = right_tree_center - left_tree_center
        通道法向量：n = [-v_y, v_x] / ||v||
        横向误差：error = (P - C) · n

        Args:
            path: 路径点数组 (N, 3)
            left_trees: 左侧树列表
            right_trees: 右侧树列表

        Returns:
            centers: 通道中心列表 [np.array([x, y]), ...]
            normals: 通道法向量列表 [np.array([nx, ny]), ...]
            errors: 横向误差列表 [float, ...]
        """
        N = len(path)
        centers = []
        normals = []
        errors = []

        for i in range(N):
            left_tree = left_trees[i]
            right_tree = right_trees[i]

            if left_tree and right_tree:
                # 两树中心
                xL, yL, rL = left_tree
                xR, yR, rR = right_tree

                # 通道中心
                center = np.array([(xL + xR) / 2, (yL + yR) / 2])

                # 通道方向向量
                v = np.array([xR - xL, yR - yL])
                v_norm = np.linalg.norm(v)

                if v_norm > 1e-6:
                    # 通道法向量（垂直于通道方向）
                    normal = np.array([-v[1], v[0]]) / v_norm

                    # 路径点
                    P = np.array([path[i, 0], path[i, 1]])

                    # 横向误差
                    error = np.dot(P - center, normal)

                    centers.append(center)
                    normals.append(normal)
                    errors.append(error)
                else:
                    # 两树重叠，无法计算
                    centers.append(None)
                    normals.append(None)
                    errors.append(0.0)
            else:
                # 没有两侧树
                centers.append(None)
                normals.append(None)
                errors.append(0.0)

        return centers, normals, errors


    def compute_offset(
        self,
        errors: List[float],
        normals: List[np.ndarray],
        narrow_segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """计算路径中心化调整偏移

        只调整狭窄区域内的路径点：
        offset = -alpha * error
        P_new = P + offset * n

        Args:
            errors: 横向误差列表
            normals: 法向量列表
            narrow_segments: 狭窄区域列表

        Returns:
            offsets: 偏移向量数组 (N, 2)
        """
        N = len(errors)
        offsets = np.zeros((N, 2))

        for start_idx, end_idx in narrow_segments:
            for i in range(start_idx, end_idx + 1):
                if normals[i] is not None:
                    # 计算偏移量：-alpha * error
                    offset_magnitude = -self.params.alpha * errors[i]
                    # 偏移方向：法向量
                    offsets[i] = offset_magnitude * normals[i]

        return offsets

    def optimize_path(
        self,
        path: np.ndarray,
        centers: List[np.ndarray],
        normals: List[np.ndarray],
        narrow_segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """路径优化

        目标函数：
        J = w1*J_center + w2*J_smooth + w3*J_original

        其中：
        J_center: 路径靠近树间中心
        J_smooth: 路径平滑（二阶差分）
        J_original: 不偏离原路径

        Args:
            path: 原始路径 (N, 3)
            centers: 通道中心列表
            normals: 法向量列表
            narrow_segments: 狭窄区域列表

        Returns:
            optimized_path: 优化后的路径 (N, 3)
        """
        N = len(path)
        original_path = path.copy()

        # 只优化狭窄区域的xy，yaw保持不变
        X0 = path[:, :2].flatten()  # [x1, y1, x2, y2, ...]

        # 定义优化目标函数
        def cost_function(X):
            X_reshaped = X.reshape(N, 2)
            cost = 0.0

            # 1. 中心约束：路径靠近树间中心
            center_cost = 0.0
            for start_idx, end_idx in narrow_segments:
                for i in range(start_idx, end_idx + 1):
                    if centers[i] is not None and normals[i] is not None:
                        P = X_reshaped[i]
                        error = np.dot(P - centers[i], normals[i])
                        center_cost += error ** 2
            cost += self.params.w_center * center_cost

            # 2. 平滑约束：二阶差分
            smooth_cost = 0.0
            for i in range(1, N - 1):
                diff = X_reshaped[i + 1] - 2 * X_reshaped[i] + X_reshaped[i - 1]
                smooth_cost += np.sum(diff ** 2)
            cost += self.params.w_smooth * smooth_cost

            # 3. 原路径约束：不偏离过大
            original_cost = 0.0
            for i in range(N):
                diff = X_reshaped[i] - original_path[i, :2]
                original_cost += np.sum(diff ** 2)
            cost += self.params.w_original * original_cost

            return cost

        # 使用scipy优化或梯度下降
        if HAS_SCIPY:
            result = minimize(
                cost_function,
                X0,
                method='L-BFGS-B',
                options={'maxiter': self.params.max_iterations, 'ftol': self.params.tolerance}
            )
            X_opt = result.x
        else:
            X_opt = self._gradient_descent(cost_function, X0)

        # 重构路径
        optimized_path = path.copy()
        optimized_path[:, :2] = X_opt.reshape(N, 2)

        return optimized_path


    def _gradient_descent(self, cost_function, X0: np.ndarray) -> np.ndarray:
        """梯度下降优化（scipy不可用时的后备方案）

        Args:
            cost_function: 代价函数
            X0: 初始值

        Returns:
            优化后的X
        """
        X = X0.copy()
        learning_rate = 0.01
        epsilon = 1e-8

        for iteration in range(self.params.max_iterations):
            # 数值梯度
            grad = np.zeros_like(X)
            f0 = cost_function(X)

            for i in range(len(X)):
                X_plus = X.copy()
                X_plus[i] += epsilon
                grad[i] = (cost_function(X_plus) - f0) / epsilon

            # 更新
            X_new = X - learning_rate * grad

            # 检查收敛
            if np.linalg.norm(X_new - X) < self.params.tolerance:
                break

            X = X_new

        return X

    def check_path(self, path: np.ndarray) -> Tuple[bool, str, float]:
        """检查路径约束

        检查：
        1. 路径点间距是否连续
        2. 曲率是否超过Ackermann车辆限制

        Args:
            path: 路径 (N, 3)

        Returns:
            is_valid: 是否有效
            reason: 原因
            max_curvature: 最大曲率
        """
        N = len(path)

        # 检查路径点间距
        for i in range(N - 1):
            dist = np.hypot(path[i + 1, 0] - path[i, 0], path[i + 1, 1] - path[i, 1])
            if dist > 2.0:  # 间距过大
                return False, "Path discontinuous", 0.0

        # 检查曲率
        max_curvature = 0.0
        for i in range(1, N - 1):
            curvature = self._compute_curvature(
                path[i - 1, :2], path[i, :2], path[i + 1, :2]
            )
            max_curvature = max(max_curvature, abs(curvature))

        # 最大曲率限制
        k_max = 1.0 / self.params.min_turning_radius
        if max_curvature > k_max * 1.5:  # 允许一定裕度
            return False, f"Curvature too high: {max_curvature:.3f} > {k_max:.3f}", max_curvature

        return True, "Valid", max_curvature

    def _compute_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算三点曲率

        k = 2A / (abc)

        Args:
            p1, p2, p3: 三个点

        Returns:
            曲率
        """
        # 三边长度
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0

        # 三角形面积（海伦公式）
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)

        if area_squared <= 0:
            return 0.0

        area = np.sqrt(area_squared)

        # 曲率
        curvature = 2 * area / (a * b * c)

        return curvature


def refine_global_path(
    global_path: np.ndarray,
    obstacles: List[Tuple[float, float, float]],
    params: Optional[OptimizeParams] = None
) -> Tuple[np.ndarray, Dict]:
    """狭窄通道路径优化主函数（即插即用接口）

    Args:
        global_path: 原始全局路径，numpy数组 (N, 3) [x, y, yaw]
        obstacles: 圆形障碍物列表 [(x, y, r), ...]
        params: 优化参数，None则使用默认值

    Returns:
        refined_path: 优化后的路径 (N, 3)
        info: 信息字典 {
            "is_narrow": bool,
            "narrow_segments": [(start, end), ...],
            "min_corridor_width": float,
            "max_offset": float,
            "success": bool,
            "reason": str,
            "max_curvature": float
        }
    """
    # 输入检查
    if len(global_path) < 3:
        return global_path, {
            "is_narrow": False,
            "narrow_segments": [],
            "min_corridor_width": float('inf'),
            "max_offset": 0.0,
            "success": True,
            "reason": "Path too short",
            "max_curvature": 0.0
        }

    # 转换为numpy数组
    path = np.array(global_path)

    # 初始化优化器
    refiner = NarrowPathRefiner(params)

    # 1. 检测狭窄区域
    narrow_segments, corridor_widths, left_trees, right_trees = refiner.detect_narrow_segments(
        path, obstacles
    )

    is_narrow = len(narrow_segments) > 0
    min_corridor_width = min(corridor_widths) if corridor_widths else float('inf')

    # 如果没有狭窄区域，直接返回原路径
    if not is_narrow:
        return path, {
            "is_narrow": False,
            "narrow_segments": [],
            "min_corridor_width": min_corridor_width,
            "max_offset": 0.0,
            "success": True,
            "reason": "No narrow region detected",
            "max_curvature": 0.0
        }

    # 2. 计算树间通道中心和横向误差
    centers, normals, errors = refiner.find_corridor_center(path, left_trees, right_trees)

    # 3. 优化路径
    optimized_path = refiner.optimize_path(path, centers, normals, narrow_segments)

    # 4. 检查路径约束
    is_valid, reason, max_curvature = refiner.check_path(optimized_path)

    # 如果优化失败，返回原路径
    if not is_valid:
        return path, {
            "is_narrow": True,
            "narrow_segments": narrow_segments,
            "min_corridor_width": min_corridor_width,
            "max_offset": 0.0,
            "success": False,
            "reason": f"Optimization failed ({reason}), original path returned",
            "max_curvature": max_curvature
        }

    # 计算最大偏移
    max_offset = 0.0
    for i in range(len(path)):
        offset = np.linalg.norm(optimized_path[i, :2] - path[i, :2])
        max_offset = max(max_offset, offset)

    return optimized_path, {
        "is_narrow": True,
        "narrow_segments": narrow_segments,
        "min_corridor_width": min_corridor_width,
        "max_offset": max_offset,
        "success": True,
        "reason": "Optimization successful",
        "max_curvature": max_curvature
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """测试：两排树木形成的通道"""

    print("=" * 80)
    print("基于圆形障碍物模型的狭窄通道路径优化 - 测试")
    print("=" * 80)

    # 生成测试场景：两排树木
    obstacles = []
    for x in np.arange(0, 21, 2.0):
        obstacles.append((x, 2.0, 0.3))   # 左侧树
        obstacles.append((x, -2.0, 0.3))  # 右侧树

    # 生成偏向右侧的路径
    N = 50
    x = np.linspace(0, 20, N)
    y = np.ones(N) * (-0.5)  # 偏向右侧
    yaw = np.zeros(N)
    global_path = np.column_stack([x, y, yaw])

    print(f"\n测试场景：")
    print(f"  障碍物数量: {len(obstacles)}")
    print(f"  路径点数量: {N}")
    print(f"  路径初始偏移: y = -0.5m (偏右)")

    # 优化前统计
    print(f"\n优化前：")
    original_errors = []
    for i in range(N):
        # 通道中心在y=0
        error = abs(global_path[i, 1] - 0.0)
        original_errors.append(error)

    original_avg_error = np.mean(original_errors)
    original_max_error = np.max(original_errors)
    print(f"  平均中心偏差: {original_avg_error:.3f}m")
    print(f"  最大中心偏差: {original_max_error:.3f}m")

    # 调用优化
    refined_path, info = refine_global_path(global_path, obstacles)

    # 输出结果
    print(f"\n优化结果：")
    print(f"  是否狭窄区域: {info['is_narrow']}")
    print(f"  狭窄区域数量: {len(info['narrow_segments'])}")
    if info['narrow_segments']:
        print(f"  狭窄区域范围: {info['narrow_segments']}")
    print(f"  最小通道宽度: {info['min_corridor_width']:.3f}m")
    print(f"  优化成功: {info['success']}")
    print(f"  原因: {info['reason']}")
    print(f"  最大偏移: {info['max_offset']:.3f}m")
    print(f"  最大曲率: {info['max_curvature']:.6f}")

    # 优化后统计（正确计算）
    if info['success']:
        # 重新创建refiner以获取centers和normals
        refiner = NarrowPathRefiner()
        narrow_segs, widths, left_trees, right_trees = refiner.detect_narrow_segments(global_path, obstacles)
        centers, normals, errors_original = refiner.find_corridor_center(global_path, left_trees, right_trees)
        _, _, errors_refined = refiner.find_corridor_center(refined_path, left_trees, right_trees)

        # 计算真实的横向误差（沿法向量）
        original_lateral_errors = [abs(e) for e in errors_original if e != 0.0]
        refined_lateral_errors = [abs(e) for e in errors_refined if e != 0.0]

        if original_lateral_errors and refined_lateral_errors:
            original_avg = np.mean(original_lateral_errors)
            original_max = np.max(original_lateral_errors)
            refined_avg = np.mean(refined_lateral_errors)
            refined_max = np.max(refined_lateral_errors)

            print(f"\n优化后（真实横向误差）：")
            print(f"  优化前平均偏差: {original_avg:.3f}m")
            print(f"  优化前最大偏差: {original_max:.3f}m")
            print(f"  优化后平均偏差: {refined_avg:.3f}m")
            print(f"  优化后最大偏差: {refined_max:.3f}m")

            # 改善率
            improvement = (original_avg - refined_avg) / original_avg * 100
            print(f"\n改善：")
            print(f"  平均偏差改善: {improvement:.1f}%")
            print(f"  ({original_avg:.3f}m → {refined_avg:.3f}m)")

        # 输出一些样本点的变化
        print(f"\n样本点变化（每10个点）：")
        for i in range(0, N, 10):
            dx = refined_path[i, 0] - global_path[i, 0]
            dy = refined_path[i, 1] - global_path[i, 1]
            dist = np.hypot(dx, dy)
            print(f"  点{i}: (Δx={dx:+.3f}m, Δy={dy:+.3f}m, 距离={dist:.3f}m)")

    # 可视化
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制障碍物
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color='green', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(ox, oy, 'go', markersize=3)

        # 绘制原始路径
        ax.plot(global_path[:, 0], global_path[:, 1], 'b--',
                linewidth=2, label='Original Path', alpha=0.7)

        # 绘制优化路径
        if info['success']:
            ax.plot(refined_path[:, 0], refined_path[:, 1], 'r-',
                    linewidth=2, label='Optimized Path')

        # 绘制通道中心线
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1,
                   label='Corridor Center')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Narrow Corridor Path Refinement (Circular Obstacle Model)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig('narrow_path_test.png', dpi=150)
        print(f"\n可视化已保存: narrow_path_test.png")

    except ImportError:
        print("\nmatplotlib未安装，跳过可视化")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

