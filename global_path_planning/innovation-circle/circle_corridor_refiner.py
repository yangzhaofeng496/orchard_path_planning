#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
圆形通道路径修正模块

功能：修正穿过两个不同半径圆形障碍物之间的路径，使其经过两圆边界间隙的几何中点，
     并与通道方向平行，同时保证前后平滑连接。

作者：Claude Code
日期：2026-07-19
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class RefineParams:
    """路径修正参数"""
    adjust_before: float = 5.0        # 中心点前调整距离(m)
    adjust_after: float = 5.0         # 中心点后调整距离(m)
    safe_margin: float = 0.0          # 安全膨胀距离(m)
    w_center: float = 5.0             # 中心线代价权重
    w_parallel: float = 3.0           # 平行代价权重
    w_smooth: float = 2.0             # 平滑代价权重
    w_original: float = 0.5           # 原路径代价权重
    max_iterations: int = 100         # 最大优化迭代次数
    tolerance: float = 1e-6           # 收敛容差


class CircleCorridorRefiner:
    """圆形通道路径修正器"""

    def __init__(self, params: Optional[RefineParams] = None):
        """初始化

        Args:
            params: 修正参数
        """
        self.params = params if params is not None else RefineParams()


    def refine_path(
        self,
        path: np.ndarray,
        circle1: Tuple[float, float, float],
        circle2: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, Dict]:
        """修正路径

        Args:
            path: 原始路径，shape=(N, 3)，每个点为[x, y, yaw]
            circle1: 圆1，(x1, y1, r1)
            circle2: 圆2，(x2, y2, r2)

        Returns:
            refined_path: 修正后的路径
            info: 修正信息字典
        """
        if len(path) < 3:
            return path.copy(), {"valid": False, "reason": "Path too short"}

        # 解析圆参数
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        C1 = np.array([x1, y1])
        C2 = np.array([x2, y2])

        # 膨胀半径
        R1 = r1 + self.params.safe_margin
        R2 = r2 + self.params.safe_margin

        # 计算圆心距离和方向
        dx = x2 - x1
        dy = y2 - y1
        D = np.hypot(dx, dy)

        if D < 1e-6:
            return path.copy(), {"valid": False, "reason": "Circles coincide"}

        # 圆心连线单位向量
        u = np.array([dx / D, dy / D])

        # 通道方向（垂直于圆心连线）
        t = np.array([-u[1], u[0]])

        # 目标航向
        yaw_target = np.arctan2(t[1], t[0])

        # 检查有效间隙
        free_gap = D - R1 - R2
        if free_gap <= 0:
            return path.copy(), {
                "valid": False,
                "reason": "No valid gap between circles",
                "free_gap": free_gap
            }

        # 计算两圆边界之间的中点
        # B1 = C1 + R1 * u (圆1靠近圆2的边界点)
        # B2 = C2 - R2 * u (圆2靠近圆1的边界点)
        # Q = (B1 + B2) / 2
        s = (D + R1 - R2) / 2  # C1到Q的距离
        Q = C1 + s * u

        # 找到距离Q最近的路径点索引
        distances_to_Q = np.linalg.norm(path[:, :2] - Q, axis=1)
        center_index = int(np.argmin(distances_to_Q))

        # 计算累计路径距离
        cumulative_dist = self._compute_cumulative_distance(path)
        center_dist = cumulative_dist[center_index]

        # 确定调整区间
        start_dist = center_dist - self.params.adjust_before
        end_dist = center_dist + self.params.adjust_after

        start_index = np.searchsorted(cumulative_dist, start_dist)
        end_index = np.searchsorted(cumulative_dist, end_dist)

        start_index = max(0, start_index)
        end_index = min(len(path) - 1, end_index)

        if end_index <= start_index + 1:
            return path.copy(), {
                "valid": False,
                "reason": "Adjustment range too small"
            }

        # 计算原始误差
        original_center_error = self._compute_center_error(
            path[start_index:end_index+1], Q, u
        )
        original_angle_error = self._compute_angle_error(
            path[start_index:end_index+1], yaw_target
        )

        # 初始化修正路径
        refined_path = path.copy()

        # 第一步：使用余弦窗平滑调整位置和角度
        refined_path = self._smooth_adjustment(
            refined_path, start_index, center_index, end_index, Q, u, yaw_target
        )

        # 第二步：优化局部路径
        refined_path = self._optimize_local_path(
            refined_path, start_index, end_index, Q, u, yaw_target
        )

        # 重新计算yaw（基于几何关系）
        refined_path = self._recompute_yaw(refined_path, start_index, end_index)

        # 检查可行性
        is_valid, reason = self._check_feasibility(
            refined_path, circle1, circle2, R1, R2
        )

        if not is_valid:
            return path.copy(), {
                "valid": False,
                "reason": reason,
                "center_point": tuple(Q),
                "target_yaw": yaw_target,
                "free_gap": free_gap,
            }

        # 计算修正后误差
        refined_center_error = self._compute_center_error(
            refined_path[start_index:end_index+1], Q, u
        )
        refined_angle_error = self._compute_angle_error(
            refined_path[start_index:end_index+1], yaw_target
        )

        # 返回结果
        info = {
            "valid": True,
            "center_point": tuple(Q),
            "target_yaw": float(yaw_target),
            "free_gap": float(free_gap),
            "start_index": int(start_index),
            "center_index": int(center_index),
            "end_index": int(end_index),
            "original_center_error": float(original_center_error),
            "refined_center_error": float(refined_center_error),
            "original_angle_error": float(original_angle_error),
            "refined_angle_error": float(refined_angle_error),
            "center_error_reduction": float(
                (original_center_error - refined_center_error) /
                (original_center_error + 1e-9) * 100
            ),
            "angle_error_reduction": float(
                (original_angle_error - refined_angle_error) /
                (original_angle_error + 1e-9) * 100
            ),
        }

        return refined_path, info


    def _compute_cumulative_distance(self, path: np.ndarray) -> np.ndarray:
        """计算累计路径距离

        Args:
            path: 路径

        Returns:
            累计距离数组
        """
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(
                path[i, :2] - path[i-1, :2]
            )
        return distances


    def _smooth_adjustment(
        self,
        path: np.ndarray,
        start_idx: int,
        center_idx: int,
        end_idx: int,
        Q: np.ndarray,
        u: np.ndarray,
        yaw_target: float
    ) -> np.ndarray:
        """使用余弦窗平滑调整位置和角度

        Args:
            path: 路径
            start_idx: 起始索引
            center_idx: 中心索引
            end_idx: 结束索引
            Q: 通道中心点
            u: 圆心连线单位向量（法向量）
            yaw_target: 目标航向

        Returns:
            调整后的路径
        """
        adjusted_path = path.copy()

        for i in range(start_idx, end_idx + 1):
            P_original = path[i, :2]
            yaw_original = path[i, 2]

            # 计算到中心线的垂直误差
            e_i = np.dot(P_original - Q, u)

            # 目标位置
            P_target = P_original - e_i * u

            # 计算权重（余弦窗）
            if i <= center_idx:
                # 入口到中心：0 → 1
                progress = (i - start_idx) / max(1, center_idx - start_idx)
                weight = 0.5 - 0.5 * np.cos(np.pi * progress)
            else:
                # 中心到出口：1 → 0
                progress = (i - center_idx) / max(1, end_idx - center_idx)
                weight = 0.5 + 0.5 * np.cos(np.pi * progress)

            # 位置修正
            P_new = P_original + weight * (P_target - P_original)
            adjusted_path[i, :2] = P_new

            # 角度修正
            yaw_error = self._normalize_angle(yaw_target - yaw_original)
            yaw_new = yaw_original + weight * yaw_error
            adjusted_path[i, 2] = yaw_new

        return adjusted_path


    def _optimize_local_path(
        self,
        path: np.ndarray,
        start_idx: int,
        end_idx: int,
        Q: np.ndarray,
        u: np.ndarray,
        yaw_target: float
    ) -> np.ndarray:
        """优化局部路径

        Args:
            path: 路径
            start_idx: 起始索引
            end_idx: 结束索引
            Q: 通道中心点
            u: 法向量
            yaw_target: 目标航向

        Returns:
            优化后的路径
        """
        if not HAS_SCIPY:
            return path  # 没有scipy就跳过优化

        # 提取局部路径
        local_path = path[start_idx:end_idx+1].copy()
        original_local = local_path.copy()
        N = len(local_path)

        # 优化变量：只优化xy
        X0 = local_path[:, :2].flatten()

        # 定义代价函数
        def cost_function(X):
            X_reshaped = X.reshape(N, 2)

            # 中心线代价
            J_center = 0.0
            for i in range(N):
                e = np.dot(X_reshaped[i] - Q, u)
                J_center += e ** 2

            # 平滑代价
            J_smooth = 0.0
            for i in range(1, N - 1):
                diff = X_reshaped[i+1] - 2 * X_reshaped[i] + X_reshaped[i-1]
                J_smooth += np.sum(diff ** 2)

            # 原路径代价
            J_original = 0.0
            for i in range(N):
                diff = X_reshaped[i] - original_local[i, :2]
                J_original += np.sum(diff ** 2)

            # 总代价
            J = (self.params.w_center * J_center +
                 self.params.w_smooth * J_smooth +
                 self.params.w_original * J_original)

            return J

        # 优化
        result = minimize(
            cost_function,
            X0,
            method='L-BFGS-B',
            options={
                'maxiter': self.params.max_iterations,
                'ftol': self.params.tolerance
            }
        )

        # 更新路径
        optimized_path = path.copy()
        optimized_path[start_idx:end_idx+1, :2] = result.x.reshape(N, 2)

        return optimized_path


    def _recompute_yaw(
        self,
        path: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """重新计算yaw（基于几何关系）

        Args:
            path: 路径
            start_idx: 起始索引
            end_idx: 结束索引

        Returns:
            更新yaw后的路径
        """
        updated_path = path.copy()

        for i in range(start_idx, end_idx + 1):
            if i < len(path) - 1:
                # 使用当前点到下一点的方向
                dx = path[i+1, 0] - path[i, 0]
                dy = path[i+1, 1] - path[i, 1]
                if np.hypot(dx, dy) > 1e-6:
                    updated_path[i, 2] = np.arctan2(dy, dx)
            else:
                # 最后一点保持前一点的yaw
                if i > 0:
                    updated_path[i, 2] = updated_path[i-1, 2]

        return updated_path


    def _check_feasibility(
        self,
        path: np.ndarray,
        circle1: Tuple[float, float, float],
        circle2: Tuple[float, float, float],
        R1: float,
        R2: float
    ) -> Tuple[bool, str]:
        """检查路径可行性

        Args:
            path: 路径
            circle1: 圆1
            circle2: 圆2
            R1: 圆1膨胀半径
            R2: 圆2膨胀半径

        Returns:
            (是否可行, 原因)
        """
        x1, y1, _ = circle1
        x2, y2, _ = circle2
        C1 = np.array([x1, y1])
        C2 = np.array([x2, y2])

        # 检查碰撞
        for i in range(len(path)):
            P = path[i, :2]

            # 检查与圆1的碰撞
            dist1 = np.linalg.norm(P - C1)
            if dist1 < R1 - 1e-3:
                return False, f"Collision with circle1 at index {i}"

            # 检查与圆2的碰撞
            dist2 = np.linalg.norm(P - C2)
            if dist2 < R2 - 1e-3:
                return False, f"Collision with circle2 at index {i}"

        # 检查路径点间距
        max_step = 0.0
        for i in range(len(path) - 1):
            dist = np.linalg.norm(path[i+1, :2] - path[i, :2])
            max_step = max(max_step, dist)
            if dist > 5.0:  # 异常跳变（放宽到5m）
                return False, f"Abnormal jump at index {i} (dist={dist:.2f}m)"

        return True, "Valid"


    def _compute_center_error(
        self,
        path_segment: np.ndarray,
        Q: np.ndarray,
        u: np.ndarray
    ) -> float:
        """计算中心线误差

        Args:
            path_segment: 路径片段
            Q: 中心点
            u: 法向量

        Returns:
            平均绝对误差
        """
        errors = []
        for i in range(len(path_segment)):
            P = path_segment[i, :2]
            e = np.dot(P - Q, u)
            errors.append(abs(e))
        return float(np.mean(errors))


    def _compute_angle_error(
        self,
        path_segment: np.ndarray,
        yaw_target: float
    ) -> float:
        """计算角度误差

        Args:
            path_segment: 路径片段
            yaw_target: 目标航向

        Returns:
            平均绝对角度误差（度）
        """
        errors = []
        for i in range(len(path_segment)):
            yaw = path_segment[i, 2]
            error = abs(self._normalize_angle(yaw - yaw_target))
            errors.append(np.degrees(error))
        return float(np.mean(errors))


    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到[-pi, pi]

        Args:
            angle: 角度（弧度）

        Returns:
            归一化后的角度
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def refine_path_between_circles(
    path: np.ndarray,
    circle1: Tuple[float, float, float],
    circle2: Tuple[float, float, float],
    params: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """修正穿过两个圆形障碍物之间的路径（主接口）

    Args:
        path: 原始路径，shape=(N, 3)，每个点为[x, y, yaw]
        circle1: 圆1，(x1, y1, r1)
        circle2: 圆2，(x2, y2, r2)
        params: 可选参数字典

    Returns:
        refined_path: 修正后的路径
        info: 修正信息字典
    """
    # 解析参数
    if params is None:
        refine_params = RefineParams()
    else:
        refine_params = RefineParams(**params)

    # 创建修正器
    refiner = CircleCorridorRefiner(refine_params)

    # 执行修正
    return refiner.refine_path(path, circle1, circle2)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("圆形通道路径修正模块 - 测试")
    print("=" * 80)

    # 创建测试场景
    # 两个不同半径的圆
    circle1 = (0.0, 2.0, 0.8)   # 上方，较小半径
    circle2 = (0.0, -2.0, 1.2)  # 下方，较大半径

    # 构造一条倾斜穿过两圆之间的路径
    N = 50
    x = np.linspace(-5, 5, N)
    y = 0.5 * x  # 倾斜路径
    yaw = np.full(N, np.arctan2(0.5, 1.0))  # 初始倾斜角度

    original_path = np.column_stack([x, y, yaw])

    print(f"\n测试场景：")
    print(f"  圆1: center=({circle1[0]:.1f}, {circle1[1]:.1f}), r={circle1[2]:.1f}")
    print(f"  圆2: center=({circle2[0]:.1f}, {circle2[1]:.1f}), r={circle2[2]:.1f}")
    print(f"  原始路径: {N}个点，倾斜穿过两圆")

    # 执行修正
    refined_path, info = refine_path_between_circles(
        original_path,
        circle1,
        circle2
    )

    # 输出结果
    print(f"\n修正结果：")
    print(f"  有效: {info['valid']}")
    if info['valid']:
        print(f"  通道中心点: ({info['center_point'][0]:.3f}, {info['center_point'][1]:.3f})")
        print(f"  目标航向: {np.degrees(info['target_yaw']):.1f}°")
        print(f"  自由间隙: {info['free_gap']:.3f}m")
        print(f"  调整区间: [{info['start_index']}, {info['end_index']}]")
        print(f"  中心索引: {info['center_index']}")
        print(f"\n误差改善：")
        print(f"  中心线误差: {info['original_center_error']:.3f}m → {info['refined_center_error']:.3f}m")
        print(f"  改善率: {info['center_error_reduction']:.1f}%")
        print(f"  角度误差: {info['original_angle_error']:.1f}° → {info['refined_angle_error']:.1f}°")
        print(f"  改善率: {info['angle_error_reduction']:.1f}%")

        # 验证关键点
        center_idx = info['center_index']
        center_point = refined_path[center_idx, :2]
        Q = np.array(info['center_point'])
        distance_to_Q = np.linalg.norm(center_point - Q)
        print(f"\n验证：")
        print(f"  路径中心点到通道中心距离: {distance_to_Q:.3f}m")

        # 可视化
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 左图：路径对比
            ax = axes[0]

            # 绘制圆
            circle1_patch = plt.Circle(
                (circle1[0], circle1[1]), circle1[2],
                fill=False, edgecolor='red', linewidth=2, label='Circle 1'
            )
            circle2_patch = plt.Circle(
                (circle2[0], circle2[1]), circle2[2],
                fill=False, edgecolor='blue', linewidth=2, label='Circle 2'
            )
            ax.add_patch(circle1_patch)
            ax.add_patch(circle2_patch)

            # 绘制路径
            ax.plot(original_path[:, 0], original_path[:, 1],
                   'gray', linewidth=2, alpha=0.5, label='Original')
            ax.plot(refined_path[:, 0], refined_path[:, 1],
                   'green', linewidth=2, label='Refined')

            # 标记通道中心点
            ax.plot(Q[0], Q[1], 'r*', markersize=15, label='Corridor Center')

            # 标记调整区间
            start_idx = info['start_index']
            end_idx = info['end_index']
            ax.plot(refined_path[start_idx, 0], refined_path[start_idx, 1],
                   'bo', markersize=8, label='Adjust Start')
            ax.plot(refined_path[end_idx, 0], refined_path[end_idx, 1],
                   'mo', markersize=8, label='Adjust End')

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Path Refinement')
            ax.legend()
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # 右图：误差对比
            ax = axes[1]

            # 计算误差
            dx = circle2[0] - circle1[0]
            dy = circle2[1] - circle1[1]
            D = np.hypot(dx, dy)
            u = np.array([dx / D, dy / D])

            original_errors = []
            refined_errors = []
            for i in range(len(original_path)):
                P_orig = original_path[i, :2]
                P_ref = refined_path[i, :2]
                e_orig = abs(np.dot(P_orig - Q, u))
                e_ref = abs(np.dot(P_ref - Q, u))
                original_errors.append(e_orig)
                refined_errors.append(e_ref)

            indices = np.arange(len(original_path))
            ax.plot(indices, original_errors, 'gray', linewidth=2, label='Original')
            ax.plot(indices, refined_errors, 'green', linewidth=2, label='Refined')
            ax.axvline(start_idx, color='b', linestyle='--', alpha=0.5, label='Adjust Range')
            ax.axvline(end_idx, color='b', linestyle='--', alpha=0.5)
            ax.axvline(center_idx, color='r', linestyle='--', alpha=0.5, label='Center')

            ax.set_xlabel('Path Index')
            ax.set_ylabel('Center Line Error (m)')
            ax.set_title('Error Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('circle_corridor_test.png', dpi=150, bbox_inches='tight')
            print(f"\n可视化已保存: circle_corridor_test.png")

        except ImportError:
            print("\n(matplotlib未安装，跳过可视化)")

    else:
        print(f"  原因: {info.get('reason', 'Unknown')}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
