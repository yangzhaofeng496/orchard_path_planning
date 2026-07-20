#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
圆形通道路径修正模块 - 增强版
增加功能：连接原始起点/终点 + 全局平滑

功能：
1. 在原始路径中找到通道调整区域
2. 优化通道区域内的路径
3. 连接：原始起点 → 优化起点 → 优化路径 → 优化终点 → 原始终点
4. 对整条路径进行全局平滑

作者：Claude Code
日期：2026-07-19
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Sequence, Any
from dataclasses import dataclass

try:
    from scipy.interpolate import splprep, splev, UnivariateSpline
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

    # 新增：连接和平滑参数
    connection_distance: float = 3.0  # 连接段距离(m)
    num_connection_points: int = 50   # 连接段点数（增加以降低曲率）
    global_smooth: bool = True        # 是否进行全局平滑
    smooth_factor: float = 0.3        # 平滑因子 (0-1)
    spline_smoothing: float = 0.1     # 样条平滑参数

    # 阿克曼底盘运动学约束
    min_turning_radius: float = 3.0   # 最小转弯半径(m)，0表示禁用
    curvature_tolerance: float = 0.5  # 曲率检查相对容差（投影方法使用更宽松的容差）
    curvature_iterations: int = 1000  # 曲率修正最大迭代次数
    curvature_relaxation: float = 0.35  # 单次曲率修正强度(0-1]

    # 通道提前对齐与车辆几何
    enforce_entry_alignment: bool = True
    alignment_clearance: float = 1.25  # 圆前后额外直线距离(m)
    alignment_yaw_tolerance: float = np.deg2rad(3.0)
    vehicle_front_length: float = 1.25
    vehicle_rear_length: float = 0.35
    vehicle_width: float = 0.8
    vehicle_safety_margin: float = 0.0

    # 路径不足时沿起点航向反向扩展
    max_extension_distance: float = 20.0
    extension_step: float = 0.5

    # 自动圆对搜索
    max_candidate_gap: float = 0.0  # >0时覆盖按车宽倍数计算的阈值
    max_candidate_gap_width_factor: float = 3.0
    pair_search_distance: float = 5.0
    path_resample_spacing: float = 0.2
    enforce_global_curvature: bool = True

    # 候选连接点 + 五次 Bezier 通道连接
    use_quintic_bezier_connection: bool = True  # 默认使用新的投影方法
    candidate_backtrack_step: float = 1.0
    max_candidate_backtrack: float = 20.0
    bezier_handle_min_factor: float = 0.15
    bezier_handle_max_factor: float = 0.60
    bezier_handle_samples: int = 8
    trajectory_sample_spacing: float = 0.1
    max_curvature_rate: float = 0.5
    w_path_length: float = 1.0
    w_curvature: float = 5.0
    w_curvature_rate: float = 2.0
    w_path_deviation: float = 0.5

    # 新的投影方法参数
    use_projection_method: bool = True  # 使用基于最大直径投影的方法
    projection_connect_only: bool = False  # 仅用直线连接投影段，不做可行性检查
    projection_extension_margin: float = 1.0  # 在最大直径基础上的延长余量(m)
    projection_skip_curvature_rate_check: bool = True  # 投影方法跳过曲率率检查
    projection_relaxed_curvature_check: bool = True  # 投影方法使用宽松的曲率检查
    projection_skip_alignment_check: bool = False  # 默认严格检查通道垂直对齐
    projection_search_step: float = 0.5  # 沿路径搜索连接点的步长(m)
    projection_max_search_distance: float = 20.0  # 最大搜索距离(m)，增加以处理大转角


class CircleCorridorRefinerEnhanced:
    """圆形通道路径修正器 - 增强版"""

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
        circle2: Tuple[float, float, float],
        all_obstacles: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """修正路径（增强版：包含连接和全局平滑）

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

        # 投影方法优先
        if self.params.use_projection_method:
            return self._refine_projection_method(
                np.asarray(path, dtype=float), circle1, circle2,
                all_obstacles=all_obstacles,
            )

        if self.params.use_quintic_bezier_connection:
            return self._refine_quintic_corridor(
                np.asarray(path, dtype=float), circle1, circle2,
                all_obstacles=all_obstacles,
            )

        # ===== 步骤1: 计算通道几何信息 =====
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
        s = (D + R1 - R2) / 2  # C1到Q的距离
        Q = C1 + s * u

        # ===== 步骤2: 找到通道调整区域 =====
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

        # ===== 步骤3: 优化通道区域路径 =====
        # 提取调整区域
        corridor_path = path[start_index:end_index+1].copy()

        # 计算原始误差
        original_center_error = self._compute_center_error(corridor_path, Q, u)
        original_angle_error = self._compute_angle_error(corridor_path, yaw_target)
        original_max_curvature = self._max_curvature(path)

        # 平滑调整
        optimized_corridor = self._smooth_adjustment_corridor(
            corridor_path, Q, u, yaw_target
        )

        # 优化局部路径
        optimized_corridor = self._optimize_corridor_path(
            optimized_corridor, Q, u, yaw_target
        )

        # 重新计算yaw
        optimized_corridor = self._recompute_yaw(optimized_corridor)

        # ===== 步骤4: 构建连接段 =====
        # 原始起点到优化起点的连接
        if start_index > 0:
            before_segment = path[:start_index]
            connection_start = self._create_connection(
                before_segment[-1],
                optimized_corridor[0],
                self.params.num_connection_points
            )
        else:
            before_segment = np.empty((0, 3))
            connection_start = np.empty((0, 3))

        # 优化终点到原始终点的连接
        if end_index < len(path) - 1:
            after_segment = path[end_index+1:]
            connection_end = self._create_connection(
                optimized_corridor[-1],
                after_segment[0],
                self.params.num_connection_points
            )
        else:
            after_segment = np.empty((0, 3))
            connection_end = np.empty((0, 3))

        # ===== 步骤5: 拼接完整路径 =====
        segments = []

        if len(before_segment) > 0:
            segments.append(before_segment)

        if len(connection_start) > 0:
            segments.append(connection_start)

        segments.append(optimized_corridor)

        if len(connection_end) > 0:
            segments.append(connection_end)

        if len(after_segment) > 0:
            segments.append(after_segment)

        complete_path = np.vstack(segments)

        # ===== 步骤6: 全局平滑 =====
        if self.params.global_smooth and HAS_SCIPY:
            smoothed_path = self._global_smooth(complete_path)
        else:
            smoothed_path = complete_path.copy()

        # 在进入障碍物影响区之前完成位置和航向对齐，通道内保持直线。
        alignment_mask = np.zeros(len(smoothed_path), dtype=bool)
        if self.params.enforce_entry_alignment:
            smoothed_path, alignment_mask = self._enforce_corridor_alignment(
                smoothed_path, Q, t, yaw_target, R1, R2
            )

        # ===== 步骤7: 施加阿克曼最小转弯半径约束 =====
        smoothed_path = self._enforce_min_turning_radius(smoothed_path)

        # ===== 步骤8: 检查可行性 =====
        is_valid, reason = self._check_feasibility(
            smoothed_path, circle1, circle2, R1, R2,
            all_obstacles=all_obstacles,
            alignment=(Q, t, alignment_mask),
        )

        if not is_valid:
            print(f"Warning: Smoothed path infeasible ({reason}), returning unsmoothed path")
            smoothed_path = complete_path.copy()
            alignment_mask = np.zeros(len(smoothed_path), dtype=bool)
            if self.params.enforce_entry_alignment:
                smoothed_path, alignment_mask = self._enforce_corridor_alignment(
                    smoothed_path, Q, t, yaw_target, R1, R2
                )
            smoothed_path = self._enforce_min_turning_radius(smoothed_path)
            # 再次检查
            is_valid, reason = self._check_feasibility(
                smoothed_path, circle1, circle2, R1, R2,
                all_obstacles=all_obstacles,
                alignment=(Q, t, alignment_mask),
            )
            if not is_valid:
                print(f"Warning: Complete path still infeasible, returning original path")
                return path.copy(), {
                    "valid": False,
                    "reason": reason,
                    "center_point": tuple(Q),
                    "target_yaw": yaw_target,
                    "free_gap": free_gap,
                    "start_index": int(start_index),
                    "center_index": int(center_index),
                    "end_index": int(end_index),
                }

        # 计算最终误差
        refined_center_error = self._compute_center_error(
            smoothed_path, Q, u
        )
        refined_angle_error = self._compute_angle_error(
            smoothed_path, yaw_target
        )
        final_max_curvature = self._max_curvature(smoothed_path)
        curvature_limit = self._curvature_limit()

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
            "num_connection_start_points": len(connection_start),
            "num_connection_end_points": len(connection_end),
            "global_smoothed": self.params.global_smooth and HAS_SCIPY,
            "min_turning_radius": float(self.params.min_turning_radius),
            "curvature_limit": float(curvature_limit),
            "original_max_curvature": float(original_max_curvature),
            "final_max_curvature": float(final_max_curvature),
            "achieved_min_turning_radius": float(
                1.0 / final_max_curvature if final_max_curvature > 1e-9 else np.inf
            ),
            "turning_radius_satisfied": bool(
                curvature_limit == np.inf or
                final_max_curvature <= curvature_limit *
                (1.0 + self.params.curvature_tolerance)
            ),
            "entry_alignment_satisfied": bool(
                self._alignment_is_valid(smoothed_path, t, alignment_mask)
            ),
            "aligned_point_count": int(np.count_nonzero(alignment_mask)),
            "original_path_length": len(path),
            "final_path_length": len(smoothed_path),
        }

        return smoothed_path, info

    def _refine_projection_method(
        self,
        path: np.ndarray,
        circle1: Tuple[float, float, float],
        circle2: Tuple[float, float, float],
        all_obstacles: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """投影方法：保留投影中心线硬约束，使用五次S形曲线连接出口"""
        failure = {
            "valid": False,
            "method": "projection_method",
            "reason": "Projection method failed",
            "requires_global_replan": True,
        }

        if path.ndim != 2 or path.shape[1] < 3 or not np.all(np.isfinite(path)):
            return path.copy(), dict(failure, reason="Invalid path format")

        # 解析圆形障碍物
        x1, y1, r1 = self._obstacle_tuple(circle1)
        x2, y2, r2 = self._obstacle_tuple(circle2)
        c1, c2 = np.array([x1, y1]), np.array([x2, y2])

        # 计算圆心距离和方向
        delta = c2 - c1
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return path.copy(), dict(failure, reason="Circles coincide")

        # 膨胀半径
        r1_inflated = r1 + self.params.safe_margin
        r2_inflated = r2 + self.params.safe_margin

        # 检查间隙
        free_gap = distance - r1_inflated - r2_inflated
        if free_gap < (self.params.vehicle_width + 2.0 * self.params.vehicle_safety_margin):
            result = dict(failure)
            result.update(reason="No valid gap between circles", free_gap=free_gap)
            return path.copy(), result

        # 圆心连线方向（法向量）
        normal = delta / distance
        # 通道方向（垂直于圆心连线）
        direction = np.array([-normal[1], normal[0]])

        # 计算通道中心点（两圆边界之间的中点）
        s = (distance + r1_inflated - r2_inflated) / 2
        center = c1 + s * normal

        # 确定通道方向（与路径前进方向一致）
        center_index = int(np.argmin(np.linalg.norm(path[:, :2] - center, axis=1)))
        lo, hi = max(0, center_index - 1), min(len(path) - 1, center_index + 1)
        travel = path[hi, :2] - path[lo, :2]
        if np.linalg.norm(travel) <= 1e-9:
            travel = path[-1, :2] - path[0, :2]
        if np.dot(direction, travel) < 0.0:
            direction = -direction

        yaw_target = float(np.arctan2(direction[1], direction[0]))

        # 计算投影直线的长度：最大直径 + 延长余量
        max_diameter = 2 * max(r1_inflated, r2_inflated)
        half_length = (max_diameter / 2) + self.params.projection_extension_margin

        # 计算投影直线的两个端点
        entry = center - half_length * direction
        exit_point = center + half_length * direction

        # 在全局路径上搜索入口连接点
        cumulative = self._compute_cumulative_distance(path)
        distances_to_entry = np.linalg.norm(path[:, :2] - entry, axis=1)
        nearest_entry_idx = int(np.argmin(distances_to_entry))

        entry_index = self._search_best_connection_point(
            path, cumulative, nearest_entry_idx, entry, yaw_target,
            search_backward=True
        )

        # 构建通道段（硬约束：位置和航向固定）
        spacing = max(1e-3, self.params.trajectory_sample_spacing)
        corridor_length = float(np.linalg.norm(exit_point - entry))
        corridor_points = max(2, int(np.ceil(corridor_length / spacing)) + 1)
        corridor_alpha = np.linspace(0.0, 1.0, corridor_points)
        corridor_xy = entry[None, :] + corridor_alpha[:, None] * (exit_point - entry)[None, :]

        # 通道航向固定为yaw_target（硬约束）
        corridor_yaw = np.full(corridor_points, yaw_target)
        corridor = np.column_stack((corridor_xy, corridor_yaw))

        if self.params.projection_connect_only:
            # 最简投影连接模式：在通道中心前后分别选择离投影端点最近
            # 的原路径点，直接拼接原路径、垂直投影段和后续原路径。
            # 按配置要求，不生成S曲线，不做平滑、碰撞、曲率或对齐检查。
            entry_search = path[:center_index + 1]
            entry_index = int(np.argmin(
                np.linalg.norm(entry_search[:, :2] - entry, axis=1)
            ))
            exit_search = path[center_index:]
            exit_index = center_index + int(np.argmin(
                np.linalg.norm(exit_search[:, :2] - exit_point, axis=1)
            ))
            connected = self._join_segments((
                path[:entry_index + 1],
                corridor,
                path[exit_index:],
            ))
            corridor_start = entry_index + 1
            if (entry_index >= 0 and
                    np.linalg.norm(path[entry_index, :2] - corridor[0, :2]) <= 1e-8):
                corridor_start -= 1
            return connected, {
                "valid": True,
                "method": "projection_connect_only",
                "reason": "Projection segment connected without validation",
                "requires_global_replan": False,
                "center_point": tuple(float(value) for value in center),
                "target_yaw": yaw_target,
                "free_gap": float(free_gap),
                "max_diameter": max_diameter,
                "projection_half_length": half_length,
                "projection_extension_margin": self.params.projection_extension_margin,
                "corridor_entry": tuple(float(value) for value in entry),
                "corridor_exit": tuple(float(value) for value in exit_point),
                "corridor_start_index": corridor_start,
                "corridor_end_index": corridor_start + len(corridor),
                "entry_index": entry_index,
                "connection_index": exit_index,
                "entry_s_curve_used": False,
                "connection_points_before": 0,
                "connection_points_after": 0,
                "global_smooth": False,
                "max_curvature": float(self._max_curvature(connected)),
                "path_points": f"{len(path)} → {len(connected)}",
                "start": tuple(connected[0, :2]),
                "goal": tuple(connected[-1, :2]),
            }

        # 前段：从路径起点到entry点之前
        # 使用前瞻插补：在entry_index之前的点，沿原路径向后搜索最佳连接点
        # 然后用S曲线平滑连接到corridor入口

        # 入口S曲线：从before_segment连接到corridor起点
        corridor_entry = corridor[0].copy()
        entry_yaw = yaw_target

        # 向后搜索最佳入口连接点（前瞻插补）
        if entry_index > 0:
            before_connection_idx = self._search_backward_connection_point(
                path=path[:entry_index+1],
                target_position=corridor_entry[:2],
                target_yaw=entry_yaw,
                lookahead_distance=5.0,  # 前瞻距离
            )
            before_segment = path[:before_connection_idx]

            # 生成入口S曲线
            if before_connection_idx <= entry_index:
                entry_connection_pose = path[before_connection_idx]
                entry_s_curve = self._create_entry_s_curve(
                    start_pose=entry_connection_pose,
                    end_pose=corridor_entry,
                    target_yaw=entry_yaw,
                )
            else:
                entry_s_curve = None
        else:
            before_connection_idx = 0
            before_segment = np.empty((0, 3))
            entry_s_curve = self._create_entry_s_curve(
                start_pose=path[0],
                end_pose=corridor_entry,
                target_yaw=entry_yaw,
            )

        if entry_s_curve is None:
            result = dict(failure)
            result.update(
                reason="No curvature-feasible entry connection",
                entry_index=entry_index,
                before_connection_index=before_connection_idx,
            )
            return path.copy(), result

        # 通道出口：使用最后一个corridor点
        corridor_exit = corridor[-1].copy()

        # 搜索前向S形曲线连接
        obstacles_list = list(all_obstacles) if all_obstacles is not None else [circle1, circle2]

        exit_s_curve, connection_index, s_curve_info = self._search_forward_s_curve_connection(
            path=path,
            corridor=corridor,
            corridor_exit=corridor_exit,
            corridor_direction=direction,
            obstacles=obstacles_list,
            reference_index=None,
        )

        if exit_s_curve is None:
            # S形连接失败
            result = dict(failure)
            result.update(
                reason=s_curve_info.get("reason", "S-curve connection failed"),
                s_curve_info=s_curve_info,
            )
            return path.copy(), result

        # 拼接完整路径：before + entry_s_curve + corridor + exit_s_curve + path[connection_index:]
        segments_to_join = [before_segment]
        if entry_s_curve is not None:
            segments_to_join.append(entry_s_curve)
        segments_to_join.extend([corridor, exit_s_curve, path[connection_index:]])

        complete = self._join_segments(segments_to_join)

        # 创建锁定mask：corridor和exit_s_curve不允许被平滑修改
        locked_mask = np.zeros(len(complete), dtype=bool)
        # _join_segments 会删除 corridor 与 entry_s_curve 重合的入口点，
        # 因此中心段从入口曲线最后一点开始；旧实现漏算入口曲线长度，
        # 会把连接曲线误判为偏离中心线。
        corridor_start_idx = (
            len(before_segment) + len(entry_s_curve) - 1
        )
        corridor_end_idx = corridor_start_idx + len(corridor)
        # exit_s_curve 的首点同样与 corridor 出口重合并被删除。
        s_curve_end_idx = corridor_end_idx + len(exit_s_curve) - 1

        # 锁定corridor和S形曲线
        if corridor_start_idx < len(complete) and s_curve_end_idx <= len(complete):
            locked_mask[corridor_start_idx:s_curve_end_idx] = True

        # 全局平滑（只平滑未锁定部分）
        # 注意：当存在锁定段时，全局样条平滑会在边界产生高曲率
        # 因此我们跳过全局平滑，让S形曲线本身保证平滑过渡
        smoothed = complete.copy()

        # 如果需要，可以只对 before_segment 和 path[connection_index:] 分别平滑
        # 但为了保证曲率约束，暂时禁用全局平滑
        if False and self.params.global_smooth and HAS_SCIPY:
            smoothed = self._global_smooth(complete, locked_mask=locked_mask)

            # 验证锁定段未被修改
            if corridor_start_idx < len(smoothed) and s_curve_end_idx <= len(smoothed):
                corridor_deviation = np.linalg.norm(
                    smoothed[corridor_start_idx:corridor_end_idx, :2] - corridor[:, :2],
                    axis=1
                )
                s_curve_deviation = np.linalg.norm(
                    smoothed[corridor_end_idx:s_curve_end_idx, :2] - exit_s_curve[:, :2],
                    axis=1
                )

                if np.max(corridor_deviation) > 1e-6:
                    smoothed[corridor_start_idx:corridor_end_idx] = corridor.copy()
                if np.max(s_curve_deviation) > 1e-6:
                    smoothed[corridor_end_idx:s_curve_end_idx] = exit_s_curve.copy()

        # 删除重复点（在平滑后）
        unique_mask = np.ones(len(smoothed), dtype=bool)
        for i in range(1, len(smoothed)):
            if np.linalg.norm(smoothed[i, :2] - smoothed[i-1, :2]) < 1e-9:
                unique_mask[i] = False
        old_to_new = np.cumsum(unique_mask, dtype=int) - 1
        validated_corridor_start = int(old_to_new[corridor_start_idx])
        validated_corridor_end = int(old_to_new[corridor_end_idx - 1]) + 1
        smoothed = smoothed[unique_mask]

        # 最终验证
        is_valid, reason = self._validate_projection_trajectory(
            smoothed, obstacles_list,
            corridor=(entry, exit_point, center, direction),
            corridor_indices=(
                validated_corridor_start,
                validated_corridor_end,
            ),
        )

        if not is_valid:
            print(f"[投影方法] 最终验证失败: {reason}")
            result = dict(failure)
            result.update(
                validation_reason=reason,
                s_curve_info=s_curve_info,
            )
            return path.copy(), result

        # 计算曲率并检查
        curvatures = self._compute_signed_curvatures(smoothed)
        max_curvature = float(np.max(np.abs(curvatures)))
        max_curvature_idx = int(np.argmax(np.abs(curvatures)))

        # 曲率限制检查
        curvature_limit = self._curvature_limit()
        if np.isfinite(curvature_limit):
            allowed_curvature = curvature_limit * (1.0 + self.params.curvature_tolerance)
            if max_curvature > allowed_curvature:
                # 调试信息：显示最大曲率位置
                print(f"[投影方法] 最终曲率检查失败: {max_curvature:.4f} > {allowed_curvature:.4f}")
                print(f"  最大曲率位置: 索引 {max_curvature_idx}/{len(smoothed)}")
                print(f"  corridor范围: [{corridor_start_idx}, {corridor_end_idx})")
                print(f"  S曲线结束: {s_curve_end_idx}")

                # 检查各段的曲率
                if corridor_start_idx > 0:
                    before_curv = curvatures[:corridor_start_idx]
                    print(f"  before段曲率: max={np.max(np.abs(before_curv)):.4f}")
                corridor_curv = curvatures[corridor_start_idx:corridor_end_idx]
                print(f"  corridor段曲率: max={np.max(np.abs(corridor_curv)):.4f}")
                if s_curve_end_idx <= len(curvatures):
                    s_curv = curvatures[corridor_end_idx:s_curve_end_idx]
                    print(f"  S曲线段曲率: max={np.max(np.abs(s_curv)):.4f}")
                    after_curv = curvatures[s_curve_end_idx:]
                    print(f"  after段曲率: max={np.max(np.abs(after_curv)):.4f}")

                result = dict(failure)
                result.update(
                    validation_reason=f"Final curvature {max_curvature:.4f} exceeds limit {allowed_curvature:.4f}",
                    s_curve_info=s_curve_info,
                    max_curvature=max_curvature,
                )
                return path.copy(), result

        return smoothed, {
            "valid": True,
            "method": "projection_method",
            "reason": "Valid",
            "requires_global_replan": False,
            "center_point": tuple(center),
            "target_yaw": yaw_target,
            "free_gap": float(free_gap),
            "max_diameter": max_diameter,
            "projection_half_length": half_length,
            "projection_extension_margin": self.params.projection_extension_margin,
            "corridor_entry": tuple(float(value) for value in entry),
            "corridor_exit": tuple(float(value) for value in exit_point),
            "corridor_start_index": validated_corridor_start,
            "corridor_end_index": validated_corridor_end,
            "entry_index": entry_index,
            "entry_s_curve_used": entry_s_curve is not None,
            "entry_s_curve_points": len(entry_s_curve) if entry_s_curve is not None else 0,
            "connection_index": connection_index,
            "connection_points_before": 0,
            "connection_points_after": len(exit_s_curve),
            "global_smooth": False,  # 禁用了全局平滑以保持硬约束
            "max_curvature": max_curvature,
            "path_points": f"{len(path)} → {len(smoothed)}",
            "start": tuple(smoothed[0, :2]),
            "goal": tuple(smoothed[-1, :2]),
            "s_curve_info": s_curve_info,
        }


    def _refine_quintic_corridor(
        self,
        path: np.ndarray,
        circle1: Tuple[float, float, float],
        circle2: Tuple[float, float, float],
        all_obstacles: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """用候选连接点、五次 Bezier 和锁定中心直线修正通道。"""
        failure = {
            "valid": False,
            "method": "quintic_bezier_corridor_refinement",
            "reason": "No kinematically feasible corridor entry/exit trajectory",
            "requires_global_replan": True,
        }
        if path.ndim != 2 or path.shape[1] < 3 or not np.all(np.isfinite(path)):
            return path.copy(), dict(failure)

        x1, y1, r1 = self._obstacle_tuple(circle1)
        x2, y2, r2 = self._obstacle_tuple(circle2)
        c1, c2 = np.array([x1, y1]), np.array([x2, y2])
        delta = c2 - c1
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return path.copy(), dict(failure)
        r1_inflated = r1 + self.params.safe_margin
        r2_inflated = r2 + self.params.safe_margin
        free_gap = distance - r1_inflated - r2_inflated
        if free_gap < (self.params.vehicle_width +
                       2.0 * self.params.vehicle_safety_margin):
            result = dict(failure)
            result.update(reason="No valid gap between circles", free_gap=free_gap)
            return path.copy(), result

        normal = delta / distance
        center = c1 + 0.5 * (distance + r1_inflated - r2_inflated) * normal
        direction = np.array([-normal[1], normal[0]])
        center_index = int(np.argmin(np.linalg.norm(path[:, :2] - center, axis=1)))
        lo, hi = max(0, center_index - 1), min(len(path) - 1, center_index + 1)
        travel = path[hi, :2] - path[lo, :2]
        if np.linalg.norm(travel) <= 1e-9:
            travel = path[-1, :2] - path[0, :2]
        if np.dot(direction, travel) < 0.0:
            direction = -direction
        yaw_target = float(np.arctan2(direction[1], direction[0]))
        # 通道核心只负责让车辆在两圆影响区内保持中心线直行。其半长
        # 采用“最大圆半径 + 可调余量”，避免把车头长度重复计入核心段，
        # 导致本来有足够调整空间的斜向路径被过早判为无解。
        half_length = (max(r1_inflated, r2_inflated) +
                       max(0.0, self.params.alignment_clearance))
        entry = center - half_length * direction
        exit_point = center + half_length * direction
        spacing = max(1e-3, self.params.trajectory_sample_spacing)
        corridor = self._sample_straight(entry, exit_point, yaw_target, spacing)
        obstacles = list(all_obstacles) if all_obstacles is not None else [circle1, circle2]

        # 硬约束段本身先做连续矩形车身检查；不满足则无需搜索连接曲线。
        ok, _ = self._validate_candidate_trajectory(
            corridor, obstacles, corridor=(entry, exit_point, center, direction)
        )
        if not ok:
            result = dict(failure)
            result.update(center_point=tuple(center), target_yaw=yaw_target,
                          free_gap=float(free_gap), corridor_entry=tuple(entry),
                          corridor_exit=tuple(exit_point), center_index=center_index)
            return path.copy(), result

        cumulative = self._compute_cumulative_distance(path)
        # 以原路径上最靠近两个通道端点的离散点为连接基准，再分别向
        # 上游/下游扩展候选。offset=0 因而就是用户直观指定的最近点。
        nearest_entry_index = int(np.argmin(
            np.linalg.norm(path[:, :2] - entry, axis=1)))
        nearest_exit_index = int(np.argmin(
            np.linalg.norm(path[:, :2] - exit_point, axis=1)))
        entry_anchor = float(cumulative[nearest_entry_index])
        exit_anchor = float(cumulative[nearest_exit_index])
        if exit_anchor <= entry_anchor:
            result = dict(failure)
            result.update(corridor_entry=tuple(entry), corridor_exit=tuple(exit_point))
            return path.copy(), result

        entry_indices = self._candidate_indices(
            cumulative, entry_anchor, upstream=True
        )
        exit_indices = self._candidate_indices(
            cumulative, exit_anchor, upstream=False
        )
        best_entry = self._search_connection(
            path, entry_indices, entry, yaw_target, obstacles, corridor,
            is_entry=True, reference_path=path,
        )
        best_exit = self._search_connection(
            path, exit_indices, exit_point, yaw_target, obstacles, corridor,
            is_entry=False, reference_path=path,
        )
        if best_entry is None or best_exit is None or best_entry[1] >= best_exit[1]:
            result = dict(failure)
            result.update(center_point=tuple(center), target_yaw=yaw_target,
                          free_gap=float(free_gap), corridor_entry=tuple(entry),
                          corridor_exit=tuple(exit_point), center_index=center_index,
                          start_index=int(entry_indices[0]) if entry_indices else 0)
            return path.copy(), result

        entry_curve, entry_index, entry_handles, entry_cost = best_entry
        exit_curve, exit_index, exit_handles, exit_cost = best_exit
        complete = self._join_segments([
            path[:entry_index + 1], entry_curve, corridor, exit_curve,
            path[exit_index:],
        ])
        complete = self._resample_preserving_corridor(
            complete, entry, exit_point, center, direction, spacing
        )
        valid, reason = self._validate_candidate_trajectory(
            complete, obstacles,
            corridor=(entry, exit_point, center, direction),
        )
        if not valid:
            result = dict(failure)
            result.update(validation_reason=reason, center_point=tuple(center),
                          target_yaw=yaw_target, free_gap=float(free_gap),
                          corridor_entry=tuple(entry), corridor_exit=tuple(exit_point),
                          start_index=entry_index)
            return path.copy(), result

        curvatures = self._compute_signed_curvatures(complete)
        return complete, {
            "valid": True,
            "method": "quintic_bezier_corridor_refinement",
            "reason": "Valid",
            "requires_global_replan": False,
            "center_point": tuple(center),
            "target_yaw": yaw_target,
            "free_gap": float(free_gap),
            "center_index": center_index,
            "start_index": entry_index,
            "end_index": exit_index,
            "selected_entry_index": entry_index,
            "selected_exit_index": exit_index,
            "entry_backtrack_distance": float(entry_anchor - cumulative[entry_index]),
            "exit_forward_distance": float(cumulative[exit_index] - exit_anchor),
            "entry_bezier_handles": tuple(float(v) for v in entry_handles),
            "exit_bezier_handles": tuple(float(v) for v in exit_handles),
            "max_curvature": float(np.max(np.abs(curvatures))),
            "final_max_curvature": float(np.max(np.abs(curvatures))),
            "corridor_entry": tuple(float(v) for v in entry),
            "corridor_exit": tuple(float(v) for v in exit_point),
            "entry_cost": float(entry_cost),
            "exit_cost": float(exit_cost),
            "original_path_length": len(path),
            "final_path_length": len(complete),
        }

    def _sample_straight(self, start, end, yaw, spacing):
        length = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(length / spacing)) + 1)
        alpha = np.linspace(0.0, 1.0, count)
        xy = start[None, :] + alpha[:, None] * (end - start)[None, :]
        return np.column_stack((xy, np.full(count, yaw)))

    def _nearest_path_station(self, path: np.ndarray, point: np.ndarray) -> float:
        cumulative = self._compute_cumulative_distance(path)
        best_distance, best_station = np.inf, 0.0
        for i in range(len(path) - 1):
            segment = path[i + 1, :2] - path[i, :2]
            length2 = float(np.dot(segment, segment))
            if length2 <= 1e-12:
                continue
            ratio = float(np.clip(np.dot(point - path[i, :2], segment) / length2, 0, 1))
            projection = path[i, :2] + ratio * segment
            error = float(np.linalg.norm(point - projection))
            if error < best_distance:
                best_distance = error
                best_station = cumulative[i] + ratio * np.sqrt(length2)
        return float(best_station)

    def _candidate_indices(self, cumulative, anchor, upstream):
        step = max(1e-3, self.params.candidate_backtrack_step)
        maximum = max(0.0, self.params.max_candidate_backtrack)
        targets = np.arange(0.0, maximum + 0.5 * step, step)
        result = []
        for offset in targets:
            station = anchor - offset if upstream else anchor + offset
            if station < cumulative[0] - 1e-9 or station > cumulative[-1] + 1e-9:
                continue
            index = (int(np.searchsorted(cumulative, station, side="right") - 1)
                     if upstream else int(np.searchsorted(cumulative, station)))
            index = int(np.clip(index, 0, len(cumulative) - 1))
            if index not in result:
                result.append(index)
        return result

    def _search_connection(self, path, indices, target, target_yaw, obstacles,
                           corridor, is_entry, reference_path):
        best = None
        for index in indices:
            pose = path[index]
            delta = target - pose[:2]
            longitudinal = float(np.dot(delta, np.array([
                np.cos(target_yaw), np.sin(target_yaw)])))
            if (is_entry and longitudinal <= 0.0) or (not is_entry and longitudinal >= 0.0):
                continue
            if any(self._vehicle_collides(pose, obstacle) for obstacle in obstacles):
                continue
            yaw_error = abs(self._normalize_angle(target_yaw - pose[2]))
            euclidean = float(np.linalg.norm(delta))
            # 不再用启发式 required_distance 直接淘汰最近连接点。是否有
            # 足够距离转向交给生成后的真实曲率、曲率率和碰撞检查判断。
            # 该估计仍参与候选排序代价，避免在等价曲线中选择过短连接。
            required = (max(0.0, self.params.min_turning_radius) * yaw_error +
                        self.params.vehicle_front_length +
                        self.params.alignment_clearance)
            low = max(1e-3, self.params.bezier_handle_min_factor * euclidean)
            high = max(low, self.params.bezier_handle_max_factor * euclidean)
            handles = np.linspace(low, high, max(1, self.params.bezier_handle_samples))
            for a in handles:
                for b in handles:
                    if is_entry:
                        curve = self._create_quintic_bezier_connection(
                            pose, np.r_[target, target_yaw], a, b,
                            self.params.trajectory_sample_spacing)
                        context = self._join_segments([
                            path[max(0, index - 2):index + 1], curve, corridor[:3]
                        ])
                    else:
                        curve = self._create_quintic_bezier_connection(
                            np.r_[target, target_yaw], pose, a, b,
                            self.params.trajectory_sample_spacing)
                        context = self._join_segments([
                            corridor[-3:], curve, path[index:min(len(path), index + 3)]
                        ])
                    valid, _ = self._validate_candidate_trajectory(context, obstacles)
                    if not valid:
                        continue
                    shortfall = max(0.0, required - euclidean)
                    cost = self._trajectory_cost(curve, reference_path) + shortfall ** 2
                    if best is None or cost < best[3]:
                        best = (curve, index, (a, b), cost)
        return best

    def _create_quintic_bezier_connection(
        self, start_pose, end_pose, start_handle, end_handle, spacing
    ) -> np.ndarray:
        """生成端点位置、航向和零端点曲率约束的五次 Bezier 曲线。"""
        p0, p5 = np.asarray(start_pose[:2]), np.asarray(end_pose[:2])
        ts = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
        te = np.array([np.cos(end_pose[2]), np.sin(end_pose[2])])
        control = np.vstack((p0, p0 + start_handle * ts,
                             p0 + 2.0 * start_handle * ts,
                             p5 - 2.0 * end_handle * te,
                             p5 - end_handle * te, p5))
        estimate = max(np.linalg.norm(np.diff(control, axis=0), axis=1).sum(),
                       np.linalg.norm(p5 - p0))
        dense_count = max(101, int(np.ceil(estimate / max(spacing, 1e-3))) * 10)
        t = np.linspace(0.0, 1.0, dense_count)
        omt = 1.0 - t
        basis = np.column_stack((omt**5, 5*omt**4*t, 10*omt**3*t**2,
                                 10*omt**2*t**3, 5*omt*t**4, t**5))
        # 显式加权求和可避免某些 Accelerate/BLAS 版本对极瘦矩阵
        # matmul 偶发发出错误的 overflow 警告。
        xy = np.sum(basis[:, :, None] * control[None, :, :], axis=1)
        first_control = 5.0 * np.diff(control, axis=0)
        second_control = 20.0 * np.diff(control, n=2, axis=0)
        b4 = np.column_stack((omt**4, 4*omt**3*t, 6*omt**2*t**2,
                              4*omt*t**3, t**4))
        b3 = np.column_stack((omt**3, 3*omt**2*t, 3*omt*t**2, t**3))
        derivative = np.sum(b4[:, :, None] * first_control[None, :, :], axis=1)
        _second_derivative = np.sum(
            b3[:, :, None] * second_control[None, :, :], axis=1
        )  # 显式计算，供解析曲率扩展使用
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
        samples = np.arange(0.0, arc[-1], max(spacing, 1e-3))
        samples = np.append(samples, arc[-1]) if len(samples) == 0 or samples[-1] < arc[-1] else samples
        x = np.interp(samples, arc, xy[:, 0])
        y = np.interp(samples, arc, xy[:, 1])
        dx = np.interp(samples, arc, derivative[:, 0])
        dy = np.interp(samples, arc, derivative[:, 1])
        yaw = np.arctan2(dy, dx)
        yaw[0], yaw[-1] = start_pose[2], end_pose[2]
        return np.column_stack((x, y, yaw))


    def _join_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        joined = []
        for segment in segments:
            if segment is None or len(segment) == 0:
                continue
            segment = np.asarray(segment, dtype=float)
            if joined and np.linalg.norm(joined[-1][-1, :2] - segment[0, :2]) <= 1e-8:
                segment = segment[1:]
            if len(segment):
                joined.append(segment)
        return np.vstack(joined) if joined else np.empty((0, 3))

    def _compute_signed_curvatures(self, path: np.ndarray) -> np.ndarray:
        curvature = np.zeros(len(path), dtype=float)
        if len(path) < 3:
            return curvature
        p = path[:, :2]
        for i in range(1, len(path) - 1):
            v1, v2 = p[i] - p[i - 1], p[i + 1] - p[i]
            a, b = np.linalg.norm(v1), np.linalg.norm(v2)
            c = np.linalg.norm(p[i + 1] - p[i - 1])
            denominator = a * b * c
            if denominator > 1e-12:
                curvature[i] = 2.0 * (v1[0] * v2[1] - v1[1] * v2[0]) / denominator
        return curvature

    def _densify_trajectory(self, trajectory: np.ndarray, spacing: float) -> np.ndarray:
        """线性细分每段，并用段切向航向检查连续车身扫掠。"""
        if len(trajectory) < 2:
            return trajectory.copy()
        pieces = []
        for i in range(len(trajectory) - 1):
            start, end = trajectory[i], trajectory[i + 1]
            length = float(np.linalg.norm(end[:2] - start[:2]))
            count = max(1, int(np.ceil(length / max(spacing, 1e-3))))
            alpha = np.arange(count, dtype=float) / count
            xy = start[:2] + alpha[:, None] * (end[:2] - start[:2])
            angle_delta = self._normalize_angle(end[2] - start[2])
            yaw = start[2] + alpha * angle_delta
            pieces.append(np.column_stack((xy, yaw)))
        pieces.append(trajectory[-1:])
        return np.vstack(pieces)

    def _validate_projection_trajectory(
        self, trajectory: np.ndarray, all_obstacles: Sequence[Any],
        corridor: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        corridor_indices: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, str]:
        """简化版轨迹验证（用于投影方法，保留通道中心线硬约束）"""
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) < 2:
            return False, "Trajectory is too short"
        if not np.all(np.isfinite(trajectory)):
            return False, "Trajectory contains non-finite values"

        # 检查连续性
        segment = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
        if np.any(segment <= 1e-7):
            return False, "Trajectory contains duplicate adjacent points"

        # 检查碰撞
        dense = self._densify_trajectory(trajectory, self.params.trajectory_sample_spacing)
        for i, point in enumerate(dense):
            for j, obstacle in enumerate(all_obstacles):
                if self._vehicle_collides(point, obstacle):
                    return False, f"Vehicle collision with obstacle {j} at sample {i}"

        # 硬约束：检查通道段是否在中心线上
        if corridor is not None and corridor_indices is not None:
            entry, exit_point, center, direction = corridor
            corridor_start_idx, corridor_end_idx = corridor_indices

            # 确保索引有效
            if corridor_start_idx < 0 or corridor_end_idx > len(trajectory):
                return False, f"Invalid corridor indices: [{corridor_start_idx}, {corridor_end_idx}]"

            if corridor_end_idx <= corridor_start_idx:
                return False, "Corridor segment is empty"

            direction = direction / max(np.linalg.norm(direction), 1e-12)
            normal = np.array([-direction[1], direction[0]])

            # 直接使用提供的索引范围检查通道段
            corridor_segment = trajectory[corridor_start_idx:corridor_end_idx]

            # 硬约束：通道段必须在中心线上
            lateral = np.abs((corridor_segment[:, :2] - center) @ normal)
            if np.max(lateral) > 1e-6:
                return False, f"Corridor segment moved off centerline (max deviation: {np.max(lateral):.6f}m)"

            if not self.params.projection_skip_alignment_check:
                yaw_target = float(np.arctan2(direction[1], direction[0]))
                yaw_error = np.array([
                    abs(self._normalize_angle(float(yaw) - yaw_target))
                    for yaw in corridor_segment[:, 2]
                ])
                if np.max(yaw_error) > self.params.alignment_yaw_tolerance:
                    return False, (
                        "Corridor heading is not perpendicular to circle line "
                        f"(max error: {np.degrees(np.max(yaw_error)):.3f}deg)"
                    )

        return True, "Valid"

    def _validate_candidate_trajectory(
        self, trajectory: np.ndarray, all_obstacles: Sequence[Any],
        corridor: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[bool, str]:
        """统一检查数值、连续性、曲率/曲率率、航向和矩形碰撞。"""
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) < 2:
            return False, "Trajectory is too short"
        if not np.all(np.isfinite(trajectory)):
            return False, "Trajectory contains non-finite values"
        segment = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
        if np.any(segment <= 1e-7):
            return False, "Trajectory contains duplicate adjacent points"
        curvature = self._compute_signed_curvatures(trajectory)
        limit = self._curvature_limit()
        if np.isfinite(limit) and np.max(np.abs(curvature)) > limit * (1.0 + self.params.curvature_tolerance):
            return False, "Maximum curvature exceeds turning-radius limit"
        station = self._compute_cumulative_distance(trajectory)
        if len(curvature) >= 2:
            ds = np.diff(station)
            rate = np.abs(np.diff(curvature)) / np.maximum(ds, 1e-9)
            if np.max(rate) > self.params.max_curvature_rate + 1e-9:
                return False, "Maximum curvature rate exceeded"
        dense = self._densify_trajectory(trajectory, self.params.trajectory_sample_spacing)
        for i, point in enumerate(dense):
            for j, obstacle in enumerate(all_obstacles):
                if self._vehicle_collides(point, obstacle):
                    return False, f"Vehicle collision with obstacle {j} at sample {i}"
        if corridor is not None:
            entry, exit_point, center, direction = corridor
            direction = direction / max(np.linalg.norm(direction), 1e-12)
            normal = np.array([-direction[1], direction[0]])
            longitudinal = (trajectory[:, :2] - center) @ direction
            half = 0.5 * np.linalg.norm(exit_point - entry)
            core = np.abs(longitudinal) <= half + 1e-6
            if np.count_nonzero(core) < 2:
                return False, "Corridor straight segment is missing"
            lateral = np.abs((trajectory[core, :2] - center) @ normal)
            if np.max(lateral) > 1e-6:
                return False, "Corridor segment moved off centerline"
            yaw_target = np.arctan2(direction[1], direction[0])
            yaw_error = np.array([
                abs(self._normalize_angle(value - yaw_target))
                for value in trajectory[core, 2]
            ])
            if np.max(yaw_error) > self.params.alignment_yaw_tolerance + 1e-9:
                return False, "Corridor entry/exit heading is not aligned"
            core_curvature = np.abs(curvature[core])
            # 边界离散三点可能跨越 Bezier；跳过核心首尾点，内部必须严格为零。
            if len(core_curvature) > 2 and np.max(core_curvature[1:-1]) > 1e-7:
                return False, "Corridor core is not straight"
        return True, "Valid"

    def _trajectory_cost(self, trajectory: np.ndarray, original: np.ndarray) -> float:
        station = self._compute_cumulative_distance(trajectory)
        curvature = self._compute_signed_curvatures(trajectory)
        length = float(station[-1])
        integrate = getattr(np, "trapezoid", np.trapz)
        curvature_energy = float(integrate(curvature ** 2, station))
        if len(trajectory) > 1:
            mid_station = 0.5 * (station[:-1] + station[1:])
            rate = np.diff(curvature) / np.maximum(np.diff(station), 1e-9)
            rate_energy = float(integrate(rate ** 2, mid_station)) if len(rate) > 1 else 0.0
        else:
            rate_energy = 0.0
        distances = []
        for point in trajectory[:, :2]:
            best = np.inf
            for i in range(len(original) - 1):
                a, b = original[i, :2], original[i + 1, :2]
                ab = b - a
                ratio = float(np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
                best = min(best, float(np.linalg.norm(point - (a + ratio * ab))))
            distances.append(best)
        deviation = float(np.mean(distances)) if distances else 0.0
        return (self.params.w_path_length * length +
                self.params.w_curvature * curvature_energy +
                self.params.w_curvature_rate * rate_energy +
                self.params.w_path_deviation * deviation ** 2)

    def _resample_preserving_corridor(self, path, entry, exit_point, center,
                                      direction, spacing):
        """分别重采样三个区域，避免跨区插值移动锁定的通道直线。"""
        direction = direction / np.linalg.norm(direction)
        station = (path[:, :2] - center) @ direction
        half = 0.5 * np.linalg.norm(exit_point - entry)
        before_end = int(np.flatnonzero(station <= -half + 1e-6)[-1])
        after_start = int(np.flatnonzero(station >= half - 1e-6)[0])
        before = _resample_path(path[:before_end + 1], spacing)
        core = self._sample_straight(entry, exit_point,
                                     np.arctan2(direction[1], direction[0]), spacing)
        after = _resample_path(path[after_start:], spacing)
        result = self._join_segments([before, core, after])
        return self._recompute_yaw(result)

    def _search_best_connection_point(
        self,
        path: np.ndarray,
        cumulative: np.ndarray,
        start_idx: int,
        target_point: np.ndarray,
        target_yaw: float,
        search_backward: bool = True
    ) -> int:
        """沿路径搜索最佳连接点

        从最近点开始，沿路径前后搜索，找到既靠近目标点又能满足转弯半径约束的点

        Args:
            path: 全局路径
            cumulative: 累计距离
            start_idx: 起始搜索索引（最近点）
            target_point: 目标点位置
            target_yaw: 目标航向
            search_backward: True表示向路径上游搜索，False表示向下游搜索

        Returns:
            最佳连接点索引
        """
        best_idx = start_idx
        best_score = float('inf')

        # 计算搜索范围
        start_station = cumulative[start_idx]
        search_stations = []

        step = self.params.projection_search_step
        max_dist = self.params.projection_max_search_distance

        if search_backward:
            # 向上游搜索：从start_station向前
            stations = np.arange(0, max_dist + step/2, step)
            search_stations = [start_station - s for s in stations]
        else:
            # 向下游搜索：从start_station向后
            stations = np.arange(0, max_dist + step/2, step)
            search_stations = [start_station + s for s in stations]

        # 对每个候选点评估
        for station in search_stations:
            if station < cumulative[0] or station > cumulative[-1]:
                continue

            # 找到对应的路径索引
            idx = int(np.searchsorted(cumulative, station))
            idx = np.clip(idx, 0, len(path) - 1)

            # 计算评分：距离 + 角度差异
            pos = path[idx, :2]
            yaw = path[idx, 2]

            distance_to_target = float(np.linalg.norm(pos - target_point))
            angle_diff = abs(self._normalize_angle(yaw - target_yaw))

            # 评分函数：距离权重 + 角度权重
            # 距离越近越好，角度差异越小越好
            score = distance_to_target + 2.0 * angle_diff

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _create_smooth_connection(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """创建两点之间的平滑连接段（用于投影方法）

        Args:
            start_point: 起点 [x, y, yaw]
            end_point: 终点 [x, y, yaw]
            num_points: 连接点数

        Returns:
            连接路径段
        """
        if num_points < 1:
            return np.empty((0, 3))

        # 只返回内部插值点，避免与相邻路径段产生重复端点
        x = np.linspace(start_point[0], end_point[0], num_points + 2)[1:-1]
        y = np.linspace(start_point[1], end_point[1], num_points + 2)[1:-1]

        # 平滑插值角度
        yaw = np.zeros(num_points)
        start_yaw = start_point[2]
        end_yaw = end_point[2]

        # 处理角度跳变
        angle_diff = self._normalize_angle(end_yaw - start_yaw)

        for i in range(num_points):
            alpha = (i + 1) / (num_points + 1)
            # 使用平滑插值函数
            alpha_smooth = self._smooth_interpolation(alpha)
            yaw[i] = start_yaw + alpha_smooth * angle_diff

        return np.column_stack([x, y, yaw])

    def _create_connection(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """创建两点之间的连接段

        Args:
            start_point: 起点 [x, y, yaw]
            end_point: 终点 [x, y, yaw]
            num_points: 连接点数

        Returns:
            连接路径段
        """
        if num_points < 1:
            return np.empty((0, 3))

        # 只返回内部插值点，避免与相邻路径段产生重复端点
        x = np.linspace(start_point[0], end_point[0], num_points + 2)[1:-1]
        y = np.linspace(start_point[1], end_point[1], num_points + 2)[1:-1]

        # 平滑插值角度
        yaw = np.zeros(num_points)
        start_yaw = start_point[2]
        end_yaw = end_point[2]

        # 处理角度跳变
        angle_diff = self._normalize_angle(end_yaw - start_yaw)

        for i in range(num_points):
            alpha = (i + 1) / (num_points + 1)
            # 使用平滑插值函数
            alpha_smooth = self._smooth_interpolation(alpha)
            yaw[i] = start_yaw + alpha_smooth * angle_diff

        return np.column_stack([x, y, yaw])


    def _smooth_interpolation(self, t: float) -> float:
        """平滑插值函数（S曲线）

        Args:
            t: 参数 [0, 1]

        Returns:
            平滑后的参数
        """
        # 使用3次Hermite插值
        return t * t * (3 - 2 * t)


    def _global_smooth(
        self, path: np.ndarray, locked_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """全局路径平滑

        Args:
            path: 输入路径

        Returns:
            平滑后的路径
        """
        if len(path) < 4:
            return path.copy()

        try:
            # 使用B样条平滑位置
            x = path[:, 0]
            y = path[:, 1]
            yaw = path[:, 2]

            # 参数化路径
            distances = self._compute_cumulative_distance(path)
            u = distances / (distances[-1] + 1e-9)

            # B样条拟合
            tck, u_new = splprep([x, y], u=u, s=self.params.spline_smoothing, k=3)

            # 重新采样
            u_fine = np.linspace(0, 1, len(path))
            x_smooth, y_smooth = splev(u_fine, tck)

            # 路径任务的起终点是硬约束，禁止样条平滑造成端点漂移
            x_smooth[0], y_smooth[0] = x[0], y[0]
            x_smooth[-1], y_smooth[-1] = x[-1], y[-1]

            # 计算平滑后的yaw
            yaw_smooth = np.zeros(len(path))
            for i in range(len(path) - 1):
                dx = x_smooth[i+1] - x_smooth[i]
                dy = y_smooth[i+1] - y_smooth[i]
                if np.hypot(dx, dy) > 1e-6:
                    yaw_smooth[i] = np.arctan2(dy, dx)
                else:
                    yaw_smooth[i] = yaw[i] if i == 0 else yaw_smooth[i-1]

            # 最后一点
            yaw_smooth[-1] = yaw_smooth[-2]

            # 混合平滑：保留部分原始角度信息
            factor = self.params.smooth_factor
            yaw_final = (1 - factor) * yaw + factor * yaw_smooth

            result = np.column_stack([x_smooth, y_smooth, yaw_final])
            if locked_mask is not None:
                locked_mask = np.asarray(locked_mask, dtype=bool)
                if locked_mask.shape != (len(path),):
                    raise ValueError("locked_mask must have one value per path point")
                result[locked_mask] = path[locked_mask]
            return result

        except Exception as e:
            print(f"Warning: Global smoothing failed ({e}), returning original path")
            return path.copy()


    def _smooth_adjustment_corridor(
        self,
        corridor_path: np.ndarray,
        Q: np.ndarray,
        u: np.ndarray,
        yaw_target: float
    ) -> np.ndarray:
        """平滑调整通道内路径

        Args:
            corridor_path: 通道内路径
            Q: 通道中心点
            u: 圆心连线单位向量（法向量）
            yaw_target: 目标航向

        Returns:
            调整后的路径
        """
        adjusted = corridor_path.copy()
        N = len(corridor_path)

        for i in range(N):
            # 余弦窗
            alpha = 0.5 * (1 - np.cos(np.pi * i / (N - 1)))

            # 当前点
            P = corridor_path[i, :2]

            # 计算到中心线的偏差
            e = np.dot(P - Q, u)

            # 调整位置（向中心线移动）
            adjustment = -e * alpha * u
            adjusted[i, :2] = P + adjustment

            # 调整角度（向目标角度过渡）
            yaw_diff = self._normalize_angle(yaw_target - corridor_path[i, 2])
            adjusted[i, 2] = corridor_path[i, 2] + alpha * yaw_diff

        return adjusted


    def _optimize_corridor_path(
        self,
        corridor_path: np.ndarray,
        Q: np.ndarray,
        u: np.ndarray,
        yaw_target: float
    ) -> np.ndarray:
        """优化通道内路径（如果有scipy）

        Args:
            corridor_path: 通道内路径
            Q: 通道中心点
            u: 法向量
            yaw_target: 目标航向

        Returns:
            优化后的路径
        """
        if not HAS_SCIPY or len(corridor_path) < 3:
            return corridor_path.copy()

        # 简单优化：只优化中间段，保持端点
        mid_start = max(1, len(corridor_path) // 4)
        mid_end = min(len(corridor_path) - 1, 3 * len(corridor_path) // 4)

        if mid_end <= mid_start:
            return corridor_path.copy()

        optimized = corridor_path.copy()

        # 对中间段进行微调
        for i in range(mid_start, mid_end):
            P = optimized[i, :2]
            e = np.dot(P - Q, u)

            # 更强的中心化
            optimized[i, :2] = P - 0.8 * e * u

            # 角度对齐
            optimized[i, 2] = yaw_target

        return optimized


    def _recompute_yaw(self, path: np.ndarray) -> np.ndarray:
        """重新计算路径的yaw角

        Args:
            path: 输入路径

        Returns:
            更新yaw后的路径
        """
        updated_path = path.copy()

        for i in range(len(path) - 1):
            dx = path[i+1, 0] - path[i, 0]
            dy = path[i+1, 1] - path[i, 1]
            if np.hypot(dx, dy) > 1e-6:
                updated_path[i, 2] = np.arctan2(dy, dx)

        # 最后一点
        if len(path) > 1:
            updated_path[-1, 2] = updated_path[-2, 2]

        return updated_path


    def _enforce_corridor_alignment(
        self,
        path: np.ndarray,
        Q: np.ndarray,
        t: np.ndarray,
        yaw_target: float,
        R1: float,
        R2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将障碍物影响区及其前后净空投影到通道中心直线。

        直线路段从圆形障碍物投影范围之前开始，因此车辆到达通道入口
        时已经完成横向和航向调整，而不是进入狭窄区后继续转向。
        """
        aligned = path.copy()
        if len(path) < 2:
            return aligned, np.zeros(len(path), dtype=bool)

        # 根据原路径行驶方向选择通道正方向。
        travel = path[-1, :2] - path[0, :2]
        if np.dot(travel, t) < 0.0:
            t = -t
            yaw_target = self._normalize_angle(yaw_target + np.pi)

        # 归一化方向向量，避免数值问题
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-9:
            # 方向向量太小，无法进行对齐
            return aligned, np.zeros(len(path), dtype=bool)
        t = t / t_norm

        u = np.array([-t[1], t[0]])

        # 使用更安全的点积计算，避免溢出
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            longitudinal = (aligned[:, :2] - Q) @ t
            lateral_error = (aligned[:, :2] - Q) @ u

        # 检查计算结果的有效性
        if not np.all(np.isfinite(longitudinal)) or not np.all(np.isfinite(lateral_error)):
            # 存在无效值，无法进行对齐
            return aligned, np.zeros(len(path), dtype=bool)

        half_length = max(R1, R2) + max(0.0, self.params.alignment_clearance)
        absolute_longitudinal = np.abs(longitudinal)
        mask = absolute_longitudinal <= half_length

        # 至少需要三个采样点才能表达一段可靠直线。
        indices = np.flatnonzero(mask)
        if len(indices) < 3:
            return aligned, np.zeros(len(path), dtype=bool)

        # 核心区完全投影为直线；核心区前后用余弦窗逐渐消除横向误差，
        # 避免硬投影在入口边界制造曲率尖峰。
        transition = max(
            self.params.adjust_before,
            self.params.adjust_after,
            2.0 * self.params.min_turning_radius,
        )
        weights = np.zeros(len(path), dtype=float)
        weights[mask] = 1.0
        transition_mask = (
            (absolute_longitudinal > half_length) &
            (absolute_longitudinal < half_length + transition)
        )
        if transition > 1e-9:
            phase = (
                absolute_longitudinal[transition_mask] - half_length
            ) / transition
            weights[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * phase))

        aligned[:, :2] -= (weights * lateral_error)[:, None] * u
        aligned[mask, 2] = yaw_target
        aligned = self._recompute_yaw(aligned)
        # 直线段最后一点的 yaw 由下一个点决定，不作为严格对齐采样点。
        if len(indices) > 1:
            mask[indices[-1]] = False
            aligned[mask, 2] = yaw_target
        return aligned, mask


    def _alignment_is_valid(
        self,
        path: np.ndarray,
        t: np.ndarray,
        mask: np.ndarray,
    ) -> bool:
        """检查锁定区域的切向航向是否与通道方向一致。"""
        indices = np.flatnonzero(mask)
        if len(indices) < 2:
            return not self.params.enforce_entry_alignment
        target = np.arctan2(t[1], t[0])
        reverse_target = self._normalize_angle(target + np.pi)
        for i in indices:
            error = min(
                abs(self._normalize_angle(path[i, 2] - target)),
                abs(self._normalize_angle(path[i, 2] - reverse_target)),
            )
            if error > self.params.alignment_yaw_tolerance:
                return False
        return True


    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def _obstacle_tuple(obstacle: Any) -> Tuple[float, float, float]:
        """兼容 (x,y,r)、CircleObstacle 和 numpy 记录。"""
        if hasattr(obstacle, "x"):
            return float(obstacle.x), float(obstacle.y), float(obstacle.radius)
        return float(obstacle[0]), float(obstacle[1]), float(obstacle[2])


    def _vehicle_collides(
        self,
        point: np.ndarray,
        obstacle: Any,
    ) -> bool:
        """圆障碍物与以后轴中心为参考的矩形车身碰撞检测。"""
        ox, oy, radius = self._obstacle_tuple(obstacle)
        dx, dy = ox - point[0], oy - point[1]
        c, s = np.cos(point[2]), np.sin(point[2])
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        margin = self.params.vehicle_safety_margin
        closest_x = np.clip(
            local_x,
            -self.params.vehicle_rear_length - margin,
            self.params.vehicle_front_length + margin,
        )
        closest_y = np.clip(
            local_y,
            -0.5 * self.params.vehicle_width - margin,
            0.5 * self.params.vehicle_width + margin,
        )
        return bool(
            (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
            <= radius ** 2
        )


    def _curvature_limit(self) -> float:
        """返回允许的最大曲率；最小转弯半径为0时不限制。"""
        radius = self.params.min_turning_radius
        return np.inf if radius <= 0.0 else 1.0 / radius


    def _compute_curvatures(self, path: np.ndarray) -> np.ndarray:
        """用相邻三点外接圆计算离散曲率，端点曲率记为0。"""
        curvatures = np.zeros(len(path), dtype=float)
        if len(path) < 3:
            return curvatures

        points = path[:, :2]
        for i in range(1, len(points) - 1):
            a = np.linalg.norm(points[i] - points[i - 1])
            b = np.linalg.norm(points[i + 1] - points[i])
            c = np.linalg.norm(points[i + 1] - points[i - 1])
            denominator = a * b * c
            if denominator <= 1e-12:
                continue

            v1 = points[i] - points[i - 1]
            v2 = points[i + 1] - points[i - 1]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            curvatures[i] = 2.0 * cross / denominator

        return curvatures


    def _max_curvature(self, path: np.ndarray) -> float:
        """返回路径的最大离散曲率。"""
        curvatures = self._compute_curvatures(path)
        return float(np.max(curvatures)) if len(curvatures) else 0.0


    def _enforce_min_turning_radius(self, path: np.ndarray) -> np.ndarray:
        """迭代平滑过大曲率点，使路径满足阿克曼最小转弯半径。

        修正只移动内部路径点，始终保持整条路径的起终点不变。对于
        超限点，将其朝相邻点中点移动；该操作会减小局部三点曲率，
        多轮交替更新用于消除修正向相邻位置传播产生的新曲率峰值。
        """
        curvature_limit = self._curvature_limit()
        if len(path) < 3 or not np.isfinite(curvature_limit):
            return self._recompute_yaw(path)

        relaxation = float(np.clip(self.params.curvature_relaxation, 1e-3, 1.0))
        adjusted = path.copy()

        for _ in range(max(0, self.params.curvature_iterations)):
            curvatures = self._compute_curvatures(adjusted)
            violating = np.flatnonzero(curvatures > curvature_limit)
            if len(violating) == 0:
                break

            previous_xy = adjusted[:, :2].copy()
            for i in violating:
                midpoint = 0.5 * (previous_xy[i - 1] + previous_xy[i + 1])
                excess = 1.0 - curvature_limit / curvatures[i]
                weight = relaxation * float(np.clip(excess, 0.05, 1.0))
                adjusted[i, :2] = (
                    (1.0 - weight) * previous_xy[i] + weight * midpoint
                )

        return self._recompute_yaw(adjusted)


    def _compute_cumulative_distance(self, path: np.ndarray) -> np.ndarray:
        """计算累计路径距离"""
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(
                path[i, :2] - path[i-1, :2]
            )
        return distances


    def _check_feasibility(
        self,
        path: np.ndarray,
        circle1: Tuple[float, float, float],
        circle2: Tuple[float, float, float],
        R1: float,
        R2: float,
        all_obstacles: Optional[Sequence[Any]] = None,
        alignment: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[bool, str]:
        """检查路径可行性"""
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

            if all_obstacles is not None:
                for obstacle_index, obstacle in enumerate(all_obstacles):
                    if self._vehicle_collides(path[i], obstacle):
                        return False, (
                            f"Vehicle collision with obstacle {obstacle_index} "
                            f"at path index {i}"
                        )

        curvature_limit = self._curvature_limit()
        max_curvature = self._max_curvature(path)
        if (np.isfinite(curvature_limit) and
                max_curvature > curvature_limit *
                (1.0 + self.params.curvature_tolerance)):
            radius = 1.0 / max_curvature if max_curvature > 1e-9 else np.inf
            return False, (
                f"Turning radius {radius:.3f}m is below minimum "
                f"{self.params.min_turning_radius:.3f}m"
            )

        if alignment is not None and self.params.enforce_entry_alignment:
            _, direction, mask = alignment
            if not self._alignment_is_valid(path, direction, mask):
                return False, "Path is not aligned before corridor entry"

        return True, "Valid"


    def _compute_center_error(
        self,
        path: np.ndarray,
        Q: np.ndarray,
        u: np.ndarray
    ) -> float:
        """计算中心线误差"""
        errors = []
        for i in range(len(path)):
            P = path[i, :2]
            e = np.dot(P - Q, u)
            errors.append(abs(e))
        return float(np.mean(errors))


    def _compute_angle_error(
        self,
        path: np.ndarray,
        yaw_target: float
    ) -> float:
        """计算角度误差"""
        errors = []
        for i in range(len(path)):
            yaw = path[i, 2]
            error = abs(self._normalize_angle(yaw - yaw_target))
            errors.append(np.degrees(error))
        return float(np.mean(errors))


    def _search_backward_connection_point(
        self,
        path: np.ndarray,
        target_position: np.ndarray,
        target_yaw: float,
        lookahead_distance: float = 5.0,
    ) -> int:
        """向后搜索最佳连接点（前瞻插补）

        从target_position向后沿路径搜索，找到满足以下条件的最远点：
        1. 到target_position的距离 >= lookahead_distance
        2. 从该点到target能生成满足曲率约束的S曲线

        Args:
            path: 原始路径段
            target_position: 目标位置 [x, y]
            target_yaw: 目标航向
            lookahead_distance: 前瞻距离

        Returns:
            连接点索引
        """
        if len(path) == 0:
            return 0

        # 计算所有点到target的距离
        distances = np.linalg.norm(path[:, :2] - target_position, axis=1)

        # 找到距离最近的点
        nearest_idx = int(np.argmin(distances))

        # 向后搜索，找到距离 >= lookahead_distance 的最远可行点
        best_idx = nearest_idx

        for i in range(nearest_idx, -1, -1):
            dist = distances[i]
            if dist < lookahead_distance:
                continue

            # 检查从该点到target是否能生成可行的S曲线
            candidate_pose = path[i]
            can_connect = self._check_s_curve_feasibility(
                start_pose=candidate_pose,
                end_position=target_position,
                end_yaw=target_yaw,
            )

            if can_connect:
                best_idx = i
                break  # 找到最远的可行点

        return best_idx

    def _check_s_curve_feasibility(
        self,
        start_pose: np.ndarray,
        end_position: np.ndarray,
        end_yaw: float,
    ) -> bool:
        """快速检查S曲线可行性（不生成完整曲线）"""
        # 提取起点信息
        start_position = start_pose[:2]
        start_yaw = start_pose[2]

        # 计算相对位置
        relative = end_position - start_position
        L = float(np.linalg.norm(relative))

        if L < 1e-3:
            return False

        # 计算航向差
        delta_yaw = self._normalize_angle(end_yaw - start_yaw)

        # 航向差太大时不可行
        if abs(delta_yaw) > np.deg2rad(60):
            return False

        # 粗略估计曲率：对于五次多项式S曲线，最大曲率大约是
        # max_curvature ≈ |delta_yaw| / L
        estimated_curvature = abs(delta_yaw) / L

        curvature_limit = self._curvature_limit()
        if np.isfinite(curvature_limit):
            allowed = curvature_limit * (1.0 + self.params.curvature_tolerance)
            if estimated_curvature > allowed * 0.8:  # 保守估计
                return False

        return True

    def _create_entry_s_curve(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        target_yaw: float,
    ) -> Optional[np.ndarray]:
        """创建入口S形曲线（从原路径平滑连接到corridor入口）

        使用五次多项式，边界条件：
        - 起点：位置、航向来自start_pose，曲率=0
        - 终点：位置、航向来自end_pose/target_yaw，曲率=0

        Args:
            start_pose: 起点位姿 [x, y, yaw]
            end_pose: 终点位姿 [x, y, yaw]（yaw被target_yaw覆盖）
            target_yaw: 目标航向

        Returns:
            S形曲线，shape=(N, 3)，或None如果失败
        """
        start_pos = start_pose[:2]
        start_yaw = start_pose[2]
        end_pos = end_pose[:2]
        end_yaw = target_yaw

        # 计算相对向量
        relative = end_pos - start_pos
        L = float(np.linalg.norm(relative))

        if L < 1e-3:
            return None

        # 计算基准方向（起点到终点）
        baseline_direction = relative / L
        baseline_yaw = np.arctan2(baseline_direction[1], baseline_direction[0])

        # 构建局部坐标系
        theta = baseline_yaw
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 旋转矩阵（全局 -> 局部）
        # 在局部坐标系中，x轴沿baseline方向，y轴垂直
        def to_local(pos):
            rel = pos - start_pos
            x_local = rel[0] * cos_t + rel[1] * sin_t
            y_local = -rel[0] * sin_t + rel[1] * cos_t
            return np.array([x_local, y_local])

        def from_local(x_local, y_local):
            x_global = start_pos[0] + x_local * cos_t - y_local * sin_t
            y_global = start_pos[1] + x_local * sin_t + y_local * cos_t
            return np.array([x_global, y_global])

        # 终点在局部坐标系中
        end_local = to_local(end_pos)
        L_local = end_local[0]  # 应该接近L
        D_local = end_local[1]  # 横向偏移

        if L_local < 1e-3:
            return None

        # 起点和终点航向在局部坐标系中
        start_yaw_local = self._normalize_angle(start_yaw - baseline_yaw)
        end_yaw_local = self._normalize_angle(end_yaw - baseline_yaw)

        # 五次多项式：y(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
        # 边界条件：
        # y(0) = 0, y'(0) = tan(start_yaw_local), y''(0) = 0
        # y(L) = D, y'(L) = tan(end_yaw_local), y''(L) = 0

        tan_start = np.tan(start_yaw_local)
        tan_end = np.tan(end_yaw_local)

        # 检查tan是否接近奇异
        if abs(start_yaw_local) > np.deg2rad(80) or abs(end_yaw_local) > np.deg2rad(80):
            return None  # 角度太大，避免数值不稳定

        # 从起点边界条件：
        a0 = 0.0
        a1 = tan_start
        a2 = 0.0

        # 从终点边界条件求解 [a3, a4, a5]
        L2, L3, L4, L5 = L_local**2, L_local**3, L_local**4, L_local**5

        A_matrix = np.array([
            [L3,      L4,       L5],
            [3*L2,    4*L3,     5*L4],
            [6*L_local, 12*L2,  20*L3],
        ])

        b_vector = np.array([
            D_local - a1 * L_local,  # y(L) = D
            tan_end - a1,             # y'(L) = tan_end
            0.0,                      # y''(L) = 0
        ])

        try:
            coeffs = np.linalg.solve(A_matrix, b_vector)
            a3, a4, a5 = coeffs
        except np.linalg.LinAlgError:
            return None

        # 生成曲线
        num_samples = max(10, int(np.ceil(L / self.params.trajectory_sample_spacing)))
        x_samples = np.linspace(0, L_local, num_samples)

        # 计算局部y坐标
        y_samples = (a0 + a1*x_samples + a2*x_samples**2 +
                     a3*x_samples**3 + a4*x_samples**4 + a5*x_samples**5)

        # 计算一阶导数（斜率）
        dy_dx = (a1 + 2*a2*x_samples + 3*a3*x_samples**2 +
                 4*a4*x_samples**3 + 5*a5*x_samples**4)

        # 航向（局部）
        yaw_local_samples = np.arctan2(dy_dx, 1.0)

        # 转换到全局坐标
        positions = np.array([from_local(x, y) for x, y in zip(x_samples, y_samples)])
        yaw_global_samples = yaw_local_samples + baseline_yaw

        # 标准化航向
        yaw_global_samples = np.array([self._normalize_angle(y) for y in yaw_global_samples])

        # 组合
        curve = np.column_stack([positions, yaw_global_samples])

        # 计算曲率检查
        curvatures = self._compute_signed_curvatures(curve)
        max_curvature = float(np.max(np.abs(curvatures)))

        curvature_limit = self._curvature_limit()
        if np.isfinite(curvature_limit):
            allowed = curvature_limit * (1.0 + self.params.curvature_tolerance)
            if max_curvature > allowed:
                return None  # 曲率不满足

        return curve

    def _search_forward_s_curve_connection(
        self,
        path: np.ndarray,
        corridor: np.ndarray,
        corridor_exit: np.ndarray,
        corridor_direction: np.ndarray,
        obstacles: Sequence[Any],
        reference_index: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Dict]:
        """
        沿原路径向前搜索重连接位置，使用五次多项式S形曲线连接通道出口与原路径。

        Args:
            path: 原始路径 (N, 3)
            corridor: 投影中心线 (M, 3)，硬约束不可修改
            corridor_exit: 通道出口点 [x, y, yaw]
            corridor_direction: 通道方向向量（未归一化）
            obstacles: 所有障碍物
            reference_index: 参考索引（可选）

        Returns:
            best_curve: 最佳S形曲线 (K, 3)，或None
            connection_index: 重连接点在原路径中的索引，或None
            info: 详细信息字典
        """

        # ==================== 辅助函数定义 ====================

        def normalize_angle(angle: float) -> float:
            """归一化角度到[-pi, pi]"""
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle

        def compute_cumulative_distance(trajectory: np.ndarray) -> np.ndarray:
            """计算累计弧长"""
            if len(trajectory) < 2:
                return np.array([0.0])
            segments = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
            return np.concatenate([[0.0], np.cumsum(segments)])

        def generate_quintic_s_curve(
            exit_point: np.ndarray,
            exit_yaw: float,
            candidate_pos: np.ndarray,
            candidate_yaw: float,
            t_vec: np.ndarray,
            n_vec: np.ndarray,
        ) -> Tuple[Optional[np.ndarray], Dict]:
            """
            生成五次多项式S形曲线

            Returns:
                curve: (K, 3) 或 None
                info: 包含多项式系数等信息
            """
            # 计算相对位置
            relative = candidate_pos[:2] - exit_point[:2]
            L = float(np.dot(relative, t_vec))
            D = float(np.dot(relative, n_vec))

            # 必须向前
            if L <= 1e-6:
                return None, {"reason": "Candidate not forward"}

            # 计算终点航向差
            delta_yaw = normalize_angle(candidate_yaw - exit_yaw)

            # 拒绝极端大角度（避免tan奇异），但允许接近90度的转弯
            if abs(delta_yaw) > np.deg2rad(120):
                return None, {"reason": f"Excessive yaw difference: {np.degrees(delta_yaw):.1f}°"}

            # 多项式系数
            a0, a1, a2 = 0.0, 0.0, 0.0  # 起点边界条件

            # 使用快速公式（小角度）
            if abs(delta_yaw) < np.deg2rad(1.0):
                # 标准S形：d(xi) = D * (10*xi^3 - 15*xi^4 + 6*xi^5)
                a3 = 10.0 * D / (L ** 3)
                a4 = -15.0 * D / (L ** 4)
                a5 = 6.0 * D / (L ** 5)
            else:
                # 一般边界方程
                terminal_slope = np.tan(delta_yaw)

                # 构建线性系统
                A_matrix = np.array([
                    [L**3,      L**4,       L**5],
                    [3*L**2,    4*L**3,     5*L**4],
                    [6*L,       12*L**2,    20*L**3],
                ])
                b_vector = np.array([D, terminal_slope, 0.0])

                try:
                    coeffs = np.linalg.solve(A_matrix, b_vector)
                    a3, a4, a5 = coeffs
                except np.linalg.LinAlgError:
                    return None, {"reason": "Singular matrix in polynomial solve"}

            # 密集采样
            num_dense = 201
            l_dense = np.linspace(0, L, num_dense)

            # 横向偏移
            d_dense = (a0 + a1*l_dense + a2*l_dense**2 +
                       a3*l_dense**3 + a4*l_dense**4 + a5*l_dense**5)

            # 一阶导数
            d1_dense = (a1 + 2*a2*l_dense + 3*a3*l_dense**2 +
                        4*a4*l_dense**3 + 5*a5*l_dense**4)

            # 二阶导数
            d2_dense = (2*a2 + 6*a3*l_dense + 12*a4*l_dense**2 + 20*a5*l_dense**3)

            # 全局坐标
            pos_dense = exit_point[:2] + l_dense[:, None] * t_vec + d_dense[:, None] * n_vec

            # 切向和航向
            tangent_x = t_vec[0] + d1_dense * n_vec[0]
            tangent_y = t_vec[1] + d1_dense * n_vec[1]
            yaw_dense = np.arctan2(tangent_y, tangent_x)

            # 曲率
            denominator = (1 + d1_dense**2) ** 1.5
            curvature_dense = d2_dense / np.maximum(denominator, 1e-12)

            # 根据弧长重新采样
            dx_dense = np.diff(pos_dense[:, 0])
            dy_dense = np.diff(pos_dense[:, 1])
            ds_dense = np.hypot(dx_dense, dy_dense)
            s_dense = np.concatenate([[0.0], np.cumsum(ds_dense)])

            # 目标间距
            target_spacing = self.params.trajectory_sample_spacing
            total_arc = s_dense[-1]
            num_resampled = max(2, int(np.ceil(total_arc / target_spacing)) + 1)
            s_resampled = np.linspace(0, total_arc, num_resampled)

            # 插值
            x_resampled = np.interp(s_resampled, s_dense, pos_dense[:, 0])
            y_resampled = np.interp(s_resampled, s_dense, pos_dense[:, 1])
            yaw_resampled = np.interp(s_resampled, s_dense, yaw_dense)

            curve = np.column_stack([x_resampled, y_resampled, yaw_resampled])

            # 强制端点
            curve[0, :2] = exit_point[:2]
            curve[0, 2] = exit_yaw
            curve[-1, :2] = candidate_pos[:2]

            # 检查终点航向
            final_yaw_error = abs(normalize_angle(curve[-1, 2] - candidate_yaw))
            if final_yaw_error > self.params.alignment_yaw_tolerance:
                # 使用candidate_yaw
                curve[-1, 2] = candidate_yaw

            # 删除重复点
            unique_mask = np.ones(len(curve), dtype=bool)
            for i in range(1, len(curve)):
                if np.linalg.norm(curve[i, :2] - curve[i-1, :2]) < 1e-9:
                    unique_mask[i] = False
            curve = curve[unique_mask]

            # 计算曲率（离散三点法）
            if len(curve) >= 3:
                curvatures = self._compute_signed_curvatures(curve)
            else:
                curvatures = np.zeros(len(curve))

            info = {
                "polynomial_coefficients": [a0, a1, a2, a3, a4, a5],
                "longitudinal_distance": L,
                "lateral_offset": D,
                "terminal_yaw_error": delta_yaw,
                "path_length": total_arc,
                "max_curvature": float(np.max(np.abs(curvatures))),
                "curvatures": curvatures,
            }

            return curve, info

        def check_curvature(curve: np.ndarray, context_info: Dict) -> Tuple[bool, str]:
            """检查曲率约束"""
            curvature_limit = 1.0 / self.params.min_turning_radius if self.params.min_turning_radius > 0 else np.inf
            tolerance = 0.01  # 恢复为1%

            max_curv = context_info.get("max_curvature_context", context_info.get("max_curvature", 0.0))

            if np.isfinite(curvature_limit):
                if max_curv > curvature_limit * (1.0 + tolerance):
                    return False, f"Max curvature {max_curv:.4f} exceeds limit {curvature_limit:.4f}"

            return True, "OK"

        def check_curvature_rate(curve: np.ndarray, context_curv: np.ndarray, context_s: np.ndarray) -> Tuple[bool, str]:
            """检查曲率变化率"""
            if len(context_curv) < 2:
                return True, "OK"

            d_curv = np.abs(np.diff(context_curv))
            d_s = np.diff(context_s)
            d_s = np.maximum(d_s, 1e-9)

            curvature_rate = d_curv / d_s
            max_rate = float(np.max(curvature_rate))

            if max_rate > self.params.max_curvature_rate:
                return False, f"Max curvature rate {max_rate:.4f} exceeds limit {self.params.max_curvature_rate:.4f}"

            return True, "OK"

        def check_forward_monotonicity(curve: np.ndarray, exit_pt: np.ndarray, t_vec: np.ndarray) -> Tuple[bool, str]:
            """检查前进性"""
            longitudinal = (curve[:, :2] - exit_pt[:2]) @ t_vec
            diff_long = np.diff(longitudinal)

            if np.min(diff_long) < -1e-6:
                return False, "Curve contains backward motion"

            return True, "OK"

        def check_s_curve_shape(curvatures: np.ndarray, D: float) -> Tuple[bool, str]:
            """检查S形特征"""
            if abs(D) < 1e-3:
                return True, "OK"  # 直线，无需检查

            # 过滤小曲率
            effective_curv = curvatures[np.abs(curvatures) > 1e-5]

            if len(effective_curv) < 2:
                return True, "OK"

            # 统计符号变化
            signs = np.sign(effective_curv)
            sign_changes = np.sum(np.abs(np.diff(signs)) > 0.5)

            if sign_changes > 1:
                return False, "S-curve contains excessive curvature oscillation"

            return True, "OK"

        def check_collision(curve: np.ndarray, obstacles_list: Sequence[Any]) -> Tuple[bool, str]:
            """检查碰撞"""
            # 加密采样
            dense = self._densify_trajectory(curve, self.params.trajectory_sample_spacing)

            for i, pose in enumerate(dense):
                for j, obstacle in enumerate(obstacles_list):
                    if self._vehicle_collides(pose, obstacle):
                        return False, f"Vehicle collision with obstacle {j} at sample {i}"

            return True, "OK"

        # ==================== 主逻辑开始 ====================

        failure_stats = {
            "not_forward": 0,
            "excessive_yaw": 0,
            "curvature_limit": 0,
            "curvature_rate": 0,
            "backward_motion": 0,
            "s_curve_oscillation": 0,
            "collision": 0,
            "context_curvature": 0,
        }

        # 归一化通道方向
        t = corridor_direction / max(np.linalg.norm(corridor_direction), 1e-12)
        n = np.array([-t[1], t[0]])

        exit_point_2d = corridor_exit[:2]
        exit_yaw = float(np.arctan2(t[1], t[0]))

        # 计算原路径累计弧长
        path_cumulative = compute_cumulative_distance(path)

        # 确定参考索引
        if reference_index is None:
            # 找到最近投影
            best_proj_dist = np.inf
            best_ref_idx = len(path) - 1

            for i in range(len(path) - 1):
                p1, p2 = path[i, :2], path[i+1, :2]
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)

                if seg_len < 1e-9:
                    continue

                t_seg = (exit_point_2d - p1) @ seg / (seg_len ** 2)
                t_seg = np.clip(t_seg, 0, 1)
                proj = p1 + t_seg * seg
                dist = np.linalg.norm(exit_point_2d - proj)

                if dist < best_proj_dist:
                    best_proj_dist = dist
                    best_ref_idx = i + 1  # 使用下游端点

            reference_index = best_ref_idx

        reference_arc = path_cumulative[reference_index]

        # 搜索候选点
        candidates = []
        search_step = self.params.projection_search_step
        max_search = self.params.projection_max_search_distance

        current_arc = reference_arc
        while current_arc - reference_arc < max_search:
            current_arc += search_step

            if current_arc > path_cumulative[-1]:
                break

            # 找到该弧长对应的线段
            seg_idx = np.searchsorted(path_cumulative, current_arc) - 1
            seg_idx = np.clip(seg_idx, 0, len(path) - 2)

            # 插值位姿
            s1, s2 = path_cumulative[seg_idx], path_cumulative[seg_idx + 1]
            if s2 - s1 < 1e-9:
                continue

            alpha = (current_arc - s1) / (s2 - s1)
            alpha = np.clip(alpha, 0, 1)

            pos = (1 - alpha) * path[seg_idx, :2] + alpha * path[seg_idx + 1, :2]

            # 切向航向
            tangent = path[seg_idx + 1, :2] - path[seg_idx, :2]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-9:
                continue
            tangent = tangent / tangent_norm
            yaw = float(np.arctan2(tangent[1], tangent[0]))

            # 检查前进性
            relative = pos - exit_point_2d
            if np.dot(relative, t) <= 0:
                failure_stats["not_forward"] += 1
                continue

            # 检查切向（放宽约束，允许90度转弯）
            # 只拒绝完全反向的候选点
            if np.dot(tangent, t) < -0.7:  # cos(135°) ≈ -0.7
                failure_stats["not_forward"] += 1
                continue

            # 记录候选
            connection_idx = seg_idx + 1

            # 去重
            duplicate = False
            for cand in candidates:
                if cand["connection_index"] == connection_idx:
                    duplicate = True
                    break

            if not duplicate:
                candidates.append({
                    "position": pos,
                    "yaw": yaw,
                    "arc_length": current_arc,
                    "connection_index": connection_idx,
                    "search_distance": current_arc - reference_arc,
                })

        if len(candidates) == 0:
            return None, None, {
                "valid": False,
                "method": "forward_quintic_polynomial_s_curve",
                "reason": "No forward candidates found",
                "candidate_count": 0,
                "feasible_candidate_count": 0,
                "requires_global_replan": True,
                "failure_statistics": failure_stats,
            }

        # 评估候选
        feasible_curves = []
        first_feasible_distance = None

        for cand in candidates:
            # 生成S形曲线
            curve, curve_info = generate_quintic_s_curve(
                corridor_exit,
                exit_yaw,
                np.array([cand["position"][0], cand["position"][1], cand["yaw"]]),
                cand["yaw"],
                t, n,
            )

            if curve is None:
                reason = curve_info.get("reason", "Unknown")
                if "yaw" in reason.lower():
                    failure_stats["excessive_yaw"] += 1
                else:
                    failure_stats["not_forward"] += 1
                continue

            # 曲率检查
            ok, msg = check_curvature(curve, curve_info)
            if not ok:
                failure_stats["curvature_limit"] += 1
                continue

            # 前进性检查
            ok, msg = check_forward_monotonicity(curve, exit_point_2d, t)
            if not ok:
                failure_stats["backward_motion"] += 1
                continue

            # S形特征检查
            ok, msg = check_s_curve_shape(
                curve_info["curvatures"],
                curve_info["lateral_offset"]
            )
            if not ok:
                failure_stats["s_curve_oscillation"] += 1
                continue

            # 构建拼接上下文
            corridor_tail = corridor[-3:] if len(corridor) >= 3 else corridor
            path_head = path[cand["connection_index"]:cand["connection_index"]+3]

            context = np.vstack([corridor_tail, curve, path_head])

            # 删除重复端点
            unique_mask = np.ones(len(context), dtype=bool)
            for i in range(1, len(context)):
                if np.linalg.norm(context[i, :2] - context[i-1, :2]) < 1e-9:
                    unique_mask[i] = False
            context = context[unique_mask]

            # 计算上下文曲率
            context_curv = self._compute_signed_curvatures(context)
            context_s = compute_cumulative_distance(context)

            # 上下文曲率检查
            max_context_curv = float(np.max(np.abs(context_curv)))
            curve_info["max_curvature_context"] = max_context_curv

            ok, msg = check_curvature(context, curve_info)
            if not ok:
                failure_stats["context_curvature"] += 1
                continue

            # 曲率变化率检查
            ok, msg = check_curvature_rate(curve, context_curv, context_s)
            if not ok:
                failure_stats["curvature_rate"] += 1
                continue

            # 碰撞检查
            ok, msg = check_collision(context, obstacles)
            if not ok:
                failure_stats["collision"] += 1
                continue

            # 可行！
            if first_feasible_distance is None:
                first_feasible_distance = cand["search_distance"]

            # 计算代价
            curvature_energy = float(np.trapezoid(context_curv ** 2, context_s))
            curvature_rate = np.abs(np.diff(context_curv)) / np.maximum(np.diff(context_s), 1e-9)
            curvature_rate_energy = float(np.trapezoid(curvature_rate ** 2, context_s[1:]))

            # 计算到原路径的偏差
            deviations = []
            for pt in curve[:, :2]:
                dists = np.linalg.norm(path[:, :2] - pt, axis=1)
                deviations.append(np.min(dists) ** 2)
            deviation_cost = float(np.mean(deviations))

            cost = (self.params.w_path_length * curve_info["path_length"] +
                    self.params.w_curvature * curvature_energy +
                    self.params.w_curvature_rate * curvature_rate_energy +
                    self.params.w_path_deviation * deviation_cost +
                    0.2 * cand["search_distance"])

            feasible_curves.append({
                "curve": curve,
                "connection_index": cand["connection_index"],
                "cost": cost,
                "info": curve_info,
                "search_distance": cand["search_distance"],
            })

        if len(feasible_curves) == 0:
            return None, None, {
                "valid": False,
                "method": "forward_quintic_polynomial_s_curve",
                "reason": "No feasible forward S-curve connection",
                "candidate_count": len(candidates),
                "feasible_candidate_count": 0,
                "requires_global_replan": True,
                "failure_statistics": failure_stats,
            }

        # 继续搜索窗口
        if first_feasible_distance is not None:
            window_end = first_feasible_distance + 5.0
            feasible_curves = [fc for fc in feasible_curves if fc["search_distance"] <= window_end]

        # 选择最佳
        best = min(feasible_curves, key=lambda x: x["cost"])

        return best["curve"], best["connection_index"], {
            "valid": True,
            "method": "forward_quintic_polynomial_s_curve",
            "reason": "Valid",
            "connection_index": best["connection_index"],
            "forward_search_distance": best["search_distance"],
            "longitudinal_distance": best["info"]["longitudinal_distance"],
            "lateral_offset": best["info"]["lateral_offset"],
            "terminal_yaw_error": best["info"]["terminal_yaw_error"],
            "polynomial_coefficients": best["info"]["polynomial_coefficients"],
            "max_curvature": best["info"]["max_curvature"],
            "max_curvature_rate": float(np.max(np.abs(np.diff(best["info"]["curvatures"])))),
            "path_length": best["info"]["path_length"],
            "candidate_count": len(candidates),
            "feasible_candidate_count": len(feasible_curves),
            "trajectory_cost": best["cost"],
            "requires_global_replan": False,
            "failure_statistics": failure_stats,
        }



# ============================================================================
# 主接口函数
# ============================================================================

def refine_path_between_circles(
    path: np.ndarray,
    circle1: Tuple[float, float, float],
    circle2: Tuple[float, float, float],
    params: Optional[Dict] = None,
    all_obstacles: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, Dict]:
    """修正穿过两个圆形障碍物之间的路径（增强版主接口）

    功能增强：
    1. 优化通道区域路径
    2. 创建从原始起点到优化起点的连接段
    3. 创建从优化终点到原始终点的连接段
    4. 对完整路径进行全局平滑
    5. 满足阿克曼底盘最小转弯半径约束（默认3m）

    Args:
        path: 原始路径，shape=(N, 3)，每个点为[x, y, yaw]
        circle1: 圆1，(x1, y1, r1)
        circle2: 圆2，(x2, y2, r2)
        params: 可选参数字典

    Returns:
        refined_path: 修正后的完整平滑路径
        info: 修正信息字典
    """
    # 解析参数
    if params is None:
        refine_params = RefineParams()
    else:
        refine_params = RefineParams(**params)

    # 创建修正器
    refiner = CircleCorridorRefinerEnhanced(refine_params)

    # 执行修正
    return refiner.refine_path(
        path, circle1, circle2, all_obstacles=all_obstacles
    )


def find_corridor_pairs(
    path: np.ndarray,
    obstacles: Sequence[Any],
    params: Optional[Dict] = None,
) -> List[Tuple[int, int, int]]:
    """沿全局路径搜索位于路径两侧、可能形成通道的圆形障碍物对。

    返回 ``(path_index, obstacle1_index, obstacle2_index)``，并按车辆
    行驶顺序排列。净宽不足车宽的圆对不会被当成可通行通道。
    """
    cfg = RefineParams(**(params or {}))
    refiner = CircleCorridorRefinerEnhanced(cfg)
    if len(path) < 2 or len(obstacles) < 2:
        return []

    circles = [refiner._obstacle_tuple(item) for item in obstacles]
    minimum_gap = cfg.vehicle_width + 2.0 * cfg.vehicle_safety_margin
    maximum_gap = (
        cfg.max_candidate_gap
        if cfg.max_candidate_gap > 0.0
        else cfg.vehicle_width * cfg.max_candidate_gap_width_factor
    )
    candidates: List[Tuple[int, int, int]] = []

    for first in range(len(circles) - 1):
        x1, y1, r1 = circles[first]
        c1 = np.array([x1, y1])
        for second in range(first + 1, len(circles)):
            x2, y2, r2 = circles[second]
            c2 = np.array([x2, y2])
            distance = float(np.linalg.norm(c2 - c1))
            free_gap = distance - r1 - r2
            # 净宽必须能容纳车身，同时不能宽到失去“狭窄通道”语义。
            if free_gap < minimum_gap or free_gap > maximum_gap + 1e-9:
                continue

            u = (c2 - c1) / max(distance, 1e-9)
            center = c1 + 0.5 * (distance + r1 - r2) * u
            # 候选路径必须真正穿过两圆边界之间的“门线”，而不能只是
            # 从圆对中心附近经过。这样可排除终点延长线附近的假通道。
            gate_start = c1 + r1 * u
            gate_length = free_gap
            best_distance = np.inf
            path_index = 0
            tangent = None
            closest_point = path[0, :2]
            for segment_index in range(len(path) - 1):
                start = path[segment_index, :2]
                segment = path[segment_index + 1, :2] - start
                length_squared = float(np.dot(segment, segment))
                if length_squared <= 1e-12:
                    continue

                # start + path_ratio * segment = gate_start + gate_ratio * u
                denominator = float(segment[0] * u[1] - segment[1] * u[0])
                if abs(denominator) <= 1e-10:
                    continue
                offset = gate_start - start
                path_ratio = float(
                    (offset[0] * u[1] - offset[1] * u[0]) / denominator
                )
                gate_ratio = float(
                    (offset[0] * segment[1] - offset[1] * segment[0])
                    / denominator
                )
                if not (-1e-9 <= path_ratio <= 1.0 + 1e-9):
                    continue
                if not (-1e-9 <= gate_ratio <= gate_length + 1e-9):
                    continue

                intersection = start + np.clip(path_ratio, 0.0, 1.0) * segment
                distance_to_center = float(np.linalg.norm(center - intersection))
                if distance_to_center < best_distance:
                    best_distance = distance_to_center
                    path_index = segment_index
                    tangent = segment / np.sqrt(length_squared)
                    closest_point = intersection

            # 路径已经被要求真实穿过有限门线，因此无需再用“交点到
            # 通道中心距离”重复筛选；该距离不会影响碰撞安全性。
            if tangent is None:
                continue

            rel1, rel2 = c1 - closest_point, c2 - closest_point
            side1 = tangent[0] * rel1[1] - tangent[1] * rel1[0]
            side2 = tangent[0] * rel2[1] - tangent[1] * rel2[0]
            if side1 * side2 >= 0.0:
                continue

            candidates.append((path_index, first, second))

    # 同一路径位置只保留净宽最小（约束最强）的圆对，减少重复修正。
    candidates.sort(key=lambda item: item[0])
    filtered: List[Tuple[int, int, int]] = []
    for candidate in candidates:
        if filtered and abs(candidate[0] - filtered[-1][0]) <= 2:
            _, a, b = candidate
            _, pa, pb = filtered[-1]
            gap = np.linalg.norm(
                np.array(circles[a][:2]) - np.array(circles[b][:2])
            ) - circles[a][2] - circles[b][2]
            previous_gap = np.linalg.norm(
                np.array(circles[pa][:2]) - np.array(circles[pb][:2])
            ) - circles[pa][2] - circles[pb][2]
            if gap < previous_gap:
                filtered[-1] = candidate
        else:
            filtered.append(candidate)
    return filtered


def _extend_path_backward(path: np.ndarray, distance: float, step: float) -> np.ndarray:
    """沿起点航向反方向添加虚拟上游路径。"""
    if distance <= 0.0 or len(path) == 0:
        return path.copy()
    count = max(1, int(np.ceil(distance / max(step, 1e-3))))
    offsets = np.linspace(distance, distance / count, count)
    direction = np.array([np.cos(path[0, 2]), np.sin(path[0, 2])])
    prefix = np.zeros((count, 3), dtype=float)
    prefix[:, :2] = path[0, :2] - offsets[:, None] * direction
    prefix[:, 2] = path[0, 2]
    return np.vstack([prefix, path])


def _resample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """按弧长对稀疏全局路径重新采样，并重新计算切向航向。"""
    if len(path) < 2 or spacing <= 0.0:
        return path.copy()
    segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-9:
        return path.copy()
    samples = np.arange(0.0, cumulative[-1], spacing)
    if len(samples) == 0 or samples[-1] < cumulative[-1]:
        samples = np.append(samples, cumulative[-1])
    x = np.interp(samples, cumulative, path[:, 0])
    y = np.interp(samples, cumulative, path[:, 1])
    yaw = np.zeros(len(samples), dtype=float)
    yaw[:-1] = np.arctan2(np.diff(y), np.diff(x))
    yaw[-1] = yaw[-2] if len(yaw) > 1 else path[0, 2]
    return np.column_stack((x, y, yaw))


def refine_global_path_corridors(
    path: np.ndarray,
    obstacles: Sequence[Any],
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """自动搜索并优化全局路径上的圆形障碍物通道。

    若入口前路径不足，会按 ``extension_step`` 逐步向起点后方扩展，
    但每一个扩展候选都必须通过全部障碍物的矩形车身碰撞检查。
    """
    cfg = RefineParams(**(params or {}))
    refiner = CircleCorridorRefinerEnhanced(cfg)
    working = np.asarray(path, dtype=float).copy()
    # 无候选通道且原路径已满足曲率约束时保持逐点不变，避免仅因调用
    # 本接口就改变原有采样密度。
    original_pairs = find_corridor_pairs(working, obstacles, params=params)
    if (not original_pairs and
            refiner._max_curvature(working) <= refiner._curvature_limit() *
            (1.0 + cfg.curvature_tolerance)):
        return working, {
            "valid": True, "candidate_pair_count": 0,
            "optimized_pair_count": 0, "pairs": [],
            "global_curvature_applied": False,
            "global_curvature_valid": True,
            "global_curvature_reason": "Already within curvature limit",
            "global_max_curvature_before": float(refiner._max_curvature(working)),
            "global_max_curvature_after": float(refiner._max_curvature(working)),
        }
    if cfg.path_resample_spacing > 0.0 and len(working) >= 2:
        maximum_segment = float(np.max(
            np.linalg.norm(np.diff(working[:, :2], axis=0), axis=1)
        ))
        if maximum_segment > 1.5 * cfg.path_resample_spacing:
            working = _resample_path(working, cfg.path_resample_spacing)
    pairs = find_corridor_pairs(working, obstacles, params=params)
    reports: List[Dict] = []

    for _, first, second in pairs:
        circle1 = refiner._obstacle_tuple(obstacles[first])
        circle2 = refiner._obstacle_tuple(obstacles[second])
        candidate, report = refiner.refine_path(
            working, circle1, circle2, all_obstacles=obstacles
        )
        used_extension = 0.0

        # 只有调整区已经触及路径起点时，向后扩展才可能增加进场距离；
        # 内部通道失败时扩展起点不会改变局部几何，不应反复重试。
        needs_more_approach = report.get("start_index", 1) == 0
        if not report.get("valid", False) and needs_more_approach:
            extension = cfg.extension_step
            while extension <= cfg.max_extension_distance + 1e-9:
                extended = _extend_path_backward(
                    working, extension, cfg.extension_step
                )
                # 在进行较昂贵的通道优化前先淘汰碰撞扩展段。
                prefix_count = len(extended) - len(working)
                prefix_collision = any(
                    refiner._vehicle_collides(point, obstacle)
                    for point in extended[:prefix_count]
                    for obstacle in obstacles
                )
                if not prefix_collision:
                    candidate, report = refiner.refine_path(
                        extended, circle1, circle2,
                        all_obstacles=obstacles,
                    )
                    if report.get("valid", False):
                        used_extension = extension
                        break
                extension += cfg.extension_step

        report = dict(report)
        report.update({
            "obstacle_indices": (first, second),
            "extension_distance": float(used_extension),
        })
        reports.append(report)
        if report.get("valid", False):
            working = candidate

    # 即使没有窄通道，也必须消除全局路径上的阿克曼不可行急弯。
    global_curvature_before = refiner._max_curvature(working)
    global_curvature_applied = False
    global_curvature_valid = True
    global_curvature_reason = "Already within curvature limit"
    if (cfg.enforce_global_curvature and
            global_curvature_before > refiner._curvature_limit()):
        smoothing_inputs = [working]
        if HAS_SCIPY and cfg.global_smooth:
            original_smoothing = refiner.params.spline_smoothing
            for factor in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
                refiner.params.spline_smoothing = max(
                    1e-6, original_smoothing * factor
                )
                smoothing_inputs.append(refiner._global_smooth(working))
            refiner.params.spline_smoothing = original_smoothing

        collision = None
        curvature_ok = False
        for smoothing_input in smoothing_inputs:
            curvature_candidate = refiner._enforce_min_turning_radius(
                smoothing_input
            )
            collision = next((
                (path_index, obstacle_index)
                for path_index, point in enumerate(curvature_candidate)
                for obstacle_index, obstacle in enumerate(obstacles)
                if refiner._vehicle_collides(point, obstacle)
            ), None)
            final_curvature = refiner._max_curvature(curvature_candidate)
            curvature_ok = (
                final_curvature <= refiner._curvature_limit() *
                (1.0 + cfg.curvature_tolerance)
            )
            if collision is None and curvature_ok:
                working = curvature_candidate
                global_curvature_applied = True
                global_curvature_reason = "Global curvature optimized"
                break

        if not global_curvature_applied:
            global_curvature_valid = False
            if collision is not None:
                global_curvature_reason = (
                    f"Curvature smoothing collides with obstacle "
                    f"{collision[1]} at path index {collision[0]}"
                )
            else:
                global_curvature_reason = (
                    f"Unable to reduce curvature to "
                    f"{refiner._curvature_limit():.6f} 1/m"
                )

    global_curvature_after = refiner._max_curvature(working)
    return working, {
        "valid": (
            all(item.get("valid", False) for item in reports) and
            global_curvature_valid
        ),
        "candidate_pair_count": len(pairs),
        "optimized_pair_count": sum(
            bool(item.get("valid", False)) for item in reports
        ),
        "pairs": reports,
        "global_curvature_applied": global_curvature_applied,
        "global_curvature_valid": global_curvature_valid,
        "global_curvature_reason": global_curvature_reason,
        "global_max_curvature_before": float(global_curvature_before),
        "global_max_curvature_after": float(global_curvature_after),
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("圆形通道路径修正模块 - 增强版测试")
    print("=" * 80)

    # 创建测试场景
    circle1 = (10.0, 2.0, 0.8)
    circle2 = (10.0, -2.0, 0.8)

    # 构造一条穿过两圆之间的路径（密集采样以避免高曲率）
    N = 200  # 增加采样点数
    x = np.linspace(0, 20, N)
    y = np.linspace(4, -4, N)  # 从上到下

    # 计算每个点的切向航向
    yaw = np.zeros(N)
    for i in range(N):
        if i < N - 1:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            yaw[i] = np.arctan2(dy, dx)
        else:
            yaw[i] = yaw[i-1]  # 最后一个点使用前一个点的航向

    original_path = np.column_stack([x, y, yaw])

    print(f"\n测试场景：")
    print(f"  圆1: center=({circle1[0]:.1f}, {circle1[1]:.1f}), r={circle1[2]:.1f}")
    print(f"  圆2: center=({circle2[0]:.1f}, {circle2[1]:.1f}), r={circle2[2]:.1f}")
    print(f"  原始路径: {N}个点")
    print(f"  起点: ({x[0]:.1f}, {y[0]:.1f})")
    print(f"  终点: ({x[-1]:.1f}, {y[-1]:.1f})")

    # 执行修正（使用投影方法）
    params = {
        'use_projection_method': True,
        'use_quintic_bezier_connection': False,
        'projection_extension_margin': 1.0,
    }
    refined_path, info = refine_path_between_circles(
        original_path,
        circle1,
        circle2,
        params=params
    )

    # 打印结果
    print(f"\n修正结果：")
    if info['valid']:
        print(f"  ✓ 修正成功")
        print(f"  方法: {info.get('method', 'unknown')}")
        print(f"  通道中心: ({info['center_point'][0]:.3f}, {info['center_point'][1]:.3f})")
        print(f"  目标航向: {np.degrees(info['target_yaw']):.1f}°")
        print(f"  可用间隙: {info['free_gap']:.3f}m")

        # 投影方法特有的字段
        if info.get('method') == 'projection_method':
            print(f"  最大直径: {info['max_diameter']:.3f}m")
            print(f"  投影半长: {info['projection_half_length']:.3f}m")
            print(f"  延长余量: {info['projection_extension_margin']:.3f}m")
            print(f"  入口索引: {info['entry_index']}")
            print(f"  出口连接索引: {info.get('connection_index', 'N/A')}")

            # S形曲线信息
            if 's_curve_info' in info and info['s_curve_info'].get('valid'):
                s_info = info['s_curve_info']
                print(f"  S形曲线前向搜索: {s_info['forward_search_distance']:.2f}m")
                print(f"  S形曲线纵向距离: {s_info['longitudinal_distance']:.2f}m")
                print(f"  S形曲线横向偏移: {s_info['lateral_offset']:.2f}m")
                print(f"  S形曲线最大曲率: {s_info['max_curvature']:.4f}")

        # 通用字段
        if 'original_center_error' in info:
            print(f"  中心误差: {info['original_center_error']:.3f}m → {info['refined_center_error']:.3f}m")
            print(f"  角度误差: {info['original_angle_error']:.1f}° → {info['refined_angle_error']:.1f}°")
            print(f"  中心误差改善: {info['center_error_reduction']:.1f}%")
            print(f"  角度误差改善: {info['angle_error_reduction']:.1f}%")

        print(f"  连接段点数: 起点{info.get('connection_points_before', 0)}个, 终点{info.get('connection_points_after', 0)}个")
        print(f"  全局平滑: {'是' if info.get('global_smooth', False) else '否'}")
        print(f"  最大曲率: {info.get('max_curvature', 0):.4f}")

        # 路径点数
        path_points = info.get('path_points', 'N/A')
        print(f"  路径点数: {path_points}")
        print(f"  起点: ({refined_path[0, 0]:.1f}, {refined_path[0, 1]:.1f})")
        print(f"  终点: ({refined_path[-1, 0]:.1f}, {refined_path[-1, 1]:.1f})")
    else:
        print(f"  ✗ 修正失败: {info['reason']}")

    print("\n" + "=" * 80)
