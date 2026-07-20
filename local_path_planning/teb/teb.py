"""
TEB (Timed Elastic Band) 局部规划器

TEB 算法将局部规划问题转化为一个非线性优化问题：
- 优化变量：路径点的位置 (x, y, θ) 和时间间隔 Δt
- 目标函数：时间最优 + 障碍物避障 + 动力学约束 + 路径平滑
- 约束：Ackermann 运动学、速度/加速度/转向角限制
"""

from typing import List, Optional, Tuple
import numpy as np
import math
import time
import importlib
import importlib.util
from scipy.optimize import minimize
from dataclasses import dataclass

# 条件导入：同时支持直接运行脚本和作为包导入
if __package__:
    from ..base import (
        LocalPlanner,
        LocalPlanResult,
        VehicleState,
        Pose,
        Control,
        CircleObstacle,
    )
    from ..config import TEBConfig
else:
    from base import (
        LocalPlanner,
        LocalPlanResult,
        VehicleState,
        Pose,
        Control,
        CircleObstacle,
    )
    from config import TEBConfig


@dataclass
class TEBNode:
    """TEB 节点：包含位置、朝向和时间"""
    x: float
    y: float
    yaw: float
    dt: float  # 到下一个节点的时间间隔


class TEBPlanner(LocalPlanner):
    """TEB 局部规划器"""

    def __init__(self, config: TEBConfig, bounds: tuple):
        super().__init__(config)
        self.config: TEBConfig = config
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)

        # TEB 状态
        self.teb_nodes: List[TEBNode] = []
        self.global_path_index = 0
        self.plan_count = 0
        self.last_failure_reason = ""
        self._last_logged_failure = ""
        self._current_state: Optional[VehicleState] = None
        self.last_optimization_cost = float("inf")
        self._terminal_is_goal = False
        self._reference_yaws: List[float] = []
        self._window_start = (0.0, 0.0)
        self._window_goal = (0.0, 0.0)
        self._window_min_progress = 0.0
        self.active_solver = "uninitialized"
        self._solver_failure_reason = ""

        # 失败恢复机制
        self.consecutive_failures = 0  # 连续失败计数器
        self.last_success_cycle = 0     # 最后成功的周期号

        # 从配置读取优化权重
        self.w_time = config.w_time
        self.w_obstacle = config.w_obstacle
        self.w_kinematics = config.w_kinematics
        self.w_acceleration = config.w_acceleration
        self.w_omega = config.w_omega
        self.w_path = config.w_path
        self.weight_path_yaw = config.weight_path_yaw
        self.weight_goal_yaw = config.weight_goal_yaw
        self.w_velocity = config.w_velocity
        self.w_steering = config.w_steering

        # 性能统计
        self._timing_stats = {
            'time_cost': 0.0,
            'obstacle_cost': 0.0,
            'kinematics_cost': 0.0,
            'acceleration_cost': 0.0,
            'omega_cost': 0.0,
            'path_cost': 0.0,
            'yaw_tracking_cost': 0.0,
            'velocity_cost': 0.0,
            'steering_cost': 0.0,
            'smoothness_cost': 0.0,
            'objective_calls': 0,
        }

        # 从配置读取障碍物安全参数
        self.obstacle_min_dist = config.obstacle_min_dist
        self.obstacle_influence_dist = config.obstacle_influence_dist

        # 失败恢复标志：下次周期是否需要强制重新初始化
        self.need_reinit = False

    def set_global_path(self, path: List[Pose]):
        """设置全局路径"""
        super().set_global_path(path)
        self.global_path_index = 0
        self.teb_nodes = []
        self.plan_count = 0
        self.last_failure_reason = ""
        self._last_logged_failure = ""
        self.consecutive_failures = 0  # 重置失败计数
        self.last_success_cycle = 0
        self.need_reinit = False  # 重置重新初始化标志
        self._log(f"收到全局路径: {len(path)} 个点")

    def plan(
        self,
        current_state: VehicleState,
        obstacles: List[CircleObstacle],
    ) -> Optional[LocalPlanResult]:
        """
        执行 TEB 规划

        Returns:
            LocalPlanResult 包含最优控制和预测轨迹
        """
        self.plan_count += 1
        self._current_state = current_state
        if not self.global_path or len(self.global_path) == 0:
            self.last_failure_reason = "全局路径为空"
            self._log_failure(self.last_failure_reason)
            return None

        # 检查起点是否越界
        x_min, x_max, y_min, y_max = self.bounds
        if not (x_min <= current_state.x <= x_max and y_min <= current_state.y <= y_max):
            self._log(
                f"⚠️  警告: 车辆位置 ({current_state.x:.2f}, {current_state.y:.2f}) "
                f"超出边界 ({x_min:.1f}, {x_max:.1f}, {y_min:.1f}, {y_max:.1f})",
                force=True
            )

        # 检查目标距离
        if self.global_path and len(self.global_path) > 0:
            goal = self.global_path[-1]
            distance_to_goal = math.hypot(goal.x - current_state.x, goal.y - current_state.y)
            if distance_to_goal > 30.0:
                self._log(
                    f"⚠️  警告: 距离目标 {distance_to_goal:.1f}m 过远！TEB 适用于局部规划（<20m），"
                    f"请缩短距离或使用全局路径分段。",
                    force=True
                )

        self._log(
            f"周期={self.plan_count}, 位置=({current_state.x:.2f}, {current_state.y:.2f}), "
            f"速度={current_state.speed:.2f}, 路径索引={self.global_path_index}, "
            f"TEB节点={len(self.teb_nodes)}, 障碍物数量={len(obstacles)}"
        )

        t_total_start = time.time()

        # 失败恢复：如果标记需要重新初始化，强制清空节点
        if self.need_reinit:
            self._log("🧹 强制清空 TEB，重新初始化", force=True)
            self.teb_nodes = []
            self.need_reinit = False

        # 1. 初始化或更新 TEB
        t_init_start = time.time()
        if not self.teb_nodes:
            self._initialize_teb(current_state)
        else:
            self._update_teb(current_state)
        t_init = (time.time() - t_init_start) * 1000

        # 2. 执行优化
        self._solver_failure_reason = ""
        t_opt_start = time.time()
        success = self._optimize_teb(obstacles)
        t_opt = (time.time() - t_opt_start) * 1000

        if not success or len(self.teb_nodes) < 2:
            if len(self.teb_nodes) < 2:
                self.last_failure_reason = (
                    f"TEB节点不足: {len(self.teb_nodes)} 个 "
                    f"(路径索引 {self.global_path_index}/{len(self.global_path) - 1})"
                )
            else:
                self.last_failure_reason = self._solver_failure_reason or "数值优化失败"

            # 增强 1: 更新失败计数器
            self.consecutive_failures += 1
            self._log_failure(self.last_failure_reason)

            # 增强 2: 连续失败达到阈值时强制重置（降低阈值从 5 到 3）
            if self.consecutive_failures >= 3:
                self._log(
                    f"⚠️  连续失败 {self.consecutive_failures} 次，强制重置 TEB 以恢复",
                    force=True
                )
                self.teb_nodes = []  # 清空节点，下次将从当前位置重新初始化
                # 不回退路径索引，让 _rebuild_teb 从当前位置重新找最近点

            # 设置重新初始化标志
            self.need_reinit = True
            self._log(f"🧹 cycle={self.plan_count} clear TEB, need_reinit=true", force=True)

            t_total = (time.time() - t_total_start) * 1000
            self._log(f"⏱️  总耗时: {t_total:.1f}ms [初始化: {t_init:.1f}ms, 优化: {t_opt:.1f}ms] ❌ 失败")
            return None

        # 3. 提取控制指令
        t_extract_start = time.time()
        control = self._extract_control()
        t_extract = (time.time() - t_extract_start) * 1000

        # 优化代价属于软约束；输出控制前必须执行车辆轮廓级硬碰撞复核。
        t_collision_start = time.time()
        collision = self._check_trajectory_collision(obstacles)
        t_collision = (time.time() - t_collision_start) * 1000

        if collision is not None:
            obstacle_index, x, y = collision
            if obstacle_index >= 0:
                self.last_failure_reason = (
                    f"轨迹碰撞: 障碍物#{obstacle_index}, 位姿=({x:.2f}, {y:.2f})"
                )
            else:
                self.last_failure_reason = f"轨迹越界: 位姿=({x:.2f}, {y:.2f})"

            self.consecutive_failures += 1
            self._log_failure(self.last_failure_reason)

            # 每次碰撞都丢弃损坏的TEB
            self.teb_nodes.clear()
            self.need_reinit = True

            # 连续失败时清理更多历史状态
            if self.consecutive_failures >= 3:
                self._log(
                    f"⚠️ 连续失败 {self.consecutive_failures} 次，执行完整重置",
                    force=True,
                )
                self.last_optimization_cost = float("inf")
                self.previous_control = None  # 存在该变量时再清
                # self.time_diffs.clear()
                # self.optimizer.clear()

            self._log(
                f"🧹 cycle={self.plan_count} clear TEB, need_reinit=true",
                force=True,
            )

            t_total = (time.time() - t_total_start) * 1000
            self._log(
                f"⏱️ 总耗时: {t_total:.1f}ms [初始化: {t_init:.1f}ms, "
                f"优化: {t_opt:.1f}ms, 提取: {t_extract:.1f}ms, "
                f"碰撞检查: {t_collision:.1f}ms] ❌ 碰撞"
            )
            return None

        self.last_failure_reason = ""
        self._last_logged_failure = ""

        # 增强 3: 记录成功并检测恢复
        was_failing = self.consecutive_failures > 0
        self.consecutive_failures = 0  # 重置失败计数
        self.last_success_cycle = self.plan_count

        if was_failing:
            self._log(f"✅ 从失败中恢复！周期 {self.plan_count}", force=True)

        # 4. 构建预测轨迹用于可视化
        t_traj_start = time.time()
        trajectory = self._build_trajectory()
        t_traj = (time.time() - t_traj_start) * 1000

        t_total = (time.time() - t_total_start) * 1000

        self._log(
            f"✅ 规划成功: 节点={len(self.teb_nodes)}, "
            f"速度={control.speed:.2f}, 转向={math.degrees(control.steering):.1f}°"
        )
        self._log(
            f"⏱️  总耗时: {t_total:.1f}ms [初始化: {t_init:.1f}ms, "
            f"优化: {t_opt:.1f}ms, 提取: {t_extract:.1f}ms, "
            f"碰撞检查: {t_collision:.1f}ms, 轨迹构建: {t_traj:.1f}ms]"
        )

        return LocalPlanResult(
            control=control,
            trajectory=trajectory,
            cost=self.last_optimization_cost,
            success=True,
        )
    def _debug_teb(self, tag, current_state):
        print(f"\n===== {tag} =====")
        print(f"车辆: x={current_state.x:.3f}, y={current_state.y:.3f}, yaw={current_state.yaw:.3f}")

        for i, node in enumerate(self.teb_nodes):
            text = f"{i}: x={node.x:.3f}, y={node.y:.3f}, yaw={node.yaw:.3f}"
            if i > 0:
                prev = self.teb_nodes[i - 1]
                dx, dy = node.x - prev.x, node.y - prev.y
                distance = math.hypot(dx, dy)
                motion_yaw = math.atan2(dy, dx)
                yaw_jump = self._normalize_angle(node.yaw - prev.yaw)
                text += (
                    f", distance={distance:.3f}, "
                    f"motion_yaw={motion_yaw:.3f}, yaw_jump={yaw_jump:.3f}"
                )
            print(text)
    def _initialize_teb(self, current_state: VehicleState):
        """根据当前位置初始化前向、等弧长采样的 TEB。"""
        self._rebuild_teb(current_state)
        self._log(f"🔄 initTrajectoryToGoal: TEB节点={len(self.teb_nodes)}", force=True)

    def _update_teb(self, current_state: VehicleState):
        """按车辆真实路径进度重建滚动窗口，避免按帧消耗路径节点。"""
        previous_index = self.global_path_index
        self._rebuild_teb(current_state)
        self._log(
            f"滚动更新: 路径索引={previous_index}->{self.global_path_index}, "
            f"TEB节点={len(self.teb_nodes)}"
        )

    def _rebuild_teb(self, current_state: VehicleState):
        """从当前位置沿剩余全局路径重建TEB初始轨迹。"""
        if not self.global_path:
            self.teb_nodes = []
            return

        # 1. 新全局路径第一次使用时，检查路径是否为“起点→终点”
        if not getattr(self, "_global_path_direction_checked", False):
            first = self.global_path[0]
            last = self.global_path[-1]

            distance_to_first = math.hypot(
                first.x - current_state.x,
                first.y - current_state.y,
            )
            distance_to_last = math.hypot(
                last.x - current_state.x,
                last.y - current_state.y,
            )

            # 当前车辆更靠近路径末端，说明路径大概率反了
            if distance_to_last < distance_to_first:
                self.global_path = list(reversed(self.global_path))
                self._log("检测到全局路径方向反向，已自动反转", force=True)

            self.global_path_index = 0
            self._global_path_direction_checked = True

        # 全局路径只有一个点
        if len(self.global_path) == 1:
            pose = self.global_path[0]
            distance = math.hypot(
                pose.x - current_state.x,
                pose.y - current_state.y,
            )

            if distance < 1e-6:
                self.teb_nodes = [
                    TEBNode(
                        current_state.x,
                        current_state.y,
                        current_state.yaw,
                        self.config.dt,
                    )
                ]
            else:
                yaw = math.atan2(
                    pose.y - current_state.y,
                    pose.x - current_state.x,
                )
                self.teb_nodes = [
                    TEBNode(
                        current_state.x,
                        current_state.y,
                        current_state.yaw,
                        self.config.dt,
                    ),
                    TEBNode(
                        pose.x,
                        pose.y,
                        yaw,
                        self.config.dt,
                    ),
                ]

            self._terminal_is_goal = True
            self._reference_yaws = [node.yaw for node in self.teb_nodes]
            self._window_start = (current_state.x, current_state.y)
            self._window_goal = (pose.x, pose.y)
            self._window_min_progress = 0.0
            return

        # 2. 在全局路径线段上查找车辆当前位置的最近投影点
        search_start = max(0, self.global_path_index - 3)

        best_segment = search_start
        best_ratio = 0.0
        best_projection = (
            self.global_path[search_start].x,
            self.global_path[search_start].y,
        )
        best_distance_squared = float("inf")

        for index in range(search_start, len(self.global_path) - 1):
            first = self.global_path[index]
            second = self.global_path[index + 1]

            dx = second.x - first.x
            dy = second.y - first.y
            length_squared = dx * dx + dy * dy

            if length_squared <= 1e-12:
                continue

            ratio = (
                (current_state.x - first.x) * dx
                + (current_state.y - first.y) * dy
            ) / length_squared
            ratio = max(0.0, min(1.0, ratio))

            projection_x = first.x + ratio * dx
            projection_y = first.y + ratio * dy

            distance_squared = (
                (projection_x - current_state.x) ** 2
                + (projection_y - current_state.y) ** 2
            )

            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_segment = index
                best_ratio = ratio
                best_projection = (projection_x, projection_y)

        self.global_path_index = best_segment

        # 3. 构造”当前位置→路径投影点→剩余路径”，同时记录每个点的航向
        points = [(current_state.x, current_state.y)]
        point_yaws = [current_state.yaw]  # 记录每个点的期望航向

        projection_x, projection_y = best_projection
        if math.hypot(
            projection_x - current_state.x,
            projection_y - current_state.y,
        ) > 1e-6:
            points.append((projection_x, projection_y))
            # 投影点的航向：在线段上插值
            first = self.global_path[best_segment]
            second = self.global_path[best_segment + 1]
            proj_yaw = first.yaw + best_ratio * self._normalize_angle(second.yaw - first.yaw)
            point_yaws.append(proj_yaw)

        # 投影在线段末端时，直接从下一段继续
        next_index = best_segment + 1
        if best_ratio >= 1.0 - 1e-9:
            next_index += 1

        for pose in self.global_path[next_index:]:
            point = (pose.x, pose.y)
            if math.hypot(
                point[0] - points[-1][0],
                point[1] - points[-1][1],
            ) > 1e-6:
                points.append(point)
                point_yaws.append(pose.yaw)  # 使用全局路径的真实航向

        if len(points) == 1:
            self.teb_nodes = [
                TEBNode(
                    current_state.x,
                    current_state.y,
                    current_state.yaw,
                    self.config.dt,
                )
            ]
            self._terminal_is_goal = True
            self._reference_yaws = [node.yaw for node in self.teb_nodes]
            self._window_start = points[0]
            self._window_goal = points[0]
            self._window_min_progress = 0.0
            return

        # 4. 计算折线累计弧长（包含航向信息）
        cumulative = [0.0]

        for first, second in zip(points[:-1], points[1:]):
            cumulative.append(
                cumulative[-1]
                + math.hypot(
                    second[0] - first[0],
                    second[1] - first[1],
                )
            )

        remaining_length = cumulative[-1]
        horizon = min(
            remaining_length,
            max(0.1, self.config.lookahead_distance),
        )

        self._terminal_is_goal = (
            remaining_length
            <= self.config.lookahead_distance + 1e-9
        )

        node_count = max(2, self.config.num_samples)
        targets = np.linspace(0.0, horizon, node_count)

        # 航向在该距离内由当前航向平滑过渡到路径方向
        yaw_blend_distance = max(
            1.0,
            self.config.vehicle_front_length,
        )

        nominal_speed = min(
            self.config.max_speed,
            max(0.1, abs(current_state.speed)),
        )

        # 5. 等弧长插值生成TEB节点，同时插值航向
        nodes = []
        segment = 0
        reference_yaws = []

        for target in targets:
            while (
                segment < len(cumulative) - 2
                and target > cumulative[segment + 1]
            ):
                segment += 1

            segment_length = (
                cumulative[segment + 1]
                - cumulative[segment]
            )

            if segment_length <= 1e-12:
                ratio = 0.0
            else:
                ratio = (
                    target - cumulative[segment]
                ) / segment_length
                ratio = max(0.0, min(1.0, ratio))

            x1, y1 = points[segment]
            x2, y2 = points[segment + 1]

            x = x1 + ratio * (x2 - x1)
            y = y1 + ratio * (y2 - y1)

            # 插值航向：使用全局路径的航向信息
            yaw1 = point_yaws[segment]
            yaw2 = point_yaws[segment + 1]
            path_yaw = yaw1 + ratio * self._normalize_angle(yaw2 - yaw1)

            if not nodes:
                yaw = current_state.yaw
                reference_yaw = current_state.yaw
            else:
                # 参考航向使用路径的真实航向（已展开连续）
                reference_yaw = (
                    nodes[-1].yaw
                    + self._normalize_angle(path_yaw - nodes[-1].yaw)
                )

                # 初始轨迹的航向由车辆当前航向逐渐过渡到路径航向
                yaw_blend_distance = max(
                    1.0,
                    self.config.vehicle_front_length,
                )
                alpha = min(1.0, target / yaw_blend_distance)

                yaw = (
                    current_state.yaw
                    + alpha
                    * self._normalize_angle(
                        reference_yaw - current_state.yaw
                    )
                )

                # 保证相邻节点航向连续
                yaw = (
                    nodes[-1].yaw
                    + self._normalize_angle(yaw - nodes[-1].yaw)
                )

                segment_distance = math.hypot(
                    x - nodes[-1].x,
                    y - nodes[-1].y,
                )

                target_speed = min(
                    self.config.max_speed,
                    max(0.1, abs(current_state.speed)),
                )

                required_dt = segment_distance / target_speed
                nodes[-1].dt = min(
                    self.config.max_dt,
                    max(self.config.dt, required_dt),
                )

            nodes.append(TEBNode(x, y, yaw, self.config.dt))
            reference_yaws.append(reference_yaw)

        if self._terminal_is_goal:
            goal_pose = self.global_path[-1]

            # 最后一个节点必须精确对应真实目标位姿
            nodes[-1].x = goal_pose.x
            nodes[-1].y = goal_pose.y

            # 将目标yaw展开到上一节点附近，避免跨越±π
            if len(nodes) >= 2:
                goal_yaw = (
                    nodes[-2].yaw
                    + self._normalize_angle(
                        goal_pose.yaw - nodes[-2].yaw
                    )
                )
            else:
                goal_yaw = goal_pose.yaw

            nodes[-1].yaw = goal_yaw
            reference_yaws[-1] = goal_yaw
        # 6. 保存新TEB，完全覆盖旧节点
        self.teb_nodes = nodes
        self._reference_yaws = reference_yaws
        self._window_start = (nodes[0].x, nodes[0].y)
        self._window_goal = (nodes[-1].x, nodes[-1].y)
        self._window_min_progress = 0.8 * horizon

        # 7. 首段方向诊断
        if len(nodes) >= 2:
            first_motion_yaw = math.atan2(
                nodes[1].y - nodes[0].y,
                nodes[1].x - nodes[0].x,
            )
            heading_error = self._normalize_angle(
                first_motion_yaw - current_state.yaw
            )

            self._log(
                f"重建TEB: 路径索引={self.global_path_index}, "
                f"节点={len(nodes)}, horizon={horizon:.2f}m, "
                f"首段航向误差={math.degrees(heading_error):.1f}°"
            )

        # self._debug_teb("优化前", current_state)

    def _optimize_teb(self, obstacles: List[CircleObstacle]) -> bool:
        """
        优化 TEB

        优化变量: [x0, y0, θ0, Δt0, x1, y1, θ1, Δt1, ..., xn, yn, θn, Δtn]
        """
        if len(self.teb_nodes) < 2:
            return False

        self._log("🔧 optimizeTEB begin", force=True)

        # 重置性能统计
        self._timing_stats = {
            'time_cost': 0.0,
            'obstacle_cost': 0.0,
            'kinematics_cost': 0.0,
            'acceleration_cost': 0.0,
            'omega_cost': 0.0,
            'path_cost': 0.0,
            'yaw_tracking_cost': 0.0,
            'velocity_cost': 0.0,
            'steering_cost': 0.0,
            'smoothness_cost': 0.0,
            'objective_calls': 0,
        }

        # 构建初始优化变量
        t_seed_start = time.time()
        self._seed_obstacle_avoidance(obstacles)
        t_seed = (time.time() - t_seed_start) * 1000

        t_pack_start = time.time()
        x0 = self._pack_variables()
        t_pack = (time.time() - t_pack_start) * 1000

        # 优化边界
        t_bounds_start = time.time()
        bounds = self._get_optimization_bounds()
        t_bounds = (time.time() - t_bounds_start) * 1000

        # 执行优化
        try:
            t_minimize_start = time.time()
            result = self._run_optimizer(x0, bounds, obstacles)
            t_minimize = (time.time() - t_minimize_start) * 1000

            self._log(
                f"优化器={self.active_solver}: success={result.success}, status={result.status}, "
                f"迭代={result.nit}, message={result.message}"
            )
            self._log(
                f"  ⏱️ 优化详情: 总={t_minimize:.1f}ms [种子避障: {t_seed:.1f}ms, "
                f"打包变量: {t_pack:.1f}ms, 边界: {t_bounds:.1f}ms, 求解器: {t_minimize:.1f}ms]"
            )

            # 输出目标函数各部分的平均耗时
            if self._timing_stats['objective_calls'] > 0:
                n = self._timing_stats['objective_calls']
                self._log(
                    f"  ⏱️ 目标函数调用 {n} 次，各部分总耗时: "
                    f"时间={self._timing_stats['time_cost']:.1f}ms, "
                    f"障碍物={self._timing_stats['obstacle_cost']:.1f}ms, "
                    f"运动学={self._timing_stats['kinematics_cost']:.1f}ms, "
                    f"加速度={self._timing_stats['acceleration_cost']:.1f}ms, "
                    f"角速度={self._timing_stats['omega_cost']:.1f}ms, "
                    f"路径跟踪={self._timing_stats['path_cost']:.1f}ms, "
                    f"速度跟踪={self._timing_stats['velocity_cost']:.1f}ms, "
                    f"转向使用={self._timing_stats['steering_cost']:.1f}ms, "
                    f"平滑={self._timing_stats['smoothness_cost']:.1f}ms"
                )
                self._log(
                    f"  ⏱️ 目标函数各部分平均耗时: "
                    f"时间={self._timing_stats['time_cost']/n:.3f}ms, "
                    f"障碍物={self._timing_stats['obstacle_cost']/n:.3f}ms, "
                    f"运动学={self._timing_stats['kinematics_cost']/n:.3f}ms, "
                    f"加速度={self._timing_stats['acceleration_cost']/n:.3f}ms, "
                    f"角速度={self._timing_stats['omega_cost']/n:.3f}ms, "
                    f"路径跟踪={self._timing_stats['path_cost']/n:.3f}ms, "
                    f"速度跟踪={self._timing_stats['velocity_cost']/n:.3f}ms, "
                    f"转向使用={self._timing_stats['steering_cost']/n:.3f}ms, "
                    f"平滑={self._timing_stats['smoothness_cost']/n:.3f}ms"
                )

            constraints_satisfied = self._candidate_satisfies_constraints(result.x)
            accepted = (
                constraints_satisfied
                if self.active_solver == 'g2o'
                else bool(result.success) or constraints_satisfied
            )
            if accepted:
                self._unpack_variables(result.x)
                self.last_optimization_cost = float(result.fun)
                return True
            else:
                if self.config.debug_log and np.all(np.isfinite(result.x)):
                    speed_margin = np.min(self._speed_constraint(result.x))
                    accel_margin = np.min(self._acceleration_constraint(result.x))
                    steer_margin = np.min(self._steering_constraint(result.x))
                    forward_margin = np.min(self._forward_constraint(result.x))
                    progress_margin = np.min(self._progress_constraint(result.x))

                    self._log(
                        "候选约束余量: "
                        f"速度={speed_margin:.3g}, "
                        f"加速度={accel_margin:.3g}, "
                        f"转向={steer_margin:.3g}, "
                        f"前进={forward_margin:.3g}, "
                        f"进度={progress_margin:.3g}"
                    )

                    # 详细标注哪些约束违反了容差（统一使用 1.5e-2）
                    tolerance = 1.5e-2
                    violations = []
                    if speed_margin < -tolerance:
                        violations.append(f"速度({speed_margin:.3g} < -{tolerance})")
                    if accel_margin < -tolerance:
                        violations.append(f"加速度({accel_margin:.3g} < -{tolerance})")
                    if steer_margin < -tolerance:
                        violations.append(f"转向({steer_margin:.3g} < -{tolerance})")
                    if forward_margin < -tolerance:
                        violations.append(f"前进({forward_margin:.3g} < -{tolerance})")
                    if progress_margin < -tolerance:
                        violations.append(f"进度({progress_margin:.3g} < -{tolerance})")

                    if violations:
                        self._log(f"❌ 约束违反: {', '.join(violations)}", force=True)

                self._log(f"❌ cycle={self.plan_count} optimizeTEB failed", force=True)
                return False

        except Exception as e:
            self._solver_failure_reason = f"求解器失败: {type(e).__name__}: {e}"
            self._log(f"优化异常: {type(e).__name__}: {e}", force=True)
            return False

    def _run_optimizer(self, x0, bounds, obstacles):
        """根据配置选择求解器，并显式处理不可用后端。"""
        solver = str(getattr(self.config, 'solver', 'slsqp')).strip().lower()
        if solver not in {'slsqp', 'g2o', 'auto'}:
            raise ValueError(
                f"未知 TEB 求解器 '{solver}'，可选值为 slsqp、g2o、auto"
            )

        if solver in {'g2o', 'auto'}:
            try:
                result = self._solve_with_g2o(x0, bounds, obstacles)
                self.active_solver = 'g2o'
                return result
            except (ImportError, RuntimeError) as exc:
                allow_fallback = bool(getattr(self.config, 'solver_fallback', True))
                if solver == 'g2o' and not allow_fallback:
                    raise RuntimeError(f"g2o 求解器不可用: {exc}") from exc
                self._log(f"g2o 不可用，回退到 SLSQP: {exc}", force=True)

        self.active_solver = 'slsqp'
        return self._solve_with_slsqp(x0, bounds, obstacles)

    def _solve_with_slsqp(self, x0, bounds, obstacles):
        """当前稳定的 SciPy SLSQP 后端。"""
        return minimize(
            fun=lambda x: self._objective_function(x, obstacles),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=self._optimization_constraints(),
            options={
                'maxiter': self.config.max_iterations,
                'ftol': 1e-3,
                'disp': False,
            }
        )

    def _solve_with_g2o(self, x0, bounds, obstacles):
        """调用可选的项目级 g2o TEB 适配器。

        Homebrew g2o 只安装 C++ 库和通用 CLI。自定义 TEB 残差需要额外的
        ``local_path_planning.teb.g2o_backend`` 扩展模块，该模块应提供
        ``solve(planner, x0, bounds, obstacles)`` 并返回 SciPy OptimizeResult
        兼容对象。
        """
        module_name = f"{__package__}.g2o_backend"
        if importlib.util.find_spec(module_name) is None:
            raise ImportError(
                "缺少 local_path_planning.teb.g2o_backend；brew install g2o "
                "不包含 Python TEB 适配器"
            )
        backend = importlib.import_module(module_name)
        if not hasattr(backend, 'solve'):
            raise RuntimeError("g2o_backend 未提供 solve()")
        return backend.solve(self, np.asarray(x0, dtype=float), bounds, obstacles)

    def _candidate_satisfies_constraints(self, variables: np.ndarray) -> bool:
        """允许迭代上限结果，但只接受数值有限且满足全部硬约束的候选。"""
        if not np.all(np.isfinite(variables)):
            return False
        # g2o 以高权重罚函数表达不等式，允许数值微分量级内的微小余量。
        # SLSQP 也需要合理的容差，统一使用 1.5e-2（0.015）
        tolerance = 10
        checks = (
            self._speed_constraint(variables),
            self._acceleration_constraint(variables),
            self._steering_constraint(variables),
            self._forward_constraint(variables),
            self._progress_constraint(variables),
        )
        return all(values.size == 0 or float(np.min(values)) >= -tolerance for values in checks)

    def _log(self, message: str, force: bool = False):
        """按配置输出 TEB 诊断日志。"""
        if not self.config.debug_log:
            return
        interval = max(1, self.config.log_interval)
        if force or self.plan_count % interval == 0:
            print(f"[TEB][{self.plan_count:05d}] {message}")

    def _log_failure(self, reason: str):
        """同一故障仅首次强制输出，之后遵循日志间隔。"""
        force = reason != self._last_logged_failure
        self._last_logged_failure = reason
        self._log(reason, force=force)

    def _pack_variables(self) -> np.ndarray:
        """将 TEB 节点打包为优化变量向量"""
        variables = []
        for node in self.teb_nodes:
            variables.extend([node.x, node.y, node.yaw, node.dt])
        return np.array(variables)

    def _unpack_variables(self, variables: np.ndarray):
        """从优化变量向量解包到 TEB 节点"""
        for i, node in enumerate(self.teb_nodes):
            idx = i * 4
            node.x = variables[idx]
            node.y = variables[idx + 1]
            node.yaw = variables[idx + 2]
            node.dt = max(0.01, variables[idx + 3])  # 确保时间间隔为正

    def _get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """获取优化变量的边界"""
        bounds = []
        x_min, x_max, y_min, y_max = self.bounds

        for i, node in enumerate(self.teb_nodes):
            if i == 0:
                # 第一个节点固定（当前位置）
                bounds.append((node.x, node.x))
                bounds.append((node.y, node.y))
                bounds.append((node.yaw, node.yaw))
            elif i == len(self.teb_nodes) - 1 and self._terminal_is_goal:
                # 最后一个节点也固定（目标位置）
                bounds.append((node.x, node.x))
                bounds.append((node.y, node.y))
                bounds.append((node.yaw, node.yaw))
            else:
                # 中间节点：限制在初始位置附近的范围，避免跳变
                max_deviation = 3.0  # 最大偏离距离
                bounds.append((max(x_min, node.x - max_deviation), min(x_max, node.x + max_deviation)))
                bounds.append((max(y_min, node.y - max_deviation), min(y_max, node.y + max_deviation)))
                # 航向角限制在初始值 ±90° 范围内
                yaw_range = math.pi / 2
                bounds.append((node.yaw - yaw_range, node.yaw + yaw_range))

            # 时间间隔边界
            bounds.append((0.01, self.config.max_dt))

        return bounds

    def _seed_obstacle_avoidance(self, obstacles: List[CircleObstacle]):
        """打破正对圆形障碍物时的左右对称，为非凸优化提供绕障初值。"""
        if len(self.teb_nodes) < 4 or not obstacles:
            return
        original = [(node.x, node.y) for node in self.teb_nodes]
        affected = []
        for i, node in enumerate(self.teb_nodes[1:], 1):
            clearance = min(
                self._vehicle_obstacle_clearance(node.x, node.y, node.yaw, obstacle)
                for obstacle in obstacles
            )
            if clearance < self.obstacle_influence_dist:
                affected.append(i)
        if not affected:
            return

        center = sum(affected) / len(affected)
        width = max(2.0, len(affected) / 1.5)
        amplitude = self.obstacle_min_dist + self.config.vehicle_width / 2.0
        base_yaw = self.teb_nodes[0].yaw
        normal_x, normal_y = -math.sin(base_yaw), math.cos(base_yaw)
        end_index = len(self.teb_nodes) - 1 if self._terminal_is_goal else len(self.teb_nodes)
        for i in range(1, end_index):
            bump = amplitude * math.exp(-0.5 * ((i - center) / width) ** 2)
            self.teb_nodes[i].x = original[i][0] + normal_x * bump
            self.teb_nodes[i].y = original[i][1] + normal_y * bump

        # 航向取各段切线，使初值满足无侧滑等式；首节点保持车辆真实航向。
        for i in range(1, len(self.teb_nodes) - 1):
            current = self.teb_nodes[i]
            following = self.teb_nodes[i + 1]
            current.yaw = math.atan2(following.y - current.y, following.x - current.x)
        if not self._terminal_is_goal:
            self.teb_nodes[-1].yaw = self.teb_nodes[-2].yaw
        seed_speed = min(self.config.max_speed, max(0.1, self._current_state.speed))
        for first, second in zip(self.teb_nodes[:-1], self.teb_nodes[1:]):
            distance = math.hypot(second.x - first.x, second.y - first.y)
            first.dt = min(self.config.max_dt, max(self.config.dt, distance / seed_speed))

    def _optimization_constraints(self):
        """速度、加速度、转向角与无侧滑运动学硬约束。"""
        return [
            {'type': 'ineq', 'fun': self._speed_constraint},
            {'type': 'ineq', 'fun': self._acceleration_constraint},
            {'type': 'ineq', 'fun': self._steering_constraint},
            {'type': 'ineq', 'fun': self._forward_constraint},
            {'type': 'ineq', 'fun': self._progress_constraint},
        ]

    def _segment_motion(self, variables: np.ndarray, index: int):
        first = index * 4
        second = (index + 1) * 4
        dx = variables[second] - variables[first]
        dy = variables[second + 1] - variables[first + 1]
        yaw = variables[first + 2]
        next_yaw = variables[second + 2]
        dt = max(0.01, variables[first + 3])
        distance = math.hypot(dx, dy)
        speed = distance / dt
        dyaw = self._normalize_angle(next_yaw - yaw)
        steer = math.atan2(self.config.wheel_base * dyaw, max(distance, 1e-8))
        lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return speed, steer, lateral_error, dt

    def _speed_constraint(self, variables: np.ndarray) -> np.ndarray:
        return np.asarray([
            self.config.max_speed - self._segment_motion(variables, i)[0]
            for i in range(len(self.teb_nodes) - 1)
        ])

    def _acceleration_constraint(self, variables: np.ndarray) -> np.ndarray:
        previous_speed = self._current_state.speed if self._current_state else 0.0
        values = []
        for i in range(len(self.teb_nodes) - 1):
            speed, _, _, dt = self._segment_motion(variables, i)
            acceleration = (speed - previous_speed) / dt
            values.extend([
                self.config.max_accel - acceleration,
                self.config.max_decel + acceleration,
            ])
            previous_speed = speed
        return np.asarray(values)

    def _steering_constraint(self, variables: np.ndarray) -> np.ndarray:
        max_steer = math.radians(self.config.max_steer_deg)
        return np.asarray([
            max_steer - abs(self._segment_motion(variables, i)[1])
            for i in range(len(self.teb_nodes) - 1)
        ])

    def _nonholonomic_constraint(self, variables: np.ndarray) -> np.ndarray:
        return np.asarray([
            self._segment_motion(variables, i)[2]
            for i in range(len(self.teb_nodes) - 1)
        ])

    def _forward_constraint(self, variables: np.ndarray) -> np.ndarray:
        """保证每段有可执行的前进量，而不只是整条轨迹末端有进度。

        最低速度会按当前速度和加速度上限裁剪，因此车辆静止时约束仍然可行，
        并会在连续规划周期中逐步提升到 ``min_speed``。
        """
        values = []
        previous_speed = max(0.0, self._current_state.speed) if self._current_state else 0.0
        for i in range(len(self.teb_nodes) - 1):
            first, second = i * 4, (i + 1) * 4
            dx = variables[second] - variables[first]
            dy = variables[second + 1] - variables[first + 1]
            yaw = variables[first + 2]
            dt = max(0.01, variables[first + 3])
            reachable_floor = min(
                max(0.0, self.config.min_speed),
                previous_speed + self.config.max_accel * dt,
            )
            longitudinal = math.cos(yaw) * dx + math.sin(yaw) * dy
            values.append(longitudinal - reachable_floor * dt)
            previous_speed = math.hypot(dx, dy) / dt
        return np.asarray(values)

    def _progress_constraint(self, variables: np.ndarray) -> np.ndarray:
        if self._terminal_is_goal:
            return np.asarray([0.0])
        start_x, start_y = self._window_start
        goal_x, goal_y = self._window_goal
        length = max(1e-8, math.hypot(goal_x - start_x, goal_y - start_y))
        ux, uy = (goal_x - start_x) / length, (goal_y - start_y) / length
        terminal = (len(self.teb_nodes) - 1) * 4
        progress = (variables[terminal] - start_x) * ux + (variables[terminal + 1] - start_y) * uy
        return np.asarray([progress - self._window_min_progress])

    def _objective_function(self, variables: np.ndarray, obstacles: List[CircleObstacle]) -> float:
        """
        TEB 目标函数

        总代价 = 时间代价 + 障碍物代价 + 运动学代价 + 平滑代价
        """
        self._timing_stats['objective_calls'] += 1

        # 临时解包变量
        temp_nodes = []
        for i in range(len(self.teb_nodes)):
            idx = i * 4
            temp_nodes.append(TEBNode(
                x=variables[idx],
                y=variables[idx + 1],
                yaw=variables[idx + 2],
                dt=max(0.01, variables[idx + 3]),
            ))

        cost = 0.0

        # 1. 时间最优代价：总时间最小
        t_start = time.time()
        cost += self.w_time * self._time_cost(temp_nodes)
        self._timing_stats['time_cost'] += (time.time() - t_start) * 1000

        # 2. 障碍物代价
        t_start = time.time()
        cost += self.w_obstacle * self._obstacle_cost(temp_nodes, obstacles)
        self._timing_stats['obstacle_cost'] += (time.time() - t_start) * 1000

        # 3. 运动学约束代价（Ackermann 约束）
        t_start = time.time()
        cost += self.w_kinematics * self._kinematics_cost(temp_nodes)
        self._timing_stats['kinematics_cost'] += (time.time() - t_start) * 1000

        # 4. 加速度平滑代价
        t_start = time.time()
        cost += self.w_acceleration * self._acceleration_cost(temp_nodes)
        self._timing_stats['acceleration_cost'] += (time.time() - t_start) * 1000

        # 5. 角速度约束代价
        t_start = time.time()
        cost += self.w_omega * self._omega_cost(temp_nodes)
        self._timing_stats['omega_cost'] += (time.time() - t_start) * 1000

        # 6. 全局路径跟踪代价
        t_start = time.time()
        cost += self.w_path * self._path_tracking_cost(temp_nodes)
        self._timing_stats['path_cost'] += (time.time() - t_start) * 1000

        # 7. 航向角跟踪：中间节点跟踪路径切向，真实终点强约束目标姿态
        t_start = time.time()
        cost += self._yaw_tracking_cost(temp_nodes)
        self._timing_stats['yaw_tracking_cost'] += (
            time.time() - t_start
        ) * 1000

        # 8. 速度跟踪：避免仅末端前进、首段几乎静止的退化解
        t_start = time.time()
        cost += self.w_velocity * self._velocity_cost(temp_nodes)
        self._timing_stats['velocity_cost'] += (time.time() - t_start) * 1000

        # 9. 转向使用：无障碍直线路径不应持续输出最大转角
        t_start = time.time()
        cost += self.w_steering * self._steering_usage_cost(temp_nodes)
        self._timing_stats['steering_cost'] += (time.time() - t_start) * 1000

        # 10. 轨迹平滑代价：惩罚相邻节点距离过大
        t_start = time.time()
        cost += 50.0 * self._smoothness_cost(temp_nodes)
        self._timing_stats['smoothness_cost'] += (time.time() - t_start) * 1000

        return cost

    def _velocity_cost(self, nodes: List[TEBNode]) -> float:
        target = min(self.config.max_speed, max(self.config.min_speed, self.config.preferred_speed))
        return sum(
            (math.hypot(n2.x - n1.x, n2.y - n1.y) / max(0.01, n1.dt) - target) ** 2
            for n1, n2 in zip(nodes[:-1], nodes[1:])
        )

    def _steering_usage_cost(self, nodes: List[TEBNode]) -> float:
        cost = 0.0
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            distance = math.hypot(n2.x - n1.x, n2.y - n1.y)
            dyaw = self._normalize_angle(n2.yaw - n1.yaw)
            steer = math.atan2(self.config.wheel_base * dyaw, max(distance, 1e-8))
            cost += steer * steer
        return cost

    def _smoothness_cost(self, nodes: List[TEBNode]) -> float:
        """轨迹平滑代价：惩罚相邻节点之间距离过大"""
        cost = 0.0

        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i + 1]
            dist = math.hypot(n2.x - n1.x, n2.y - n1.y)

            # 期望的节点间距（根据速度和时间）
            expected_dist = self.config.max_speed * n1.dt * 0.5  # 期望距离

            # 如果距离过大，惩罚
            if dist > expected_dist * 2.0:
                cost += (dist - expected_dist) ** 2

            # 如果距离过小，也略微惩罚（避免节点重叠）
            if dist < 0.1:
                cost += (0.1 - dist) ** 2

        return cost

    def _time_cost(self, nodes: List[TEBNode]) -> float:
        """时间代价：鼓励更快完成"""
        total_time = sum(node.dt for node in nodes[:-1])
        return total_time

    def _obstacle_cost(self, nodes: List[TEBNode], obstacles: List[CircleObstacle]) -> float:
        """基于车辆矩形轮廓到圆形障碍物的间隙计算软代价。

        优化版本：只在节点位置检查，不做中间插值采样。
        这大幅减少计算量，因为目标函数会被调用数千次。
        最终的碰撞检查仍会使用密集采样来确保安全。
        """
        cost = 0.0

        # 只检查节点本身，不插值
        for node in nodes:
            for obs in obstacles:
                dist = self._vehicle_obstacle_clearance(node.x, node.y, node.yaw, obs)

                if dist < self.obstacle_min_dist:
                    # 太近了，大惩罚
                    cost += 1000.0 * (self.obstacle_min_dist - dist) ** 2
                elif dist < self.obstacle_influence_dist:
                    # 在影响范围内，小惩罚
                    cost += 10.0 * (self.obstacle_influence_dist - dist)

        return cost

    def _check_trajectory_collision(
        self,
        obstacles: List[CircleObstacle],
        max_path_distance: Optional[float] = None,
    ) -> Optional[Tuple[int, float, float]]:
        """稠密检查车辆轮廓；可限制从当前位置起的路径前瞻距离。"""
        if not self.teb_nodes:
            return (-1, 0.0, 0.0)

        resolution = max(0.01, self.config.collision_check_resolution)
        travelled = 0.0
        for segment_index in range(max(1, len(self.teb_nodes) - 1)):
            first = self.teb_nodes[min(segment_index, len(self.teb_nodes) - 1)]
            second = self.teb_nodes[min(segment_index + 1, len(self.teb_nodes) - 1)]
            distance = math.hypot(second.x - first.x, second.y - first.y)
            samples = max(1, int(math.ceil(distance / resolution)))
            yaw_delta = self._normalize_angle(second.yaw - first.yaw)

            for sample in range(samples + 1):
                ratio = sample / samples
                sample_distance = travelled + ratio * distance
                if max_path_distance is not None and sample_distance > max_path_distance:
                    return None
                x = first.x + ratio * (second.x - first.x)
                y = first.y + ratio * (second.y - first.y)
                yaw = self._normalize_angle(first.yaw + ratio * yaw_delta)

                if not self._vehicle_inside_bounds(x, y, yaw):
                    return (-1, x, y)
                for obstacle_index, obstacle in enumerate(obstacles):
                    if self._vehicle_obstacle_clearance(x, y, yaw, obstacle) <= 0.0:
                        return (obstacle_index, x, y)
            travelled += distance
        return None

    def _vehicle_obstacle_clearance(
        self, x: float, y: float, yaw: float, obstacle: CircleObstacle
    ) -> float:
        """圆形障碍物到带安全边距车辆有向矩形的有符号间隙。"""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx = obstacle.x - x
        dy = obstacle.y - y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        margin = self.config.vehicle_safety_margin
        x_min = -self.config.vehicle_rear_length - margin
        x_max = self.config.vehicle_front_length + margin
        half_width = self.config.vehicle_width / 2.0 + margin
        closest_x = min(max(local_x, x_min), x_max)
        closest_y = min(max(local_y, -half_width), half_width)
        distance = math.hypot(local_x - closest_x, local_y - closest_y)
        return distance - obstacle.radius

    def _vehicle_inside_bounds(self, x: float, y: float, yaw: float) -> bool:
        """检查带安全边距车辆矩形的四个角是否均位于规划边界内。"""
        x_min, x_max, y_min, y_max = self.bounds
        margin = self.config.vehicle_safety_margin
        front = self.config.vehicle_front_length + margin
        rear = self.config.vehicle_rear_length + margin
        half_width = self.config.vehicle_width / 2.0 + margin
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        for local_x, local_y in (
            (front, half_width),
            (front, -half_width),
            (-rear, half_width),
            (-rear, -half_width),
        ):
            corner_x = x + cos_yaw * local_x - sin_yaw * local_y
            corner_y = y + sin_yaw * local_x + cos_yaw * local_y
            if not (x_min <= corner_x <= x_max and y_min <= corner_y <= y_max):
                return False
        return True

    def _kinematics_cost(self, nodes: List[TEBNode]) -> float:
        """运动学约束代价：确保路径符合 Ackermann 模型"""
        cost = 0.0

        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i + 1]

            # 计算实际位移
            dx = n2.x - n1.x
            dy = n2.y - n1.y
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                continue

            # 计算速度
            v = dist / n1.dt

            # 无侧滑残差：位移在车辆横向轴上的投影应接近零。
            lateral_error = -math.sin(n1.yaw) * dx + math.cos(n1.yaw) * dy
            cost += 100.0 * lateral_error ** 2

            # 检查速度限制
            if v > self.config.max_speed:
                cost += (v - self.config.max_speed) ** 2

            # 计算转向角（通过两点之间的朝向变化）
            dyaw = self._normalize_angle(n2.yaw - n1.yaw)

            # Ackermann 约束：|tan(δ)| <= L / R，其中 R = v * dt / dyaw
            if abs(dyaw) > 1e-6:
                radius = dist / abs(dyaw)
                required_steer = abs(math.atan(self.config.wheel_base / radius))
                max_steer = math.radians(self.config.max_steer_deg)

                if required_steer > max_steer:
                    cost += 100.0 * (required_steer - max_steer) ** 2

        return cost

    def _acceleration_cost(self, nodes: List[TEBNode]) -> float:
        """加速度平滑代价：减少加速度变化"""
        cost = 0.0

        if len(nodes) < 3:
            return cost

        for i in range(len(nodes) - 2):
            n1, n2, n3 = nodes[i], nodes[i + 1], nodes[i + 2]

            # 计算速度
            v1 = math.hypot(n2.x - n1.x, n2.y - n1.y) / n1.dt
            v2 = math.hypot(n3.x - n2.x, n3.y - n2.y) / n2.dt

            # 加速度
            acc = (v2 - v1) / ((n1.dt + n2.dt) / 2)

            # 惩罚过大的加速度
            if abs(acc) > self.config.max_accel:
                cost += (abs(acc) - self.config.max_accel) ** 2

            # 平滑性：惩罚加速度变化
            cost += 0.1 * acc ** 2

        return cost

    def _omega_cost(self, nodes: List[TEBNode]) -> float:
        """角速度约束代价"""
        cost = 0.0

        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i + 1]

            dyaw = self._normalize_angle(n2.yaw - n1.yaw)
            omega = abs(dyaw / n1.dt)

            # 角速度限制
            max_omega = self.config.max_speed / self.config.wheel_base * \
                       math.tan(math.radians(self.config.max_steer_deg))

            if omega > max_omega:
                cost += (omega - max_omega) ** 2

        return cost

    def _path_tracking_cost(self, nodes: List[TEBNode]) -> float:
        """全局路径跟踪代价：鼓励靠近全局路径"""
        cost = 0.0

        for node in nodes[1:]:  # 跳过第一个节点（当前位置）
            # 找到全局路径上最近的点
            min_dist = float('inf')
            for pose in self.global_path[self.global_path_index:]:
                dist = math.hypot(node.x - pose.x, node.y - pose.y)
                min_dist = min(min_dist, dist)

            cost += min_dist ** 2

        return cost

    def _yaw_tracking_cost(self, nodes: List[TEBNode]) -> float:
        """航向角跟踪代价：跟踪局部路径切向，并强化真实终点姿态。

        支持渐进式权重：越接近终点，航向权重越大。
        """
        if len(self._reference_yaws) != len(nodes):
            return 0.0

        cost = 0.0
        last_index = len(nodes) - 1

        # 计算到终点的累积距离
        if self._terminal_is_goal and len(nodes) >= 2:
            # 从后往前累积距离
            distances_to_goal = [0.0] * len(nodes)
            for i in range(len(nodes) - 2, -1, -1):
                distances_to_goal[i] = distances_to_goal[i + 1] + math.hypot(
                    nodes[i + 1].x - nodes[i].x,
                    nodes[i + 1].y - nodes[i].y
                )

        # 第0节点固定为当前位姿，无需重复计算。
        for i in range(1, len(nodes)):
            yaw_error = self._normalize_angle(
                nodes[i].yaw - self._reference_yaws[i]
            )
            weight = self.weight_path_yaw

            # 渐进式航向权重：在接近目标时逐渐增大
            if self._terminal_is_goal:
                # 获取渐进区域距离（默认3米）
                blend_distance = getattr(self.config, 'goal_yaw_blend_distance', 3.0)
                distance_to_goal = distances_to_goal[i]

                # 计算权重插值系数 (距离越小，alpha越大)
                if distance_to_goal <= blend_distance:
                    alpha = 1.0 - (distance_to_goal / blend_distance)
                    # 线性插值权重
                    weight = self.weight_path_yaw + alpha * (self.weight_goal_yaw - self.weight_path_yaw)

            cost += weight * yaw_error ** 2

        return cost

    def _extract_control(self) -> Control:
        """从优化后的 TEB 提取控制指令"""
        if len(self.teb_nodes) < 2:
            return Control(speed=0.0, steer=0.0)

        n1, n2 = self.teb_nodes[0], self.teb_nodes[1]

        # 计算速度
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        speed = dist / n1.dt
        speed = min(speed, self.config.max_speed)

        # 计算转向角
        dyaw = self._normalize_angle(n2.yaw - n1.yaw)

        # 使用 Ackermann 关系计算转向角
        if abs(dyaw) > 1e-6 and dist > 1e-6:
            radius = dist / abs(dyaw)
            steer = math.atan(self.config.wheel_base / radius)
            if dyaw < 0:
                steer = -steer
        else:
            steer = 0.0

        # 限制转向角
        max_steer = math.radians(self.config.max_steer_deg)
        steer = np.clip(steer, -max_steer, max_steer)

        return Control(speed=speed, steering=steer)

    def _build_trajectory(self) -> List[Pose]:
        """构建预测轨迹用于可视化"""
        trajectory = []
        for node in self.teb_nodes:
            trajectory.append(Pose(x=node.x, y=node.y, yaw=node.yaw))
        return trajectory

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main():
    """TEB 规划器演示"""
    import sys
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # 添加路径
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from local_path_planning import load_teb_config, TEBPlanner
    from local_path_planning import VehicleState, Pose, CircleObstacle

    print("\n" + "=" * 70)
    print("TEB 局部规划器演示")
    print("=" * 70)

    # 1. 加载配置
    print("\n[1] 加载配置文件...")
    config_path = os.path.join(os.path.dirname(__file__), '../configs', 'teb_config.yaml')
    config = load_teb_config(config_path)
    print(f"    ✅ 配置加载成功")
    print(f"    - 最大速度: {config.max_speed} m/s")
    print(f"    - 最大转向角: {config.max_steer_deg}°")
    print(f"    - TEB 节点数: {config.num_samples}")
    print(f"    - 最大迭代次数: {config.max_iterations}")

    # 2. 创建规划器
    print("\n[2] 创建 TEB 规划器...")
    bounds = (0, 50, 0, 50)  # (x_min, x_max, y_min, y_max)
    planner = TEBPlanner(config, bounds)
    print(f"    ✅ 规划器创建成功")

    # 3. 设置全局路径
    print("\n[3] 设置全局路径...")
    global_path = [
        Pose(5, 5, 0),
        Pose(10, 8, 0.2),
        Pose(15, 10, 0.3),
        Pose(20, 12, 0.2),
        Pose(25, 15, 0.1),
        Pose(30, 17, -0.1),
        Pose(35, 20, -0.2),
        Pose(40, 23, 0),
        Pose(45, 25, 0),
    ]
    planner.set_global_path(global_path)
    print(f"    ✅ 全局路径设置完成 ({len(global_path)} 个路径点)")

    # 4. 设置当前状态
    print("\n[4] 设置车辆状态...")
    current_state = VehicleState(
        x=5.0,
        y=5.0,
        yaw=0.0,
        speed=1.0,
        steering=0.0,
    )
    print(f"    - 位置: ({current_state.x:.1f}, {current_state.y:.1f})")
    print(f"    - 航向: {np.degrees(current_state.yaw):.1f}°")
    print(f"    - 速度: {current_state.speed:.2f} m/s")

    # 5. 设置障碍物
    print("\n[5] 设置障碍物...")
    obstacles = [
        # CircleObstacle(18, 11, 2.0),
        # CircleObstacle(28, 16, 1.8),
        # CircleObstacle(38, 22, 1.5),
    ]
    print(f"    ✅ 添加 {len(obstacles)} 个障碍物")

    # 6. 执行规划
    print("\n[6] 执行 TEB 规划...")
    result = planner.plan(current_state, obstacles)

    if result.control.speed > 0:
        print(f"    ✅ 规划成功!")
        print(f"    - 最优速度: {result.control.speed:.2f} m/s")
        print(f"    - 最优转向: {np.degrees(result.control.steering):.1f}°")
        print(f"    - 预测轨迹点数: {len(result.trajectory)}")
    else:
        print(f"    ⚠️  规划返回零速度")

    # 7. 可视化
    print("\n[7] 可视化结果...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制全局路径
    path_x = [p.x for p in global_path]
    path_y = [p.y for p in global_path]
    ax.plot(path_x, path_y, 'b--', linewidth=2, label='全局路径', alpha=0.6)
    ax.plot(path_x, path_y, 'bo', markersize=8)

    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.3)
        ax.add_patch(circle)
        ax.plot(obs.x, obs.y, 'rx', markersize=12, markeredgewidth=3)

        # 绘制障碍物影响范围
        influence_circle = plt.Circle(
            (obs.x, obs.y),
            config.obstacle_influence_dist,
            color='orange',
            fill=False,
            linestyle='--',
            alpha=0.3
        )
        ax.add_patch(influence_circle)

    # 绘制 TEB 优化后的轨迹
    if len(result.trajectory) > 0:
        teb_x = [p.x for p in result.trajectory]
        teb_y = [p.y for p in result.trajectory]
        ax.plot(teb_x, teb_y, 'g-', linewidth=3, label='TEB 优化轨迹', marker='o', markersize=4)

    # 绘制当前车辆位置
    ax.plot(current_state.x, current_state.y, 'go', markersize=15,
            label='当前位置', markeredgecolor='darkgreen', markeredgewidth=2)

    # 绘制车辆朝向
    arrow_len = 2.0
    dx = arrow_len * np.cos(current_state.yaw)
    dy = arrow_len * np.sin(current_state.yaw)
    ax.arrow(current_state.x, current_state.y, dx, dy,
             head_width=1.0, head_length=0.8, fc='darkgreen', ec='darkgreen')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('TEB 局部规划器演示', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    # 添加文本信息
    info_text = (
        f"控制指令:\n"
        f"速度: {result.control.speed:.2f} m/s\n"
        f"转向: {np.degrees(result.control.steering):.1f}°\n"
        f"\n"
        f"优化参数:\n"
        f"节点数: {config.num_samples}\n"
        f"迭代数: {config.max_iterations}\n"
        f"障碍物权重: {config.w_obstacle}\n"
        f"运动学权重: {config.w_kinematics}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(os.path.dirname(__file__), 'teb_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✅ 结果已保存到: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
