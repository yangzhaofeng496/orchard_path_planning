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
from scipy.optimize import minimize
from dataclasses import dataclass

# 条件导入：同时支持直接运行脚本和作为包导入
if __package__:
    from .base import (
        LocalPlanner,
        LocalPlanResult,
        VehicleState,
        Pose,
        Control,
        CircleObstacle,
    )
    from .config import TEBConfig
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
        self._window_start = (0.0, 0.0)
        self._window_goal = (0.0, 0.0)
        self._window_min_progress = 0.0

        # 从配置读取优化权重
        self.w_time = config.w_time
        self.w_obstacle = config.w_obstacle
        self.w_kinematics = config.w_kinematics
        self.w_acceleration = config.w_acceleration
        self.w_omega = config.w_omega
        self.w_path = config.w_path

        # 从配置读取障碍物安全参数
        self.obstacle_min_dist = config.obstacle_min_dist
        self.obstacle_influence_dist = config.obstacle_influence_dist

    def set_global_path(self, path: List[Pose]):
        """设置全局路径"""
        super().set_global_path(path)
        self.global_path_index = 0
        self.teb_nodes = []
        self.plan_count = 0
        self.last_failure_reason = ""
        self._last_logged_failure = ""
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

        self._log(
            f"周期={self.plan_count}, 位置=({current_state.x:.2f}, {current_state.y:.2f}), "
            f"速度={current_state.speed:.2f}, 路径索引={self.global_path_index}, "
            f"TEB节点={len(self.teb_nodes)}"
        )

        # 1. 初始化或更新 TEB
        if not self.teb_nodes:
            self._initialize_teb(current_state)
        else:
            self._update_teb(current_state)

        # 2. 执行优化
        success = self._optimize_teb(obstacles)

        if not success or len(self.teb_nodes) < 2:
            if len(self.teb_nodes) < 2:
                self.last_failure_reason = (
                    f"TEB节点不足: {len(self.teb_nodes)} 个 "
                    f"(路径索引 {self.global_path_index}/{len(self.global_path) - 1})"
                )
            else:
                self.last_failure_reason = "数值优化失败"
            self._log_failure(self.last_failure_reason)
            return None

        # 3. 提取控制指令
        control = self._extract_control()

        # 优化代价属于软约束；输出控制前必须执行车辆轮廓级硬碰撞复核。
        collision = self._check_trajectory_collision(obstacles)
        if collision is not None:
            obstacle_index, x, y = collision
            if obstacle_index >= 0:
                self.last_failure_reason = (
                    f"轨迹碰撞: 障碍物#{obstacle_index}, 位姿=({x:.2f}, {y:.2f})"
                )
            else:
                self.last_failure_reason = f"轨迹越界: 位姿=({x:.2f}, {y:.2f})"
            self._log_failure(self.last_failure_reason)
            return None

        self.last_failure_reason = ""
        self._last_logged_failure = ""
        self._log(
            f"规划成功: 节点={len(self.teb_nodes)}, "
            f"速度={control.speed:.2f}, 转向={math.degrees(control.steering):.1f}°"
        )

        # 4. 构建预测轨迹用于可视化
        trajectory = self._build_trajectory()

        return LocalPlanResult(
            control=control,
            trajectory=trajectory,
            cost=self.last_optimization_cost,
            success=True,
        )

    def _initialize_teb(self, current_state: VehicleState):
        """根据当前位置初始化前向、等弧长采样的 TEB。"""
        self._rebuild_teb(current_state)
        self._log(f"初始化完成: TEB节点={len(self.teb_nodes)}", force=True)

    def _update_teb(self, current_state: VehicleState):
        """按车辆真实路径进度重建滚动窗口，避免按帧消耗路径节点。"""
        previous_index = self.global_path_index
        self._rebuild_teb(current_state)
        self._log(
            f"滚动更新: 路径索引={previous_index}->{self.global_path_index}, "
            f"TEB节点={len(self.teb_nodes)}"
        )

    def _rebuild_teb(self, current_state: VehicleState):
        """从当前位置沿剩余全局折线等弧长插值出固定数量的节点。"""
        search_start = max(0, self.global_path_index - 1)
        closest_idx = min(
            range(search_start, len(self.global_path)),
            key=lambda i: math.hypot(
                self.global_path[i].x - current_state.x,
                self.global_path[i].y - current_state.y,
            ),
        )
        self.global_path_index = max(self.global_path_index, closest_idx)

        points = [(current_state.x, current_state.y)]
        for pose in self.global_path[self.global_path_index:]:
            if math.hypot(pose.x - points[-1][0], pose.y - points[-1][1]) > 1e-6:
                points.append((pose.x, pose.y))

        if len(points) == 1:
            self.teb_nodes = [
                TEBNode(current_state.x, current_state.y, current_state.yaw, self.config.dt)
            ]
            return

        cumulative = [0.0]
        for first, second in zip(points[:-1], points[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
            )

        node_count = max(2, self.config.num_samples)
        horizon = min(cumulative[-1], max(0.1, self.config.lookahead_distance))
        self._terminal_is_goal = cumulative[-1] <= self.config.lookahead_distance + 1e-9
        targets = np.linspace(0.0, horizon, node_count)
        nodes = []
        segment = 0
        for target in targets:
            while segment < len(cumulative) - 2 and target > cumulative[segment + 1]:
                segment += 1
            length = cumulative[segment + 1] - cumulative[segment]
            ratio = 0.0 if length <= 1e-12 else (target - cumulative[segment]) / length
            x1, y1 = points[segment]
            x2, y2 = points[segment + 1]
            x = x1 + ratio * (x2 - x1)
            y = y1 + ratio * (y2 - y1)
            yaw = current_state.yaw if target <= 1e-12 else math.atan2(y2 - y1, x2 - x1)
            if nodes:
                segment_distance = math.hypot(x - nodes[-1].x, y - nodes[-1].y)
                # 从当前速度逐段爬升，给 SLSQP 一个满足首段加速度约束的初值。
                target_speed = min(self.config.max_speed, max(0.1, current_state.speed))
                dt = max(self.config.dt, segment_distance / target_speed)
                nodes[-1].dt = min(dt, self.config.max_dt)
            else:
                dt = self.config.dt
            nodes.append(TEBNode(x, y, yaw, self.config.dt))
        self.teb_nodes = nodes
        self._window_start = (nodes[0].x, nodes[0].y)
        self._window_goal = (nodes[-1].x, nodes[-1].y)
        self._window_min_progress = 0.8 * horizon

    def _optimize_teb(self, obstacles: List[CircleObstacle]) -> bool:
        """
        优化 TEB

        优化变量: [x0, y0, θ0, Δt0, x1, y1, θ1, Δt1, ..., xn, yn, θn, Δtn]
        """
        if len(self.teb_nodes) < 2:
            return False

        # 构建初始优化变量
        self._seed_obstacle_avoidance(obstacles)
        x0 = self._pack_variables()

        # 优化边界
        bounds = self._get_optimization_bounds()

        # 执行优化
        try:
            result = minimize(
                fun=lambda x: self._objective_function(x, obstacles),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=self._optimization_constraints(),
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': 1e-4,
                    'disp': False,
                }
            )

            self._log(
                f"优化器: success={result.success}, status={result.status}, "
                f"迭代={result.nit}, message={result.message}"
            )
            if result.success or self._candidate_satisfies_constraints(result.x):
                self._unpack_variables(result.x)
                self.last_optimization_cost = float(result.fun)
                return True
            else:
                if self.config.debug_log and np.all(np.isfinite(result.x)):
                    self._log(
                        "候选约束余量: "
                        f"速度={np.min(self._speed_constraint(result.x)):.3g}, "
                        f"加速度={np.min(self._acceleration_constraint(result.x)):.3g}, "
                        f"转向={np.min(self._steering_constraint(result.x)):.3g}, "
                        f"前进={np.min(self._forward_constraint(result.x)):.3g}, "
                        f"进度={np.min(self._progress_constraint(result.x)):.3g}"
                    )
                return False

        except Exception as e:
            self._log(f"优化异常: {type(e).__name__}: {e}", force=True)
            return False

    def _candidate_satisfies_constraints(self, variables: np.ndarray) -> bool:
        """允许迭代上限结果，但只接受数值有限且满足全部硬约束的候选。"""
        if not np.all(np.isfinite(variables)):
            return False
        tolerance = 1e-4
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
        values = []
        for i in range(len(self.teb_nodes) - 1):
            first, second = i * 4, (i + 1) * 4
            dx = variables[second] - variables[first]
            dy = variables[second + 1] - variables[first + 1]
            yaw = variables[first + 2]
            values.append(math.cos(yaw) * dx + math.sin(yaw) * dy)
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
        cost += self.w_time * self._time_cost(temp_nodes)

        # 2. 障碍物代价
        cost += self.w_obstacle * self._obstacle_cost(temp_nodes, obstacles)

        # 3. 运动学约束代价（Ackermann 约束）
        cost += self.w_kinematics * self._kinematics_cost(temp_nodes)

        # 4. 加速度平滑代价
        cost += self.w_acceleration * self._acceleration_cost(temp_nodes)

        # 5. 角速度约束代价
        cost += self.w_omega * self._omega_cost(temp_nodes)

        # 6. 全局路径跟踪代价
        cost += self.w_path * self._path_tracking_cost(temp_nodes)

        # 7. 轨迹平滑代价：惩罚相邻节点距离过大
        cost += 50.0 * self._smoothness_cost(temp_nodes)

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
        """基于车辆矩形轮廓到圆形障碍物的间隙计算软代价。"""
        cost = 0.0
        samples = list(nodes)
        resolution = max(0.05, self.config.collision_check_resolution)
        for first, second in zip(nodes[:-1], nodes[1:]):
            distance = math.hypot(second.x - first.x, second.y - first.y)
            count = max(1, int(math.ceil(distance / resolution)))
            yaw_delta = self._normalize_angle(second.yaw - first.yaw)
            for index in range(1, count):
                ratio = index / count
                samples.append(TEBNode(
                    first.x + ratio * (second.x - first.x),
                    first.y + ratio * (second.y - first.y),
                    self._normalize_angle(first.yaw + ratio * yaw_delta),
                    first.dt,
                ))

        for node in samples:
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
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'teb_config.yaml')
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
        CircleObstacle(18, 11, 2.0),
        CircleObstacle(28, 16, 1.8),
        CircleObstacle(38, 22, 1.5),
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
