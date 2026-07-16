import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from scipy.optimize import minimize


# ============================================================
# Matplotlib 中文字体
# ============================================================

font_candidates = [
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name]
        print(f"Matplotlib 使用中文字体：{font_name}")
        break
else:
    print("警告：没有找到中文字体，中文可能显示为方框")

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float


@dataclass
class VehicleGeometry:
    front_length: float = 1.4
    rear_length: float = 0.6
    width: float = 1.0
    safety_margin: float = 0.12


@dataclass
class TEBConfig:
    wheel_base: float = 1.2
    max_speed: float = 1.2
    max_accel: float = 1.5
    max_steer: float = math.radians(30.0)

    min_dt: float = 0.08
    max_dt: float = 1.2
    num_poses: int = 18
    max_iterations: int = 35

    obstacle_influence: float = 1.5

    w_time: float = 0.35
    w_reference: float = 0.35
    w_smooth: float = 18.0
    w_spacing: float = 5.0
    w_obstacle: float = 180.0
    w_speed: float = 40.0
    w_accel: float = 8.0
    w_curvature: float = 70.0
    w_curvature_smooth: float = 4.0
    w_start_heading: float = 35.0
    w_goal_heading: float = 15.0
    w_short_segment: float = 100.0

    # 三条不同的初始弹性带，近似不同绕障拓扑
    topology_offsets: tuple = (0.0, 1.5, -1.5)


@dataclass
class TEBCandidate:
    poses: list[Pose]
    dt: np.ndarray
    cost: float
    feasible: bool
    minimum_clearance: float
    total_time: float
    iterations: int
    offset: float


@dataclass
class TEBResult:
    best: Optional[TEBCandidate]
    candidates: list[TEBCandidate]
    reference: list[Pose]
    initial_bands: list[list[Pose]]


# ============================================================
# 基础函数
# ============================================================

def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(angle1: float, angle2: float) -> float:
    return normalize_angle(angle1 - angle2)


def generate_global_path() -> list[Pose]:
    """生成一条平滑的 S 形全局路径。"""
    x = np.linspace(1.5, 18.3, 700)
    y = 6.0 + 1.15 * np.sin((x - 1.5) * 0.42)
    yaw = np.arctan2(np.gradient(y), np.gradient(x))

    return [
        Pose(float(px), float(py), float(pyaw))
        for px, py, pyaw in zip(x, y, yaw)
    ]


def rrt_result_to_global_path(rrt_result) -> list[Pose]:
    """
    将 Ackermann-RRT 输出转换成 TEB 全局路径。

    rrt_result:
        path_x, path_y, path_yaw, directions
    """
    path_x, path_y, path_yaw, _ = rrt_result

    return [
        Pose(float(x), float(y), float(yaw))
        for x, y, yaw in zip(path_x, path_y, path_yaw)
    ]


def get_vehicle_corners(
    pose: Pose,
    vehicle: VehicleGeometry,
) -> list[tuple[float, float]]:
    front = vehicle.front_length + vehicle.safety_margin
    rear = vehicle.rear_length + vehicle.safety_margin
    half_width = vehicle.width / 2.0 + vehicle.safety_margin

    local_corners = [
        (front, half_width),
        (front, -half_width),
        (-rear, -half_width),
        (-rear, half_width),
    ]

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    return [
        (
            pose.x + local_x * cos_yaw - local_y * sin_yaw,
            pose.y + local_x * sin_yaw + local_y * cos_yaw,
        )
        for local_x, local_y in local_corners
    ]


def circle_rectangle_clearance(
    pose: Pose,
    vehicle: VehicleGeometry,
    obstacle: CircleObstacle,
) -> float:
    """
    圆形障碍物与矩形车辆之间的间距。

    > 0：安全
    = 0：接触
    < 0：碰撞
    """
    dx = obstacle.x - pose.x
    dy = obstacle.y - pose.y

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    margin = vehicle.safety_margin

    x_min = -vehicle.rear_length - margin
    x_max = vehicle.front_length + margin
    y_min = -vehicle.width / 2.0 - margin
    y_max = vehicle.width / 2.0 + margin

    nearest_x = min(max(local_x, x_min), x_max)
    nearest_y = min(max(local_y, y_min), y_max)

    return math.hypot(
        local_x - nearest_x,
        local_y - nearest_y,
    ) - obstacle.radius


def pose_inside_bounds(
    pose: Pose,
    vehicle: VehicleGeometry,
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds

    return all(
        x_min <= x <= x_max and y_min <= y <= y_max
        for x, y in get_vehicle_corners(pose, vehicle)
    )


# ============================================================
# Ackermann TEB
# ============================================================

class AckermannTEB:
    def __init__(
        self,
        config: TEBConfig,
        vehicle: VehicleGeometry,
        bounds: tuple[float, float, float, float],
    ):
        self.config = config
        self.vehicle = vehicle
        self.bounds = bounds

    @staticmethod
    def xy_to_yaw(
        xy: np.ndarray,
        start_yaw: Optional[float] = None,
        end_yaw: Optional[float] = None,
    ) -> np.ndarray:
        delta = np.diff(xy, axis=0)
        segment_yaw = np.arctan2(delta[:, 1], delta[:, 0])

        yaw = np.concatenate([segment_yaw, [segment_yaw[-1]]])

        if start_yaw is not None:
            yaw[0] = start_yaw

        if end_yaw is not None:
            yaw[-1] = end_yaw

        return np.unwrap(yaw)

    def resample_path(
        self,
        global_path: list[Pose],
    ) -> list[Pose]:
        """将全局路径重新采样为固定数量的 TEB 节点。"""
        path_xy = np.array([
            [pose.x, pose.y]
            for pose in global_path
        ])

        segment_length = np.linalg.norm(
            np.diff(path_xy, axis=0),
            axis=1,
        )

        arc_length = np.concatenate([
            [0.0],
            np.cumsum(segment_length),
        ])

        target_arc = np.linspace(
            0.0,
            arc_length[-1],
            self.config.num_poses,
        )

        resampled_xy = np.column_stack([
            np.interp(target_arc, arc_length, path_xy[:, 0]),
            np.interp(target_arc, arc_length, path_xy[:, 1]),
        ])

        yaw = self.xy_to_yaw(
            resampled_xy,
            global_path[0].yaw,
            global_path[-1].yaw,
        )

        return [
            Pose(float(x), float(y), normalize_angle(float(angle)))
            for (x, y), angle in zip(resampled_xy, yaw)
        ]

    def make_initial_band(
        self,
        reference: list[Pose],
        offset: float,
    ) -> tuple[list[Pose], np.ndarray]:
        """
        基于全局参考路径生成初始弹性带。

        offset > 0：向路径左侧偏移
        offset < 0：向路径右侧偏移
        """
        xy = np.array([
            [pose.x, pose.y]
            for pose in reference
        ])

        if abs(offset) > 1e-10:
            progress = np.linspace(0.0, 1.0, len(xy))
            yaw = self.xy_to_yaw(xy)

            # 起点、终点偏移为零，中间偏移最大
            envelope = np.sin(math.pi * progress)

            xy[:, 0] += offset * envelope * (-np.sin(yaw))
            xy[:, 1] += offset * envelope * np.cos(yaw)

        yaw = self.xy_to_yaw(
            xy,
            reference[0].yaw,
            reference[-1].yaw,
        )

        poses = [
            Pose(float(x), float(y), normalize_angle(float(angle)))
            for (x, y), angle in zip(xy, yaw)
        ]

        segment_length = np.linalg.norm(
            np.diff(xy, axis=0),
            axis=1,
        )

        dt = np.clip(
            segment_length / (0.75 * self.config.max_speed),
            self.config.min_dt,
            self.config.max_dt,
        )

        return poses, dt

    def pack(
        self,
        xy: np.ndarray,
        dt: np.ndarray,
    ) -> np.ndarray:
        """
        优化变量：

        中间节点：
            x1, y1, x2, y2, ...

        时间间隔：
            log(dt0), log(dt1), ...
        """
        return np.concatenate([
            xy[1:-1].ravel(),
            np.log(dt),
        ])

    def unpack(
        self,
        variables: np.ndarray,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        node_count = self.config.num_poses
        internal_size = 2 * (node_count - 2)

        internal_xy = variables[:internal_size].reshape(
            node_count - 2,
            2,
        )

        xy = np.vstack([
            start_xy,
            internal_xy,
            goal_xy,
        ])

        dt = np.exp(variables[internal_size:])

        return xy, dt

    def optimizer_bounds(self):
        x_min, x_max, y_min, y_max = self.bounds
        bounds = []

        for _ in range(self.config.num_poses - 2):
            bounds.extend([
                (x_min, x_max),
                (y_min, y_max),
            ])

        log_min_dt = math.log(self.config.min_dt)
        log_max_dt = math.log(self.config.max_dt)

        bounds.extend([
            (log_min_dt, log_max_dt)
        ] * (self.config.num_poses - 1))

        return bounds

    def footprint_circle_centers(
        self,
        xy: np.ndarray,
        yaw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        优化阶段使用三个圆近似矩形车辆。

        精确碰撞检测仍然使用矩形车辆。
        """
        offsets = np.array([
            -0.55 * self.vehicle.rear_length,
            0.35 * (
                self.vehicle.front_length
                - self.vehicle.rear_length
            ),
            0.78 * self.vehicle.front_length,
        ])

        center_x = (
            xy[:, 0, None]
            + np.cos(yaw)[:, None] * offsets
        )

        center_y = (
            xy[:, 1, None]
            + np.sin(yaw)[:, None] * offsets
        )

        circle_radius = (
            self.vehicle.width / 2.0
            + self.vehicle.safety_margin
        )

        return center_x, center_y, circle_radius

    def objective(
        self,
        variables: np.ndarray,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        reference_xy: np.ndarray,
        obstacles: list[CircleObstacle],
        start_yaw: float,
        goal_yaw: float,
    ) -> float:
        config = self.config

        xy, dt = self.unpack(
            variables,
            start_xy,
            goal_xy,
        )

        delta_xy = np.diff(xy, axis=0)

        segment_length = np.linalg.norm(
            delta_xy,
            axis=1,
        )

        safe_length = np.maximum(
            segment_length,
            1e-4,
        )

        segment_yaw = np.arctan2(
            delta_xy[:, 1],
            delta_xy[:, 0],
        )

        node_yaw = np.unwrap(np.concatenate([
            segment_yaw,
            [segment_yaw[-1]],
        ]))

        cost = 0.0

        # 1. 时间代价
        cost += config.w_time * np.sum(dt)

        # 2. 偏离全局参考路径的代价
        reference_error = xy - reference_xy

        cost += config.w_reference * np.mean(
            np.sum(reference_error**2, axis=1)
        )

        # 3. 路径平滑代价
        second_difference = (
            xy[2:]
            - 2.0 * xy[1:-1]
            + xy[:-2]
        )

        cost += config.w_smooth * np.mean(
            np.sum(second_difference**2, axis=1)
        )

        # 4. 节点间距均匀性
        average_length = np.mean(safe_length)

        cost += config.w_spacing * np.mean(
            (safe_length - average_length)**2
        )

        # 防止相邻节点重合
        short_segment_error = np.maximum(
            0.06 - segment_length,
            0.0,
        )

        cost += config.w_short_segment * np.sum(
            short_segment_error**2
        )

        # 5. 速度约束
        speed = safe_length / dt

        speed_violation = np.maximum(
            speed - config.max_speed,
            0.0,
        )

        cost += config.w_speed * np.sum(
            speed_violation**2
        )

        # 6. 加速度约束
        if len(speed) > 1:
            average_dt = np.maximum(
                (dt[1:] + dt[:-1]) / 2.0,
                1e-3,
            )

            acceleration = np.diff(speed) / average_dt

            acceleration_violation = np.maximum(
                np.abs(acceleration) - config.max_accel,
                0.0,
            )

            cost += config.w_accel * np.sum(
                acceleration_violation**2
            )

        # 7. Ackermann 最小转弯半径约束
        if len(segment_yaw) > 1:
            yaw_difference = np.array([
                angle_difference(
                    segment_yaw[i + 1],
                    segment_yaw[i],
                )
                for i in range(len(segment_yaw) - 1)
            ])

            average_segment_length = np.maximum(
                (
                    safe_length[1:]
                    + safe_length[:-1]
                ) / 2.0,
                1e-3,
            )

            curvature = (
                yaw_difference
                / average_segment_length
            )

            max_curvature = (
                math.tan(config.max_steer)
                / config.wheel_base
            )

            curvature_violation = np.maximum(
                np.abs(curvature) - max_curvature,
                0.0,
            )

            cost += config.w_curvature * np.sum(
                curvature_violation**2
            )

            if len(curvature) > 1:
                cost += (
                    config.w_curvature_smooth
                    * np.sum(np.diff(curvature)**2)
                )

        # 8. 起点、终点航向
        cost += (
            config.w_start_heading
            * angle_difference(
                segment_yaw[0],
                start_yaw,
            )**2
        )

        cost += (
            config.w_goal_heading
            * angle_difference(
                segment_yaw[-1],
                goal_yaw,
            )**2
        )

        # 9. 障碍物代价
        center_x, center_y, robot_radius = (
            self.footprint_circle_centers(
                xy,
                node_yaw,
            )
        )

        for obstacle in obstacles:
            clearance = np.sqrt(
                (center_x - obstacle.x)**2
                + (center_y - obstacle.y)**2
            ) - (
                robot_radius
                + obstacle.radius
            )

            obstacle_error = np.maximum(
                config.obstacle_influence
                - clearance,
                0.0,
            )

            cost += config.w_obstacle * np.sum(
                obstacle_error**2
            )

        return float(cost)

    def exact_minimum_clearance(
        self,
        poses: list[Pose],
        obstacles: list[CircleObstacle],
    ) -> float:
        minimum_clearance = math.inf

        for pose in poses:
            if not pose_inside_bounds(
                pose,
                self.vehicle,
                self.bounds,
            ):
                return -1.0

            for obstacle in obstacles:
                clearance = circle_rectangle_clearance(
                    pose,
                    self.vehicle,
                    obstacle,
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )

        return minimum_clearance

    def optimize_candidate(
        self,
        reference: list[Pose],
        obstacles: list[CircleObstacle],
        offset: float,
    ) -> TEBCandidate:
        initial_poses, initial_dt = (
            self.make_initial_band(
                reference,
                offset,
            )
        )

        initial_xy = np.array([
            [pose.x, pose.y]
            for pose in initial_poses
        ])

        reference_xy = np.array([
            [pose.x, pose.y]
            for pose in reference
        ])

        start_xy = initial_xy[0].copy()
        goal_xy = initial_xy[-1].copy()

        initial_variables = self.pack(
            initial_xy,
            initial_dt,
        )

        optimization_result = minimize(
            self.objective,
            initial_variables,
            args=(
                start_xy,
                goal_xy,
                reference_xy,
                obstacles,
                reference[0].yaw,
                reference[-1].yaw,
            ),
            method="L-BFGS-B",
            bounds=self.optimizer_bounds(),
            options={
                "maxiter": self.config.max_iterations,
                "ftol": 1e-5,
                "maxls": 25,
            },
        )

        optimized_xy, optimized_dt = self.unpack(
            optimization_result.x,
            start_xy,
            goal_xy,
        )

        optimized_yaw = self.xy_to_yaw(
            optimized_xy,
            reference[0].yaw,
            reference[-1].yaw,
        )

        optimized_poses = [
            Pose(
                float(x),
                float(y),
                normalize_angle(float(yaw)),
            )
            for (x, y), yaw in zip(
                optimized_xy,
                optimized_yaw,
            )
        ]

        minimum_clearance = (
            self.exact_minimum_clearance(
                optimized_poses,
                obstacles,
            )
        )

        return TEBCandidate(
            poses=optimized_poses,
            dt=optimized_dt,
            cost=float(optimization_result.fun),
            feasible=minimum_clearance > 0.0,
            minimum_clearance=float(minimum_clearance),
            total_time=float(np.sum(optimized_dt)),
            iterations=int(
                getattr(
                    optimization_result,
                    "nit",
                    0,
                )
            ),
            offset=float(offset),
        )

    def plan(
        self,
        global_path: list[Pose],
        obstacles: list[CircleObstacle],
    ) -> TEBResult:
        """
        TEB 统一接口。

        输入：
            全局路径
            当前障碍物

        输出：
            最优轨迹
            所有拓扑候选
            初始弹性带
        """
        reference = self.resample_path(
            global_path
        )

        initial_bands = []
        candidates = []

        for offset in self.config.topology_offsets:
            initial_band, _ = self.make_initial_band(
                reference,
                offset,
            )

            initial_bands.append(initial_band)

            candidate = self.optimize_candidate(
                reference,
                obstacles,
                offset,
            )

            candidates.append(candidate)

        feasible_candidates = [
            candidate
            for candidate in candidates
            if candidate.feasible
        ]

        if feasible_candidates:
            best = min(
                feasible_candidates,
                key=lambda candidate: candidate.cost,
            )
        else:
            best = None

        return TEBResult(
            best=best,
            candidates=candidates,
            reference=reference,
            initial_bands=initial_bands,
        )


# ============================================================
# Matplotlib 交互测试
# ============================================================

class InteractiveTEBTest:
    def __init__(self):
        self.bounds = (
            0.0,
            21.0,
            0.0,
            12.0,
        )

        self.vehicle = VehicleGeometry(
            front_length=1.4,
            rear_length=0.6,
            width=1.0,
            safety_margin=0.12,
        )

        self.config = TEBConfig()
        self.global_path = generate_global_path()

        self.planner = AckermannTEB(
            config=self.config,
            vehicle=self.vehicle,
            bounds=self.bounds,
        )

        self.initial_obstacles = [
            CircleObstacle(
                x=8.8,
                y=6.0,
                radius=0.55,
            ),
            CircleObstacle(
                x=13.2,
                y=4.9,
                radius=0.55,
            ),
        ]

        self.obstacles = [
            CircleObstacle(
                obstacle.x,
                obstacle.y,
                obstacle.radius,
            )
            for obstacle in self.initial_obstacles
        ]

        self.dragged_obstacle_index = None

        self.fig, self.ax = plt.subplots(
            figsize=(12, 8)
        )

        self.initial_band_collection = LineCollection(
            [],
            linewidths=1.0,
            linestyles="dotted",
            alpha=0.35,
            colors="gray",
            label="初始弹性带",
        )

        self.candidate_collection = LineCollection(
            [],
            linewidths=1.5,
            alpha=0.6,
            colors="steelblue",
            label="优化候选轨迹",
        )

        self.ax.add_collection(
            self.initial_band_collection
        )

        self.ax.add_collection(
            self.candidate_collection
        )

        self.best_line, = self.ax.plot(
            [],
            [],
            linewidth=3.5,
            color="limegreen",
            label="最优 TEB 轨迹",
        )

        self.band_nodes, = self.ax.plot(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=5,
            color="black",
            label="TEB 位姿节点",
        )

        self.vehicle_patches = []
        self.obstacle_patches = []

        self.draw_scene()
        self.connect_events()
        self.replan()

    def draw_scene(self):
        path_x = [
            pose.x
            for pose in self.global_path
        ]

        path_y = [
            pose.y
            for pose in self.global_path
        ]

        self.ax.plot(
            path_x,
            path_y,
            "--",
            linewidth=2,
            color="darkorange",
            label="全局路径",
        )

        self.ax.scatter(
            path_x[0],
            path_y[0],
            s=70,
            color="green",
            label="起点",
        )

        self.ax.scatter(
            path_x[-1],
            path_y[-1],
            s=70,
            color="purple",
            label="终点",
        )

        self.refresh_obstacles()

        x_min, x_max, y_min, y_max = self.bounds

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X / m")
        self.ax.set_ylabel("Y / m")

        self.ax.set_title(
            "左键拖动障碍物，松开后重新优化；"
            "右键添加；R恢复；C清空"
        )

        self.ax.legend(loc="upper left")

    def connect_events(self):
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_press,
        )

        self.fig.canvas.mpl_connect(
            "motion_notify_event",
            self.on_mouse_motion,
        )

        self.fig.canvas.mpl_connect(
            "button_release_event",
            self.on_mouse_release,
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )

    def refresh_obstacles(self):
        for patch in self.obstacle_patches:
            patch.remove()

        self.obstacle_patches = []

        for index, obstacle in enumerate(
            self.obstacles
        ):
            patch = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                facecolor="tomato",
                edgecolor="darkred",
                alpha=0.65,
                label=(
                    "可拖动障碍物"
                    if index == 0
                    else None
                ),
            )

            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)

    def clear_vehicle_patches(self):
        for patch in self.vehicle_patches:
            patch.remove()

        self.vehicle_patches = []

    @staticmethod
    def poses_to_segment(
        poses: list[Pose],
    ) -> list[tuple[float, float]]:
        return [
            (pose.x, pose.y)
            for pose in poses
        ]

    def replan(self):
        self.ax.set_title("TEB 正在优化……")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        result = self.planner.plan(
            global_path=self.global_path,
            obstacles=self.obstacles,
        )

        initial_segments = [
            self.poses_to_segment(band)
            for band in result.initial_bands
        ]

        candidate_segments = [
            self.poses_to_segment(
                candidate.poses
            )
            for candidate in result.candidates
        ]

        self.initial_band_collection.set_segments(
            initial_segments
        )

        self.candidate_collection.set_segments(
            candidate_segments
        )

        self.clear_vehicle_patches()

        if result.best is None:
            self.best_line.set_data([], [])
            self.band_nodes.set_data([], [])

            self.ax.set_title(
                "没有找到无碰撞 TEB 轨迹，"
                "请移动障碍物"
            )

            self.fig.canvas.draw_idle()
            return

        best = result.best

        self.best_line.set_data(
            [
                pose.x
                for pose in best.poses
            ],
            [
                pose.y
                for pose in best.poses
            ],
        )

        self.band_nodes.set_data(
            [
                pose.x
                for pose in best.poses
            ],
            [
                pose.y
                for pose in best.poses
            ],
        )

        vehicle_interval = max(
            1,
            len(best.poses) // 8,
        )

        for pose in best.poses[::vehicle_interval]:
            patch = Polygon(
                get_vehicle_corners(
                    pose,
                    self.vehicle,
                ),
                closed=True,
                fill=False,
                linewidth=1,
                edgecolor="black",
                alpha=0.55,
            )

            self.ax.add_patch(patch)
            self.vehicle_patches.append(patch)

        feasible_count = sum(
            candidate.feasible
            for candidate in result.candidates
        )

        self.ax.set_title(
            f"TEB 优化完成 | "
            f"可行拓扑={feasible_count}/"
            f"{len(result.candidates)} | "
            f"代价={best.cost:.1f} | "
            f"总时间={best.total_time:.2f} s | "
            f"最小间距={best.minimum_clearance:.2f} m | "
            f"迭代={best.iterations}"
        )

        self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        # 右键添加障碍物
        if event.button == 3:
            self.obstacles.append(
                CircleObstacle(
                    x=event.xdata,
                    y=event.ydata,
                    radius=0.5,
                )
            )

            self.refresh_obstacles()
            self.replan()
            return

        if event.button != 1:
            return

        # 左键选择障碍物
        for index in reversed(
            range(len(self.obstacles))
        ):
            obstacle = self.obstacles[index]

            distance = math.hypot(
                event.xdata - obstacle.x,
                event.ydata - obstacle.y,
            )

            if distance <= obstacle.radius + 0.25:
                self.dragged_obstacle_index = index
                return

    def on_mouse_motion(self, event):
        if (
            self.dragged_obstacle_index is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        obstacle = self.obstacles[
            self.dragged_obstacle_index
        ]

        x_min, x_max, y_min, y_max = self.bounds

        obstacle.x = min(
            max(
                event.xdata,
                x_min + obstacle.radius,
            ),
            x_max - obstacle.radius,
        )

        obstacle.y = min(
            max(
                event.ydata,
                y_min + obstacle.radius,
            ),
            y_max - obstacle.radius,
        )

        self.obstacle_patches[
            self.dragged_obstacle_index
        ].center = (
            obstacle.x,
            obstacle.y,
        )

        self.ax.set_title(
            "正在移动障碍物，"
            "松开鼠标后 TEB 重新优化"
        )

        self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        if self.dragged_obstacle_index is None:
            return

        self.dragged_obstacle_index = None
        self.replan()

    def on_key_press(self, event):
        if event.key in ("r", "R"):
            self.obstacles = [
                CircleObstacle(
                    obstacle.x,
                    obstacle.y,
                    obstacle.radius,
                )
                for obstacle in self.initial_obstacles
            ]

            self.refresh_obstacles()
            self.replan()

        elif event.key in ("c", "C"):
            self.obstacles = []
            self.refresh_obstacles()
            self.replan()

        elif event.key in ("t", "T"):
            self.replan()

        elif event.key == "escape":
            plt.close(self.fig)

    def show(self):
        plt.show()


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Ackermann-TEB 交互式优化测试")
    print("=" * 70)
    print("橙色虚线：全局路径")
    print("灰色点线：三条初始弹性带")
    print("蓝色细线：三个优化候选拓扑")
    print("绿色粗线：最终最优 TEB 轨迹")
    print("黑色圆点：TEB 位姿节点")
    print("左键拖动障碍物，松开后重新优化")
    print("右键添加障碍物")
    print("R：恢复初始障碍物")
    print("C：清空所有障碍物")
    print("T：手动重新优化")
    print("Esc：退出程序")

    app = InteractiveTEBTest()
    app.show()


if __name__ == "__main__":
    main()