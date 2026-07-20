"""
OAG-HRRT*-DWA interactive orchard planner.

OAG-HRRT*-DWA:
    Orchard Adaptive Guidance Hybrid RRT* with Dynamic Window Approach.

Workflow:
    1. Load a saved orchard npz map.
    2. Left mouse press/release sets the initial pose.
    3. OAG-HRRT* computes a non-reversing Ackermann global path.
    4. (可选) 路径优化器优化全局路径，减少冗余节点
    5. Full Ackermann DWA tracks the path while avoiding static and dynamic
       circular obstacles.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass

import yaml  # 🆕 导入 yaml 用于配置文件
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
INNOVATION_DIR = os.path.join(
    PROJECT_ROOT, "global_path_planning", "innovation_sample"
)
DEFAULT_MAP_PATH = os.path.join(CURRENT_DIR, "orchard_scene.npz")

for path in (INNOVATION_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from ackermann_rrt_star import AckermannRRTStar
from hybrid_sampler import SamplingCorridor
from orchard_environment import load_environment

# 🆕 使用新的局部规划接口
from local_path_planning import (
    DWAPlanner,
    DWAConfig,
    TEBPlanner,
    TEBConfig,
    VehicleState,
    CircleObstacle,
    Pose,
)
from local_path_planning.adaptive_teb_dwa import (
    AdaptivePlannerConfig,
    AdaptiveTEBDWAPlanner,
    AdaptiveWindowConfig,
    FeedbackConfig,
)

# 导入车辆几何工具函数
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'vehicle'))
from ackerman_dwa import VehicleGeometry, get_vehicle_corners


# 🆕 导入路径优化器
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "path_optimizer")))
from path_optimizer import CurvatureSmoother, ShortcutOptimizer


@dataclass
class MovingObstacle:
    x: float
    y: float
    radius: float
    speed: float
    heading: float

    def as_circle(self) -> CircleObstacle:
        return CircleObstacle(self.x, self.y, self.radius)


# 🆕 配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yaml")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典，如果文件不存在则返回 None
    """
    if not os.path.exists(config_path):
        print(f"[配置] 未找到配置文件: {config_path}")
        print("[配置] 使用默认配置")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"[配置] 已加载配置文件: {config_path}")
    return config


class PathCollisionChecker:
    """
    用于路径优化的碰撞检测器

    检查两点之间的直线路径是否与障碍物碰撞
    """

    def __init__(self, vehicle, obstacles, resolution=0.1):
        """
        Args:
            vehicle: VehicleGeometry 车辆几何参数
            obstacles: List[CircleObstacle] 障碍物列表
            resolution: 碰撞检测采样分辨率（米）
        """
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.resolution = resolution

    def check_line(self, p1, p2):
        """
        检查从 p1 到 p2 的直线路径是否无碰撞

        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)

        Returns:
            True 表示无碰撞，False 表示有碰撞
        """
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_samples = max(2, int(distance / self.resolution))

        # 计算路径方向
        yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # 沿着直线采样检查碰撞
        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            pose = Pose(x, y, yaw)

            # 检查该位姿是否与障碍物碰撞
            from vehicle.vehicle_collision import check_pose_collision
            if check_pose_collision(pose, self.vehicle, self.obstacles):
                return False  # 有碰撞

        return True  # 无碰撞


def optimize_global_path(
    result,
    vehicle,
    obstacles,
    config=None,
    verbose=True,
):
    """
    优化 RRT 生成的全局路径

    使用 Path Shortcut 算法减少路径中的冗余节点

    Args:
        result: RRT 规划结果 (path_x, path_y, path_yaw, directions)
        vehicle: VehicleGeometry 车辆几何参数
        obstacles: List[CircleObstacle] 障碍物列表
        config: 优化配置字典，如果为 None 则使用默认参数
        verbose: 是否打印详细信息

    Returns:
        优化后的路径，格式与输入相同
    """
    if result is None:
        return None

    # 解析优化配置
    if config is None:
        # 默认配置
        opt_config = {
            'max_iterations': 100,
            'min_points_distance': 0.2,
            'enable_angle_filter': True,
            'angle_threshold': 15.0,
            'random_seed': 42,
            'verbose': verbose,
            'collision_check_resolution': 0.1,
            'curvature_smoothing': {},
        }
    else:
        opt_config = {
            'max_iterations': config.get('max_iterations', 100),
            'min_points_distance': config.get('min_points_distance', 0.2),
            'enable_angle_filter': config.get('enable_angle_filter', True),
            'angle_threshold': math.radians(config.get('angle_threshold', 15.0)),
            'random_seed': config.get('random_seed', 42),
            'verbose': config.get('verbose', verbose),
            'collision_check_resolution': config.get('collision_check_resolution', 0.1),
            'curvature_smoothing': config.get('curvature_smoothing', {}),
        }

    # 1. 将 RRT 路径转换为点列表
    path_x, path_y, path_yaw, directions = result
    path_points = list(zip(path_x, path_y))

    original_count = len(path_points)

    if verbose:
        print(f"\n{'='*70}")
        print(f"[路径优化] 开始优化全局路径...")
        print(f"[路径优化] 原始路径节点数: {original_count}")

    # 2. 创建碰撞检测器
    checker = PathCollisionChecker(
        vehicle=vehicle,
        obstacles=obstacles,
        resolution=opt_config['collision_check_resolution'],
    )

    # 3. 创建优化器
    optimizer = ShortcutOptimizer(
        collision_checker=checker,
        max_iterations=opt_config['max_iterations'],
        min_points_distance=opt_config['min_points_distance'],
        enable_angle_filter=opt_config['enable_angle_filter'],
        angle_threshold=opt_config['angle_threshold'],
        random_seed=opt_config['random_seed'],
        verbose=opt_config['verbose'],
    )

    # 4. 执行优化
    optimized_points = optimizer.optimize(path_points)

    # 4.5 自动检测折线拐角，执行曲率约束插补并生成速度前瞻曲线。
    smooth_cfg = opt_config['curvature_smoothing']
    if smooth_cfg.get('enabled', True) and len(optimized_points) >= 3:
        max_curvature = smooth_cfg.get(
            'max_curvature',
            math.tan(math.radians(30.0)) / max(vehicle.front_length, 1e-6),
        )
        smoother = CurvatureSmoother(
            collision_checker=checker,
            max_curvature=max_curvature,
            interpolation_spacing=smooth_cfg.get('interpolation_spacing', 0.20),
            corner_angle_threshold=math.radians(
                smooth_cfg.get('corner_angle_threshold_deg', 8.0)
            ),
            corner_blend_distance=smooth_cfg.get('corner_blend_distance', 2.0),
            max_lateral_accel=smooth_cfg.get('max_lateral_accel', 1.2),
            max_speed=smooth_cfg.get('max_speed', 2.0),
            max_accel=smooth_cfg.get('max_accel', 1.0),
            max_decel=smooth_cfg.get('max_decel', 1.0),
            lookahead_distance=smooth_cfg.get('lookahead_distance', 5.0),
            start_speed=smooth_cfg.get('start_speed', 0.0),
            end_speed=smooth_cfg.get('end_speed', 0.0),
            verbose=opt_config['verbose'],
        )
        smooth_result = smoother.smooth(optimized_points)
        optimized_points = smooth_result.points
        print(
            f"[曲率平滑] 拐角={len(smooth_result.corner_indices)}, "
            f"插补点={len(smooth_result.points)}, "
            f"最大曲率={max(map(abs, smooth_result.curvatures), default=0.0):.4f} 1/m, "
            f"最高前瞻速度={max(smooth_result.speeds, default=0.0):.2f} m/s"
        )

    optimized_count = len(optimized_points)

    if verbose:
        print(f"[路径优化] 优化后路径节点数: {optimized_count}")
        print(f"[路径优化] 减少节点数: {original_count - optimized_count}")
        print(f"[路径优化] 减少比例: {(1 - optimized_count/original_count)*100:.2f}%")
        print(f"{'='*70}\n")

        # 打印详细统计
        optimizer.print_stats()

    # 5. 转换回 RRT 路径格式
    opt_x = [p[0] for p in optimized_points]
    opt_y = [p[1] for p in optimized_points]

    # 重新计算 yaw
    opt_yaw = []
    for i in range(len(optimized_points)):
        if i < len(optimized_points) - 1:
            yaw = math.atan2(
                optimized_points[i+1][1] - optimized_points[i][1],
                optimized_points[i+1][0] - optimized_points[i][0]
            )
        else:
            yaw = opt_yaw[-1] if opt_yaw else 0.0
        opt_yaw.append(yaw)

    # 保持原始方向信息（简化处理，假设优化后路径保持前进）
    opt_directions = [1] * len(optimized_points)

    return (opt_x, opt_y, opt_yaw, opt_directions)


def circle_overlaps_static(x, y, radius, static_obstacles, clearance):
    for obstacle in static_obstacles:
        distance = math.hypot(x - obstacle.x, y - obstacle.y)
        if distance <= radius + obstacle.radius + clearance:
            return True
    return False


def circle_inside_bounds(x, y, radius, bounds):
    x_min, x_max, y_min, y_max = bounds
    return (
        x_min + radius <= x <= x_max - radius
        and y_min + radius <= y <= y_max - radius
    )


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    closest_x = ax + ratio * dx
    closest_y = ay + ratio * dy
    return math.hypot(px - closest_x, py - closest_y)


def motion_crosses_static(start_x, start_y, end_x, end_y, radius, static_obstacles, clearance):
    for obstacle in static_obstacles:
        distance = point_segment_distance(
            obstacle.x,
            obstacle.y,
            start_x,
            start_y,
            end_x,
            end_y,
        )
        if distance <= radius + obstacle.radius + clearance:
            return True
    return False


def generate_dynamic_obstacles(
    count,
    bounds,
    static_obstacles,
    seed,
    radius,
    min_speed,
    max_speed,
    clearance,
):
    rng = random.Random(seed)
    x_min, x_max, y_min, y_max = bounds
    dynamic_obstacles = []

    for _ in range(count):
        placed = False
        for _attempt in range(2000):
            x = rng.uniform(x_min + radius, x_max - radius)
            y = rng.uniform(y_min + radius, y_max - radius)
            if circle_overlaps_static(x, y, radius, static_obstacles, clearance):
                continue
            if any(
                math.hypot(x - other.x, y - other.y)
                <= radius + other.radius + clearance
                for other in dynamic_obstacles
            ):
                continue
            dynamic_obstacles.append(
                MovingObstacle(
                    x=x,
                    y=y,
                    radius=radius,
                    speed=rng.uniform(min_speed, max_speed),
                    heading=rng.uniform(-math.pi, math.pi),
                )
            )
            placed = True
            break
        if not placed:
            print("[Dynamic obstacle] available free space is not enough")

    return dynamic_obstacles


def update_dynamic_obstacles(obstacles, static_obstacles, bounds, dt, rng, clearance):
    for index, obstacle in enumerate(obstacles):
        if rng.random() < 0.015:
            obstacle.heading += rng.uniform(-math.pi / 2.0, math.pi / 2.0)

        next_x = obstacle.x + obstacle.speed * math.cos(obstacle.heading) * dt
        next_y = obstacle.y + obstacle.speed * math.sin(obstacle.heading) * dt

        blocked = (
            not circle_inside_bounds(next_x, next_y, obstacle.radius, bounds)
            or motion_crosses_static(
                obstacle.x,
                obstacle.y,
                next_x,
                next_y,
                obstacle.radius,
                static_obstacles,
                clearance,
            )
        )

        if not blocked:
            for other_index, other in enumerate(obstacles):
                if other_index == index:
                    continue
                if math.hypot(next_x - other.x, next_y - other.y) <= (
                    obstacle.radius + other.radius + clearance
                ):
                    blocked = True
                    break

        if blocked:
            obstacle.heading = normalize_angle(
                obstacle.heading + math.pi + rng.uniform(-0.7, 0.7)
            )
            continue

        obstacle.x = next_x
        obstacle.y = next_y


def infer_goal_pose(start, goal_xy):
    return Pose(
        float(goal_xy[0]),
        float(goal_xy[1]),
        math.atan2(goal_xy[1] - start.y, goal_xy[0] - start.x),
    )


def rrt_result_to_global_path(result):
    """将 RRT/路径优化器的四元组结果转换为局部规划器使用的 Pose 列表。"""
    if result is None:
        return []
    if len(result) < 3:
        raise ValueError("RRT 路径结果至少应包含 path_x、path_y 和 path_yaw")

    path_x, path_y, path_yaw = result[:3]
    if not (len(path_x) == len(path_y) == len(path_yaw)):
        raise ValueError("RRT 路径的 x、y、yaw 数量不一致")

    return [
        Pose(float(x), float(y), float(yaw))
        for x, y, yaw in zip(path_x, path_y, path_yaw)
    ]


def append_goal_reference(path, goal_xy):
    if not path:
        return path
    last = path[-1]
    distance = math.hypot(goal_xy[0] - last.x, goal_xy[1] - last.y)
    if distance < 0.05:
        return path
    yaw = math.atan2(goal_xy[1] - last.y, goal_xy[0] - last.x)
    path.append(Pose(float(goal_xy[0]), float(goal_xy[1]), yaw))
    return path


def smooth_reference_path(path, iterations=2):
    """Smooth the RRT* polyline reference before handing it to DWA."""
    if len(path) < 3 or iterations <= 0:
        return recompute_path_yaw(path)

    points = [(pose.x, pose.y) for pose in path]
    for _ in range(iterations):
        smoothed = [points[0]]
        for first, second in zip(points[:-1], points[1:]):
            x1, y1 = first
            x2, y2 = second
            smoothed.append((0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2))
            smoothed.append((0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2))
        smoothed.append(points[-1])
        points = smoothed

    return recompute_path_yaw([Pose(x, y, 0.0) for x, y in points])


def recompute_path_yaw(path):
    if not path:
        return path
    result = []
    for index, pose in enumerate(path):
        if index < len(path) - 1:
            next_pose = path[index + 1]
            yaw = math.atan2(next_pose.y - pose.y, next_pose.x - pose.x)
        elif result:
            yaw = result[-1].yaw
        else:
            yaw = pose.yaw
        result.append(Pose(pose.x, pose.y, yaw))
    return result


def plan_oag_hrrt_star(
    start,
    goal_xy,
    env,
    vehicle,
    max_iterations,
    seed,
    rectangle_length,
    rectangle_width,
    use_goal_connector=False,  # 🆕 新增参数
    allow_reverse=False,  # 🆕 新增参数
):
    goal = infer_goal_pose(start, goal_xy)
    env.goal_rectangle.anchor_x = start.x
    env.goal_rectangle.anchor_y = start.y
    env.goal_rectangle.length = rectangle_length
    env.goal_rectangle.width = rectangle_width

    corridors = [
        SamplingCorridor(c["x1"], c["y1"], c["x2"], c["y2"], c["width"])
        for c in env.corridors
    ]
    wheel_base = 2.5
    max_steer = math.radians(30.0)
    curvature = math.tan(max_steer) / wheel_base

    planner = AckermannRRTStar(
        start=start,
        goal=goal,
        bounds=env.bounds,
        vehicle=vehicle,
        obstacles=env.obstacles,
        curvature=curvature,
        expand_length=3.0,
        step_size=0.08,
        max_iterations=max_iterations,
        near_radius=5.0,
        use_hybrid_sampling=True,
        corridors=corridors,
        goal_rectangle=env.goal_rectangle,
        rectangle_anchor_mode="closest_to_goal",
        goal_probability=0.20,
        corridor_probability=0.0,
        rectangle_probability=0.45,
        allow_reverse=allow_reverse,
        use_tangent_guidance=True,
        shrink_probability=0.35,
        shrink_length_factor=0.70,
        shrink_width_factor=0.70,
        shrink_activation_distance=18.0,
        near_anchor_probability=0.55,
        near_anchor_length_ratio=0.40,
        adaptive_sampling_probabilities=True,
        cluster_shape="ellipse",
        use_goal_connector=use_goal_connector,  # 🆕 使用参数而不是硬编码
        relax_goal_yaw=True,
        random_seed=seed,
    )
    result = planner.planning()
    return result, planner


class OAGHRRTDWAApp:
    def __init__(self, args):
        self.args = args

        # 🆕 加载配置文件（如果提供）
        self.config = None
        if hasattr(args, 'config') and args.config:
            self.config = load_config(args.config)
            if self.config:
                print("[应用] 使用配置文件参数")

        self.env = load_environment(args.map)
        self.bounds = self.env.bounds
        self.static_obstacles = self.env.obstacles
        self.rng = random.Random(args.seed + 991)

        vehicle_section = (self.config or {}).get('vehicle', {})
        geometry_section = vehicle_section.get('geometry', {})
        self.vehicle = VehicleGeometry(
            front_length=geometry_section.get('front_length', args.front_length),
            rear_length=geometry_section.get('rear_length', args.rear_length),
            width=geometry_section.get('width', args.vehicle_width),
            safety_margin=geometry_section.get('safety_margin', args.safety_margin),
        )
        # 使用 local_path_planning 中的 DWA/TEB/Adaptive TEB-DWA。
        planner_section = (self.config or {}).get('local_planner', {})
        self.local_planner_name = str(planner_section.get('type', 'dwa')).lower()
        if self.local_planner_name not in ('dwa', 'teb', 'adaptive_teb_dwa'):
            raise ValueError(
                "local_planner.type 必须是 'dwa'、'teb' 或 'adaptive_teb_dwa'"
            )

        config_class = DWAConfig if self.local_planner_name == 'dwa' else TEBConfig
        self.local_config = config_class()

        # 命令行参数作为无配置文件时的后备值。
        fallback_values = {
            'wheel_base': args.wheel_base,
            'max_speed': args.max_speed,
            'max_accel': args.max_accel,
            'max_decel': args.max_decel,
            'max_steer_deg': args.max_steer_deg,
            'max_steer_rate_deg': args.max_steer_rate_deg,
            'speed_samples': args.speed_samples,
            'steer_samples': args.steer_samples,
            'dt': args.dt,
            'predict_time': args.predict_time,
            'lookahead_distance': args.lookahead_distance,
            'goal_tolerance': args.goal_tolerance,
            'vehicle_front_length': args.front_length,
            'vehicle_rear_length': args.rear_length,
            'vehicle_width': args.vehicle_width,
            'vehicle_safety_margin': args.safety_margin,
        }
        valid_fields = self.local_config.__dataclass_fields__
        for key, value in fallback_values.items():
            if key in valid_fields:
                setattr(self.local_config, key, value)

        yaml_values = {
            'wheel_base': vehicle_section.get('wheel_base'),
            'max_steer_deg': vehicle_section.get('max_steer_deg'),
            'vehicle_front_length': geometry_section.get('front_length'),
            'vehicle_rear_length': geometry_section.get('rear_length'),
            'vehicle_width': geometry_section.get('width'),
            'vehicle_safety_margin': geometry_section.get('safety_margin'),
        }
        local_config_section = (
            'teb' if self.local_planner_name == 'adaptive_teb_dwa'
            else self.local_planner_name
        )
        yaml_values.update((self.config or {}).get(local_config_section, {}))
        unknown_fields = sorted(set(yaml_values) - set(valid_fields))
        if unknown_fields:
            print(f"[配置] 忽略 {self.local_planner_name} 未知参数: {', '.join(unknown_fields)}")
        for key, value in yaml_values.items():
            if key in valid_fields and value is not None:
                setattr(self.local_config, key, value)

        print(f"[局部规划] 使用 {self.local_planner_name.upper()}，参数来自 local_path_planning")

        # 🆕 根据配置决定是否生成动态障碍物
        dynamic_enabled = True  # 默认启用
        if self.config and 'dynamic_obstacles' in self.config:
            dynamic_enabled = self.config['dynamic_obstacles'].get('enabled', True)

        if dynamic_enabled and args.dynamic_count > 0:
            self.dynamic_obstacles = generate_dynamic_obstacles(
                count=args.dynamic_count,
                bounds=self.bounds,
                static_obstacles=self.static_obstacles,
                seed=args.seed,
                radius=args.dynamic_radius,
                min_speed=args.dynamic_min_speed,
                max_speed=args.dynamic_max_speed,
                clearance=args.dynamic_static_clearance,
            )
        else:
            self.dynamic_obstacles = []  # 空列表，不生成动态障碍物

        self.global_path = []
        self.raw_path = []  # 🆕 保存未平滑的路径
        self.rrt_planner = None  # 🆕 保存 RRT 规划器对象（用于显示搜索树）
        self.local_planner = None
        self.state = None
        self.executed_x = []
        self.executed_y = []
        self.drag_start = None
        self.running = False
        self.paused = False
        self.adaptive_update_counter = 0
        self.cached_adaptive_result = None

        self.fig, self.ax = plt.subplots(figsize=(11, 10))
        self.valid_collection = LineCollection([], linewidths=0.6, colors="gray", alpha=0.14)
        self.invalid_collection = LineCollection([], linewidths=0.6, colors="red", alpha=0.10)
        self.best_line = None
        self.executed_line = None
        self.local_goal_point = None
        self.global_line = None
        self.start_point = None
        self.goal_point = None
        self.vehicle_patch = Polygon([(0, 0), (0, 0), (0, 0)], closed=True, fill=False)
        self.dynamic_patches = []

        self.draw_base_scene()
        self.connect_events()
        self.timer = self.fig.canvas.new_timer(interval=max(1, int(self.local_config.dt * 1000)))
        self.timer.add_callback(self.update)
        self.timer.start()

    def draw_base_scene(self):
        self.ax.clear()
        self.valid_collection = LineCollection([], linewidths=0.6, colors="gray", alpha=0.14)
        self.invalid_collection = LineCollection([], linewidths=0.6, colors="red", alpha=0.10)
        self.ax.add_collection(self.valid_collection)
        self.ax.add_collection(self.invalid_collection)
        self.vehicle_patch = Polygon(
            [(0, 0), (0, 0), (0, 0)],
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=2,
            label="vehicle",
        )
        self.ax.add_patch(self.vehicle_patch)
        self.best_line, = self.ax.plot(
            [], [], color="limegreen", linewidth=2.5,
            label=f"{self.local_planner_name.upper()} best",
        )
        self.executed_line, = self.ax.plot([], [], color="royalblue", linewidth=2.2, label="executed")
        self.local_goal_point, = self.ax.plot([], [], "kx", markersize=9, label="local goal")
        # 🆕 添加搜索树显示（浅灰色细线）
        self.tree_collection = LineCollection([], linewidths=0.3, colors="lightgray", alpha=0.3, label="search tree")
        self.ax.add_collection(self.tree_collection)
        # 🆕 添加未平滑路径显示（灰色虚线）
        self.raw_path_line, = self.ax.plot([], [], ":", color="gray", linewidth=1.5, alpha=0.6, label="raw path (unsmoothed)")
        self.global_line, = self.ax.plot([], [], "--", color="darkorange", linewidth=2.0, label="smoothed path")
        self.start_point, = self.ax.plot([], [], "go", markersize=9, label="start")
        self.goal_point, = self.ax.plot(
            [self.env.goal_pos[0]], [self.env.goal_pos[1]], "r*", markersize=16, label="goal"
        )

        for obstacle in self.static_obstacles:
            self.ax.add_patch(
                Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    facecolor="lightcoral",
                    edgecolor="red",
                    alpha=0.55,
                )
            )

        self.dynamic_patches = []
        for index, obstacle in enumerate(self.dynamic_obstacles):
            patch = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                facecolor="deepskyblue",
                edgecolor="navy",
                alpha=0.75,
                label="dynamic obstacle" if index == 0 else None,
            )
            self.ax.add_patch(patch)
            self.dynamic_patches.append(patch)

        x_min, x_max, y_min, y_max = self.bounds
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(
            f"OAG-HRRT*-{self.local_planner_name.upper()} | left press/release sets start pose | space pause | R reset"
        )
        self.ax.legend(loc="upper left", fontsize=9)
        self.fig.tight_layout()

    def connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_start = (float(event.xdata), float(event.ydata))
        self.paused = True
        self.ax.set_title("Release left mouse button to set start yaw and plan")
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.drag_start is None or event.inaxes != self.ax or event.button != 1:
            self.drag_start = None
            return
        if event.xdata is None or event.ydata is None:
            self.drag_start = None
            return

        sx, sy = self.drag_start
        yaw = math.atan2(event.ydata - sy, event.xdata - sx)
        if math.hypot(event.xdata - sx, event.ydata - sy) < 0.2:
            yaw = math.atan2(self.env.goal_pos[1] - sy, self.env.goal_pos[0] - sx)
        self.drag_start = None
        self.start_with_pose(Pose(sx, sy, yaw))

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
            self.ax.set_title("paused" if self.paused else "running")
            self.fig.canvas.draw_idle()
        elif event.key in ("r", "R"):
            self.running = False
            self.paused = False
            self.global_path = []
            self.local_planner = None
            self.adaptive_update_counter = 0
            self.cached_adaptive_result = None
            self.state = None
            self.executed_x = []
            self.executed_y = []
            self.draw_base_scene()
            self.fig.canvas.draw_idle()
        elif event.key == "escape":
            self.timer.stop()
            plt.close(self.fig)

    def start_with_pose(self, start):
        if circle_overlaps_static(
            start.x,
            start.y,
            max(self.vehicle.front_length, self.vehicle.rear_length),
            self.static_obstacles,
            self.vehicle.safety_margin,
        ):
            self.ax.set_title("Start is too close to a static obstacle; choose another pose")
            self.fig.canvas.draw_idle()
            return

        self.ax.set_title("OAG-HRRT* planning global path...")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        # 🆕 从配置文件读取参数
        use_goal_connector = False  # 默认禁用
        allow_reverse = False  # 默认禁用倒车
        if self.config and 'planner' in self.config:
            use_goal_connector = self.config['planner'].get('use_goal_connector', False)
            allow_reverse = self.config['planner'].get('allow_reverse', False)

        result, _planner = plan_oag_hrrt_star(
            start=start,
            goal_xy=self.env.goal_pos,
            env=self.env,
            vehicle=self.vehicle,
            max_iterations=self.args.max_iterations,
            seed=self.args.seed,
            rectangle_length=self.args.rectangle_length,
            rectangle_width=self.args.rectangle_width,
            use_goal_connector=use_goal_connector,  # 🆕 传入配置参数
            allow_reverse=allow_reverse,  # 🆕 传入配置参数
        )
        # 🆕 保存 RRT 规划器对象用于显示搜索树
        self.rrt_planner = _planner

        if result is None:
            self.running = False
            self.paused = False
            self.ax.set_title("OAG-HRRT* failed; click another start pose or increase iterations")
            self.fig.canvas.draw_idle()
            return

        # 🆕 路径优化：如果启用，则对 RRT 生成的路径进行优化
        original_result = result
        if self.args.optimize_path:
            self.ax.set_title("Optimizing global path...")
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

            # 获取优化配置
            opt_config = None
            if self.config and 'path_optimization' in self.config:
                opt_config = self.config['path_optimization']

            result = optimize_global_path(
                result=result,
                vehicle=self.vehicle,
                obstacles=self.static_obstacles,
                config=opt_config,
                verbose=True,
            )

            if result is None:
                print("[应用] 路径优化失败，使用原始路径")
                result = original_result

        raw_path = append_goal_reference(
            rrt_result_to_global_path(result),
            self.env.goal_pos,
        )
        # 🆕 保存未平滑的路径用于显示
        self.raw_path = raw_path
        curvature_enabled = bool(
            self.config
            and self.config.get('path_optimization', {})
            .get('curvature_smoothing', {})
            .get('enabled', True)
        )
        # 曲率平滑已经完成等弧长插补，不再叠加 Chaikin，避免点数指数增长。
        smoothing_iterations = 0 if curvature_enabled else self.args.smoothing_iterations
        self.global_path = smooth_reference_path(raw_path, iterations=smoothing_iterations)
        if self.local_planner_name == 'adaptive_teb_dwa':
            adaptive = (self.config or {}).get('adaptive_teb_dwa', {})
            for key, value in adaptive.get('teb_overrides', {}).items():
                if key in self.local_config.__dataclass_fields__:
                    setattr(self.local_config, key, value)
            dwa_values = dict((self.config or {}).get('dwa', {}))
            dwa_values.update(adaptive.get('dwa_overrides', {}))
            for key in DWAConfig.__dataclass_fields__:
                if hasattr(self.local_config, key) and key not in dwa_values:
                    dwa_values[key] = getattr(self.local_config, key)
            dwa_config = DWAConfig(**{
                key: value for key, value in dwa_values.items()
                if key in DWAConfig.__dataclass_fields__
            })
            self.local_planner = AdaptiveTEBDWAPlanner(
                teb_config=self.local_config,
                dwa_config=dwa_config,
                bounds=self.bounds,
                planner_config=AdaptivePlannerConfig(**{
                    key: value for key, value in adaptive.get('planner', {}).items()
                    if key in AdaptivePlannerConfig.__dataclass_fields__
                }),
                window_config=AdaptiveWindowConfig(**{
                    key: value for key, value in adaptive.get('adaptive_window', {}).items()
                    if key in AdaptiveWindowConfig.__dataclass_fields__
                }),
                feedback_config=FeedbackConfig(**{
                    key: value for key, value in adaptive.get('feedback', {}).items()
                    if key in FeedbackConfig.__dataclass_fields__
                }),
            )
        else:
            planner_class = DWAPlanner if self.local_planner_name == 'dwa' else TEBPlanner
            self.local_planner = planner_class(config=self.local_config, bounds=self.bounds)
        self.local_planner.set_global_path(self.global_path)
        self.adaptive_update_counter = 0
        self.cached_adaptive_result = None

        self.state = VehicleState(
            x=start.x,
            y=start.y,
            yaw=start.yaw,
            speed=0.0,
            steering=0.0,
        )
        self.executed_x = [start.x]
        self.executed_y = [start.y]
        self.running = True
        self.paused = False
        self.redraw_after_global_plan()

    def redraw_after_global_plan(self):
        # 🆕 显示搜索树
        if self.rrt_planner is not None and hasattr(self.rrt_planner, 'nodes'):
            tree_segments = []
            for node in self.rrt_planner.nodes:
                if node.parent is not None and node.path_x and node.path_y:
                    # 显示每个节点到其父节点的路径
                    segment = [(node.path_x[i], node.path_y[i]) for i in range(len(node.path_x))]
                    if segment:
                        tree_segments.append(segment)
            self.tree_collection.set_segments(tree_segments)
        else:
            self.tree_collection.set_segments([])

        # 🆕 显示平滑后的路径
        path_x = [pose.x for pose in self.global_path]
        path_y = [pose.y for pose in self.global_path]
        self.global_line.set_data(path_x, path_y)

        # 🆕 显示未平滑的原始路径
        if self.raw_path and self.args.smoothing_iterations > 0:
            raw_x = [pose.x for pose in self.raw_path]
            raw_y = [pose.y for pose in self.raw_path]
            self.raw_path_line.set_data(raw_x, raw_y)
        else:
            # 如果没有平滑或未保存原始路径，则不显示
            self.raw_path_line.set_data([], [])

        self.start_point.set_data([path_x[0]], [path_y[0]])
        self.executed_line.set_data(self.executed_x, self.executed_y)
        # 使用 Pose 对象而不是 self.state.pose
        pose = Pose(self.state.x, self.state.y, self.state.yaw)
        self.vehicle_patch.set_xy(get_vehicle_corners(pose, self.vehicle))
        self.ax.set_title(
            f"OAG-HRRT*-{self.local_planner_name.upper()} running with dynamic obstacles"
        )
        self.fig.canvas.draw_idle()

    @staticmethod
    def trajectory_to_segment(trajectory):
        return [(pose.x, pose.y) for pose in trajectory]

    def draw_local_result(self, plan_result):
        """绘制 local_path_planning 返回的 DWA 或 TEB 结果。"""
        valid_segments = []
        invalid_segments = []

        # 检查是否有 candidates 属性
        if hasattr(plan_result, 'candidates'):
            for candidate in plan_result.candidates:
                segment = self.trajectory_to_segment(candidate.trajectory)
                if candidate.valid:
                    valid_segments.append(segment)
                else:
                    invalid_segments.append(segment)

        self.valid_collection.set_segments(valid_segments)
        self.invalid_collection.set_segments(invalid_segments)

        # 绘制最优轨迹
        if hasattr(plan_result, 'best') and plan_result.best is not None:
            self.best_line.set_data(
                [pose.x for pose in plan_result.best.trajectory],
                [pose.y for pose in plan_result.best.trajectory],
            )
        elif hasattr(plan_result, 'trajectory') and plan_result.trajectory:
            self.best_line.set_data(
                [pose.x for pose in plan_result.trajectory],
                [pose.y for pose in plan_result.trajectory],
            )
        else:
            self.best_line.set_data([], [])

        if hasattr(plan_result, 'evaluation') and plan_result.evaluation is not None:
            evaluation = plan_result.evaluation
            self.ax.set_title(
                f"Adaptive TEB-DWA | score={evaluation.score:.2f} | "
                f"feedback={plan_result.feedback_iterations} | {evaluation.reason}"
            )

        # 绘制局部目标点
        if hasattr(plan_result, 'local_goal'):
            self.local_goal_point.set_data(
                [plan_result.local_goal.x],
                [plan_result.local_goal.y],
            )
        else:
            self.local_goal_point.set_data([], [])

    def update(self):
        if self.paused:
            return

        update_dynamic_obstacles(
            obstacles=self.dynamic_obstacles,
            static_obstacles=self.static_obstacles,
            bounds=self.bounds,
            dt=self.local_config.dt,
            rng=self.rng,
            clearance=self.args.dynamic_static_clearance,
        )
        for patch, obstacle in zip(self.dynamic_patches, self.dynamic_obstacles):
            patch.center = (obstacle.x, obstacle.y)

        if not self.running or self.local_planner is None or self.state is None:
            self.fig.canvas.draw_idle()
            return

        # 🆕 使用新的接口
        obstacles = self.static_obstacles + [
            obstacle.as_circle() for obstacle in self.dynamic_obstacles
        ]
        if self.local_planner_name == 'adaptive_teb_dwa':
            adaptive_cfg = (self.config or {}).get('adaptive_teb_dwa', {})
            configured_interval = max(1, int(adaptive_cfg.get('replan_interval_steps', 2)))
            # 动态障碍物存在时不复用旧控制，保持每周期重规划。
            interval = 1 if self.dynamic_obstacles else configured_interval
            should_replan = (
                self.cached_adaptive_result is None
                or self.adaptive_update_counter % interval == 0
            )
            if should_replan:
                plan_result = self.local_planner.plan(self.state, obstacles)
                self.cached_adaptive_result = (
                    plan_result if getattr(plan_result, 'success', False) else None
                )
            else:
                plan_result = self.cached_adaptive_result
            self.adaptive_update_counter += 1
        else:
            plan_result = self.local_planner.plan(self.state, obstacles)
        self.draw_local_result(plan_result)

        # 检查是否有有效结果
        best_control = None
        if hasattr(plan_result, 'best') and plan_result.best is not None:
            best_control = plan_result.best.control
        elif plan_result is not None and getattr(plan_result, 'success', False):
            if hasattr(plan_result, 'control'):
                best_control = plan_result.control
            elif getattr(plan_result, 'trajectory', None):
                first = plan_result.trajectory[0]
                best_control = type(
                    'AdaptiveControl', (),
                    {'speed': first.v, 'steering': first.steering},
                )()

        if best_control is None:
            self.state.speed = 0.0
            self.state.steering = 0.0
            self.ax.set_title(
                f"{self.local_planner_name.upper()} stopped: no collision-free local trajectory"
            )
        else:
            # 更新状态
            dt = self.local_config.dt
            speed = best_control.speed
            steering = best_control.steering

            yaw_rate = speed / self.local_config.wheel_base * math.tan(steering)

            if abs(yaw_rate) < 1e-10:
                self.state.x += speed * math.cos(self.state.yaw) * dt
                self.state.y += speed * math.sin(self.state.yaw) * dt
            else:
                new_yaw = self.state.yaw + yaw_rate * dt
                radius = speed / yaw_rate
                self.state.x += radius * (math.sin(new_yaw) - math.sin(self.state.yaw))
                self.state.y -= radius * (math.cos(new_yaw) - math.cos(self.state.yaw))
                self.state.yaw = (new_yaw + math.pi) % (2.0 * math.pi) - math.pi

            self.state.speed = speed
            self.state.steering = steering

            self.executed_x.append(self.state.x)
            self.executed_y.append(self.state.y)
            self.executed_line.set_data(self.executed_x, self.executed_y)

            # 更新车辆位置
            pose = Pose(self.state.x, self.state.y, self.state.yaw)
            self.vehicle_patch.set_xy(get_vehicle_corners(pose, self.vehicle))
            if hasattr(plan_result, 'evaluation') and plan_result.evaluation is not None:
                self.ax.set_title(
                    f"OAG-HRRT*-Adaptive TEB-DWA | speed={self.state.speed:.2f} m/s | "
                    f"score={plan_result.evaluation.score:.2f} | "
                    f"feedback={plan_result.feedback_iterations}"
                )
            else:
                self.ax.set_title(
                    f"OAG-HRRT*-{self.local_planner_name.upper()} | "
                    f"speed={self.state.speed:.2f} m/s | "
                    f"steer={math.degrees(self.state.steering):.1f} deg"
                )

        # 检查是否到达目标
        goal = self.global_path[-1]
        dist_to_goal = math.hypot(self.state.x - goal.x, self.state.y - goal.y)
        if dist_to_goal < self.local_config.goal_tolerance:
            self.running = False
            self.ax.set_title(f"Goal reached by OAG-HRRT*-{self.local_planner_name.upper()}")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="OAG-HRRT*-DWA orchard demo")

    # 环境和地图
    parser.add_argument("--map", default=DEFAULT_MAP_PATH, help="orchard environment npz")
    parser.add_argument("--seed", type=int, default=7)

    # 🆕 配置文件支持
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径 (YAML格式)，如果存在则从配置文件加载参数"
    )

    # RRT 规划器参数
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--rectangle-length", type=float, default=30.0)
    parser.add_argument("--rectangle-width", type=float, default=22.0)
    parser.add_argument("--smoothing-iterations", type=int, default=2)

    # 🆕 路径优化参数
    parser.add_argument(
        "--optimize-path",
        action="store_true",
        help="启用路径优化 (Path Shortcut)，减少冗余节点"
    )
    parser.add_argument(
        "--no-optimize-path",
        dest="optimize_path",
        action="store_false",
        help="禁用路径优化"
    )
    parser.set_defaults(optimize_path=True)  # 默认启用优化

    # 动态障碍物参数
    parser.add_argument("--dynamic-count", type=int, default=6)
    parser.add_argument("--dynamic-radius", type=float, default=0.75)
    parser.add_argument("--dynamic-min-speed", type=float, default=0.25)
    parser.add_argument("--dynamic-max-speed", type=float, default=0.75)
    parser.add_argument("--dynamic-static-clearance", type=float, default=0.25)

    # DWA 局部规划器参数
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--predict-time", type=float, default=2.8)
    parser.add_argument("--lookahead-distance", type=float, default=3.2)
    parser.add_argument("--goal-tolerance", type=float, default=0.45)

    # 车辆参数
    parser.add_argument("--wheel-base", type=float, default=2.5)
    parser.add_argument("--front-length", type=float, default=3.0)
    parser.add_argument("--rear-length", type=float, default=1.0)
    parser.add_argument("--vehicle-width", type=float, default=1.6)
    parser.add_argument("--safety-margin", type=float, default=0.18)
    parser.add_argument("--max-speed", type=float, default=1.3)
    parser.add_argument("--max-accel", type=float, default=1.0)
    parser.add_argument("--max-decel", type=float, default=1.5)
    parser.add_argument("--max-steer-deg", type=float, default=30.0)
    parser.add_argument("--max-steer-rate-deg", type=float, default=70.0)
    parser.add_argument("--speed-samples", type=int, default=7)
    parser.add_argument("--steer-samples", type=int, default=19)

    return parser.parse_args()


def main():
    app = OAGHRRTDWAApp(parse_args())
    app.show()


if __name__ == "__main__":
    main()
