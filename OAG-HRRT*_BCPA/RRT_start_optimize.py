"""
独立的 OAG-HRRT* 全局路径规划器

从 npz 地图文件进行路径规划，支持路径优化
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml


# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
INNOVATION_DIR = os.path.join(PROJECT_ROOT, "global_path_planning", "innovation_sample")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "oag_hrrt_dwa", "config.yaml")

for path in (INNOVATION_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from ackermann_rrt_star import AckermannRRTStar
from hybrid_sampler import SamplingCorridor
from orchard_environment import load_environment
from vehicle.reeds_shepp_path_test import Pose
from vehicle.vehicle_collision_test import VehicleGeometry, check_pose_collision

# 导入路径优化器
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "path_optimizer")))
from path_optimizer import CurvatureSmoother, ShortcutOptimizer


def load_path_optimization_config(
    config_path=DEFAULT_CONFIG_PATH,
    shortcut_iterations=None,
    disable_curvature_smoothing=False,
):
    """读取共享的路径优化参数，供独立规划器和 ROS 桥接共用。"""
    with open(config_path, 'r', encoding='utf-8') as config_file:
        app_config = yaml.safe_load(config_file) or {}

    opt_config = dict(app_config.get('path_optimization', {}))
    opt_config['curvature_smoothing'] = dict(
        opt_config.get('curvature_smoothing', {})
    )
    if shortcut_iterations is not None:
        opt_config['max_iterations'] = int(shortcut_iterations)
    if disable_curvature_smoothing:
        opt_config['curvature_smoothing']['enabled'] = False
    return opt_config


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
    # 与 oag_hrrt_dwa_demo.py 使用完全相同的配置解析规则：
    # YAML 中 angle_threshold 的单位为度，在此统一转为弧度。
    config = config or {}
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

    # 4. 执行 Shortcut 优化
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


def plan_global_path(
    start_pose,
    goal_pose,
    env,
    vehicle,
    max_iterations=2500,
    seed=42,
    rectangle_length=30.0,
    rectangle_width=22.0,
    use_goal_connector=False,
    allow_reverse=False,
):
    """
    使用 OAG-HRRT* 规划全局路径

    Args:
        start_pose: Pose 起始位姿
        goal_pose: Pose 目标位姿
        env: 环境对象（从 npz 加载）
        vehicle: VehicleGeometry 车辆几何参数
        max_iterations: 最大迭代次数
        seed: 随机种子
        rectangle_length: 采样矩形长度
        rectangle_width: 采样矩形宽度
        use_goal_connector: 是否使用目标连接器
        allow_reverse: 是否允许倒车

    Returns:
        result: (path_x, path_y, path_yaw, directions) 或 None
        planner: AckermannRRTStar 规划器对象
    """
    # 设置矩形采样区域
    env.goal_rectangle.anchor_x = start_pose.x
    env.goal_rectangle.anchor_y = start_pose.y
    env.goal_rectangle.length = rectangle_length
    env.goal_rectangle.width = rectangle_width

    # 准备走廊（如果环境中有定义）
    corridors = [
        SamplingCorridor(c["x1"], c["y1"], c["x2"], c["y2"], c["width"])
        for c in env.corridors
    ]

    # 车辆参数：与 oag_hrrt_dwa_demo.py 一致
    wheel_base = 2.5
    max_steer = math.radians(30.0)
    curvature = math.tan(max_steer) / wheel_base

    print(f"\n{'='*70}")
    print(f"[规划] OAG-HRRT* 全局路径规划")
    print(f"[规划] 起点: ({start_pose.x:.2f}, {start_pose.y:.2f}, {math.degrees(start_pose.yaw):.1f}°)")
    print(f"[规划] 终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {math.degrees(goal_pose.yaw):.1f}°)")
    print(f"[规划] 最大迭代: {max_iterations}")
    print(f"[规划] 障碍物数: {len(env.obstacles)}")
    print(f"{'='*70}\n")

    planner = AckermannRRTStar(
        start=start_pose,
        goal=goal_pose,
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
        use_goal_connector=use_goal_connector,
        relax_goal_yaw=False,
        random_seed=seed,
    )

    result = planner.planning()

    if result is None:
        print(f"\n[规划] 失败：未找到路径\n")
        return None, planner

    path_x, path_y, path_yaw = result[:3]
    print(f"\n{'='*70}")
    print(f"[规划] 成功！")
    print(f"[规划] - 首次找到路径的迭代: {planner.first_solution_iteration}")
    print(f"[规划] - 总节点数: {len(planner.nodes)}")
    print(f"[规划] - 路径点数: {len(path_x)}")
    print(f"[规划] - 最终代价: {planner.nodes[planner.goal_index].cost:.2f}")
    print(f"[规划] - 规划耗时: {planner.planning_time:.2f}秒")
    print(f"{'='*70}\n")

    return result, planner


def select_poses_interactively(env):
    """按 oag_hrrt_dwa_demo.py 的鼠标拖拽方式选择起终位姿。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 10))
    selection = {
        "start": None,
        "goal": None,
        "drag_start": None,
        "drag_button": None,
        "completed": False,
    }

    for obs in env.obstacles:
        ax.add_patch(Circle(
            (obs.x, obs.y),
            obs.radius,
            facecolor="lightcoral",
            edgecolor="red",
            alpha=0.6,
        ))

    start_point, = ax.plot([], [], "go", markersize=12, label="Start")
    goal_point, = ax.plot([], [], "r*", markersize=16, label="Goal")
    start_heading, = ax.plot([], [], color="green", linewidth=2.5, label="Start yaw")
    goal_heading, = ax.plot([], [], color="red", linewidth=2.5, label="Goal yaw")

    def set_pose_indicator(point_artist, heading_artist, pose, length=1.0):
        point_artist.set_data([pose.x], [pose.y])
        heading_artist.set_data(
            [pose.x, pose.x + length * math.cos(pose.yaw)],
            [pose.y, pose.y + length * math.sin(pose.yaw)],
        )

    def update_title(message=None):
        if message is None:
            message = (
                "Left-drag: start pose | Right-drag: goal pose | "
                "R: reset | Esc: cancel"
            )
        ax.set_title(message)
        fig.canvas.draw_idle()

    def clear_selection():
        selection.update({
            "start": None,
            "goal": None,
            "drag_start": None,
            "drag_button": None,
            "completed": False,
        })
        start_point.set_data([], [])
        goal_point.set_data([], [])
        start_heading.set_data([], [])
        goal_heading.set_data([], [])
        update_title()

    def on_press(event):
        if event.inaxes != ax or event.button not in (1, 3):
            return
        if event.xdata is None or event.ydata is None:
            return
        selection["drag_start"] = (float(event.xdata), float(event.ydata))
        selection["drag_button"] = event.button
        pose_name = "start" if event.button == 1 else "goal"
        update_title(f"Release mouse button to set {pose_name} yaw")

    def on_release(event):
        if (
            selection["drag_start"] is None
            or event.inaxes != ax
            or event.button != selection["drag_button"]
            or event.xdata is None
            or event.ydata is None
        ):
            selection["drag_start"] = None
            selection["drag_button"] = None
            return

        sx, sy = selection["drag_start"]
        selected_button = selection["drag_button"]
        yaw = math.atan2(event.ydata - sy, event.xdata - sx)
        if math.hypot(event.xdata - sx, event.ydata - sy) < 0.2:
            if selected_button == 1:
                reference_goal = selection["goal"]
                if reference_goal is None:
                    yaw = math.atan2(env.goal_pos[1] - sy, env.goal_pos[0] - sx)
                else:
                    yaw = math.atan2(reference_goal.y - sy, reference_goal.x - sx)
            elif selection["start"] is not None:
                yaw = math.atan2(
                    sy - selection["start"].y,
                    sx - selection["start"].x,
                )
            else:
                yaw = 0.0

        pose = Pose(sx, sy, yaw)
        selection["drag_start"] = None
        selection["drag_button"] = None
        if selected_button == 1:
            selection["start"] = pose
            set_pose_indicator(start_point, start_heading, pose)
        else:
            selection["goal"] = pose
            env.goal_pos = (pose.x, pose.y)
            set_pose_indicator(goal_point, goal_heading, pose)

        if selection["start"] is not None and selection["goal"] is not None:
            selection["completed"] = True
            plt.close(fig)
        else:
            missing = "goal" if selection["goal"] is None else "start"
            update_title(f"Set {missing} pose to begin planning")

    def on_key(event):
        if event.key in ("r", "R"):
            clear_selection()
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    ax.set_xlim(env.bounds[0], env.bounds[1])
    ax.set_ylim(env.bounds[2], env.bounds[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper left")
    update_title()
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()

    if not selection["completed"]:
        return None, None
    return selection["start"], selection["goal"]


def recompute_path_yaw(path):
    """按路径切线重算 yaw，与 oag_hrrt_dwa_demo.py 的后处理一致。"""
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


def finalize_reference_path(result, goal_pose, smoothing_iterations, curvature_enabled):
    """执行与 oag_hrrt_dwa_demo.py 相同的目标点补全和 Chaikin 后处理。"""
    path_x, path_y, path_yaw = result[:3]
    path = [
        Pose(float(x), float(y), float(yaw))
        for x, y, yaw in zip(path_x, path_y, path_yaw)
    ]
    if not path:
        return result

    last = path[-1]
    if math.hypot(goal_pose.x - last.x, goal_pose.y - last.y) < 0.05:
        path[-1] = Pose(last.x, last.y, goal_pose.yaw)
    else:
        path.append(Pose(goal_pose.x, goal_pose.y, goal_pose.yaw))

    iterations = 0 if curvature_enabled else smoothing_iterations
    points = [(pose.x, pose.y) for pose in path]
    for _ in range(max(0, iterations)):
        smoothed = [points[0]]
        for first, second in zip(points[:-1], points[1:]):
            x1, y1 = first
            x2, y2 = second
            smoothed.append((0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2))
            smoothed.append((0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2))
        smoothed.append(points[-1])
        points = smoothed

    finalized = recompute_path_yaw([Pose(x, y, 0.0) for x, y in points])
    finalized[-1] = Pose(finalized[-1].x, finalized[-1].y, goal_pose.yaw)
    return (
        [pose.x for pose in finalized],
        [pose.y for pose in finalized],
        [pose.yaw for pose in finalized],
        [1] * len(finalized),
    )


def main():
    parser = argparse.ArgumentParser(description="OAG-HRRT* 独立全局路径规划器")

    # 输入参数
    parser.add_argument("--map", required=True, help="NPZ 地图文件路径")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="与 oag_hrrt_dwa_demo.py 共用的 YAML 配置文件",
    )
    parser.add_argument("--start-x", type=float, help="起点 X 坐标")
    parser.add_argument("--start-y", type=float, help="起点 Y 坐标")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="起点航向角（度）")
    parser.add_argument("--goal-x", type=float, help="终点 X 坐标")
    parser.add_argument("--goal-y", type=float, help="终点 Y 坐标")
    parser.add_argument("--goal-yaw", type=float, default=0.0, help="终点航向角（度）")

    # 规划器参数
    parser.add_argument("--max-iterations", type=int, default=None, help="最大迭代次数（默认使用 YAML）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认使用 YAML）")
    parser.add_argument("--rectangle-length", type=float, default=None, help="采样矩形长度（默认使用 YAML）")
    parser.add_argument("--rectangle-width", type=float, default=None, help="采样矩形宽度（默认使用 YAML）")
    parser.add_argument("--use-goal-connector", action="store_true", default=None, help="启用目标连接器")
    parser.add_argument("--allow-reverse", action="store_true", default=None, help="允许倒车")

    # 路径优化参数
    parser.add_argument(
        "--optimize-path",
        action="store_true",
        default=None,
        help="启用路径优化 (Shortcut + 曲率平滑)"
    )
    parser.add_argument(
        "--no-optimize-path",
        dest="optimize_path",
        action="store_false",
        default=None,
        help="禁用路径优化"
    )
    parser.add_argument(
        "--shortcut-iterations",
        type=int,
        default=None,
        help="Shortcut 优化迭代次数（默认使用 YAML 配置）",
    )
    parser.add_argument("--disable-curvature-smoothing", action="store_true", help="禁用曲率平滑")
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=None,
        help="关闭曲率平滑时的 Chaikin 平滑次数（默认使用 YAML）",
    )


    # 车辆参数
    parser.add_argument("--front-length", type=float, default=None, help="车辆前悬长度（默认使用 YAML）")
    parser.add_argument("--rear-length", type=float, default=None, help="车辆后悬长度（默认使用 YAML）")
    parser.add_argument("--vehicle-width", type=float, default=None, help="车辆宽度（默认使用 YAML）")
    parser.add_argument("--safety-margin", type=float, default=None, help="安全边距（默认使用 YAML）")

    # 输出参数
    parser.add_argument("--output", help="输出路径到 NPZ 文件（可选）")
    parser.add_argument("--visualize", action="store_true", help="可视化路径")

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        shared_config = yaml.safe_load(config_file) or {}
    planner_config = shared_config.get('planner', {})
    rectangle_config = planner_config.get('rectangle', {})
    geometry_config = shared_config.get('vehicle', {}).get('geometry', {})
    optimization_config = shared_config.get('path_optimization', {})

    args.max_iterations = args.max_iterations if args.max_iterations is not None else planner_config.get('max_iterations', 2500)
    args.seed = args.seed if args.seed is not None else shared_config.get('random_seed', 42)
    args.rectangle_length = args.rectangle_length if args.rectangle_length is not None else rectangle_config.get('length', 30.0)
    args.rectangle_width = args.rectangle_width if args.rectangle_width is not None else rectangle_config.get('width', 22.0)
    args.use_goal_connector = args.use_goal_connector if args.use_goal_connector is not None else planner_config.get('use_goal_connector', False)
    args.allow_reverse = args.allow_reverse if args.allow_reverse is not None else planner_config.get('allow_reverse', False)
    args.optimize_path = args.optimize_path if args.optimize_path is not None else optimization_config.get('enabled', True)
    args.smoothing_iterations = args.smoothing_iterations if args.smoothing_iterations is not None else planner_config.get('smoothing_iterations', 2)
    args.front_length = args.front_length if args.front_length is not None else geometry_config.get('front_length', 3.0)
    args.rear_length = args.rear_length if args.rear_length is not None else geometry_config.get('rear_length', 1.0)
    args.vehicle_width = args.vehicle_width if args.vehicle_width is not None else geometry_config.get('width', 1.6)
    args.safety_margin = args.safety_margin if args.safety_margin is not None else geometry_config.get('safety_margin', 0.18)
    print(f"[配置] 已加载统一 YAML: {args.config}")

    # 加载环境
    print(f"[加载] 地图文件: {args.map}")
    env = load_environment(args.map)

    # 未完整提供坐标时，在 Matplotlib 地图中交互选择起终点。
    coordinate_values = (args.start_x, args.start_y, args.goal_x, args.goal_y)
    interactive_mode = not any(value is not None for value in coordinate_values)
    if all(value is not None for value in coordinate_values):
        start_pose = Pose(
            float(args.start_x),
            float(args.start_y),
            math.radians(float(args.start_yaw)),
        )
        goal_pose = Pose(
            float(args.goal_x),
            float(args.goal_y),
            math.radians(float(args.goal_yaw)),
        )
    elif any(value is not None for value in coordinate_values):
        parser.error("起终点坐标需要同时提供 --start-x/--start-y/--goal-x/--goal-y")
    else:
        print("[交互] 左键拖拽设置起点位姿，右键拖拽设置目标位姿")
        start_pose, goal_pose = select_poses_interactively(env)
        if start_pose is None or goal_pose is None:
            print("[交互] 未确认起终点，已取消规划")
            return 0

    # 创建车辆几何
    vehicle = VehicleGeometry(
        front_length=args.front_length,
        rear_length=args.rear_length,
        width=args.vehicle_width,
        safety_margin=args.safety_margin,
    )

    # 执行规划
    result, planner = plan_global_path(
        start_pose=start_pose,
        goal_pose=goal_pose,
        env=env,
        vehicle=vehicle,
        max_iterations=args.max_iterations,
        seed=args.seed,
        rectangle_length=args.rectangle_length,
        rectangle_width=args.rectangle_width,
        use_goal_connector=args.use_goal_connector,
        allow_reverse=args.allow_reverse,
    )

    if result is None:
        print("[失败] 未找到路径")
        return 1

    path_x, path_y, path_yaw, directions = result

    # 与交互演示程序共用同一份路径优化配置。
    opt_config = load_path_optimization_config(
        config_path=args.config,
        shortcut_iterations=args.shortcut_iterations,
        disable_curvature_smoothing=args.disable_curvature_smoothing,
    )

    # 路径优化
    if args.optimize_path:
        print(f"\n{'='*70}")
        print("[路径优化] 开始优化...")
        print(f"{'='*70}")

        optimized_result = optimize_global_path(
            result=result,
            vehicle=vehicle,
            obstacles=env.obstacles,
            config=opt_config,
            verbose=True,
        )

        if optimized_result is not None:
            path_x, path_y, path_yaw, directions = optimized_result
            print("[路径优化] 优化完成")
        else:
            print("[路径优化] 优化失败，使用原始路径")

    curvature_enabled = bool(
        opt_config.get('curvature_smoothing', {}).get('enabled', True)
    )
    path_x, path_y, path_yaw, directions = finalize_reference_path(
        (path_x, path_y, path_yaw, directions),
        goal_pose=goal_pose,
        smoothing_iterations=args.smoothing_iterations,
        curvature_enabled=curvature_enabled,
    )

    # 保存路径到文件
    if args.output:
        np.savez(
            args.output,
            path_x=np.array(path_x, dtype=float),
            path_y=np.array(path_y, dtype=float),
            path_yaw=np.array(path_yaw, dtype=float),
            directions=np.array(directions, dtype=int),
            start=np.array([start_pose.x, start_pose.y, start_pose.yaw], dtype=float),
            goal=np.array([goal_pose.x, goal_pose.y, goal_pose.yaw], dtype=float),
        )
        print(f"[保存] 路径已保存到: {args.output}")

    # 可视化
    if args.visualize or interactive_mode:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            fig, ax = plt.subplots(figsize=(12, 10))

            # 绘制障碍物
            for obs in env.obstacles:
                circle = Circle(
                    (obs.x, obs.y),
                    obs.radius,
                    facecolor="lightcoral",
                    edgecolor="red",
                    alpha=0.6
                )
                ax.add_patch(circle)

            # 绘制路径
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Global Path')

            # 绘制起点和终点
            ax.plot(start_pose.x, start_pose.y, 'go', markersize=12, label='Start')
            ax.plot(goal_pose.x, goal_pose.y, 'r*', markersize=16, label='Goal')

            # 绘制起点和终点的航向
            arrow_len = 2.0
            ax.arrow(
                start_pose.x, start_pose.y,
                arrow_len * math.cos(start_pose.yaw),
                arrow_len * math.sin(start_pose.yaw),
                head_width=0.5, head_length=0.5,
                fc='green', ec='green'
            )
            ax.arrow(
                goal_pose.x, goal_pose.y,
                arrow_len * math.cos(goal_pose.yaw),
                arrow_len * math.sin(goal_pose.yaw),
                head_width=0.5, head_length=0.5,
                fc='red', ec='red'
            )

            ax.set_xlim(env.bounds[0], env.bounds[1])
            ax.set_ylim(env.bounds[2], env.bounds[3])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('OAG-HRRT* Global Path Planning')
            ax.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("[警告] matplotlib 未安装，跳过可视化")

    print("\n[完成] 路径规划完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
