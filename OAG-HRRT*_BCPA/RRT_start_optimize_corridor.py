"""
OAG-HRRT* 独立全局路径规划器

从 npz 地图文件进行路径规划，支持完整的路径优化流程：
1. RRT* 路径规划
2. Shortcut 优化（减少冗余节点）
3. 曲率平滑（满足阿克曼约束）
4. 窄通道对齐优化（检测并优化圆形障碍物通道）

特性：
- 支持命令行和交互式两种模式
- 可视化显示不同类型的障碍物（普通/检测到的通道/优化后的通道）
- 完整的配置文件支持
- 详细的优化统计信息

作者: Claude Code
日期: 2026-07-20
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
CORRIDOR_REFINER_DIR = os.path.join(
    PROJECT_ROOT, "global_path_planning", "innovation-circle"
)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "oag_hrrt_dwa", "config.yaml")

for path in (INNOVATION_DIR, CORRIDOR_REFINER_DIR, PROJECT_ROOT):
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
from circle_corridor_refiner_enhanced import refine_global_path_corridors


def load_path_optimization_config(
    config_path=DEFAULT_CONFIG_PATH,
    shortcut_iterations=None,
    disable_curvature_smoothing=False,
):
    """读取共享的路径优化参数

    供独立规划器和 ROS 桥接共用，确保配置一致性。

    Args:
        config_path: YAML 配置文件路径
        shortcut_iterations: 覆盖 YAML 中的 Shortcut 迭代次数（可选）
        disable_curvature_smoothing: 是否禁用曲率平滑（可选）

    Returns:
        dict: 路径优化配置字典
    """
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
    # 保留通道和车辆配置，使所有调用 optimize_global_path() 的入口
    # （包括 ROS bridge）共享同一套窄通道检测与修正流程。
    opt_config['_corridor_alignment'] = dict(
        app_config.get('corridor_alignment', {})
    )
    opt_config['_vehicle'] = dict(app_config.get('vehicle', {}))
    return opt_config


def build_corridor_params(corridor_config, vehicle_config, vehicle):
    """把共享 YAML 转换为通道修正器参数。"""
    search_cfg = corridor_config.get('search', {})
    adjustment_cfg = corridor_config.get('adjustment', {})
    alignment_cfg = corridor_config.get('alignment', {})
    extension_cfg = corridor_config.get('extension', {})
    curvature_cfg = corridor_config.get('curvature', {})
    smoothing_cfg = corridor_config.get('smoothing', {})
    projection_cfg = corridor_config.get('projection', {})
    return {
        'min_turning_radius': float(vehicle_config.get('min_turning_radius', 3.0)),
        'vehicle_front_length': float(vehicle.front_length),
        'vehicle_rear_length': float(vehicle.rear_length),
        'vehicle_width': float(vehicle.width),
        'vehicle_safety_margin': float(vehicle.safety_margin),
        'adjust_before': float(adjustment_cfg.get('before', 5.0)),
        'adjust_after': float(adjustment_cfg.get('after', 5.0)),
        'safe_margin': float(adjustment_cfg.get('safe_margin', 0.0)),
        'enforce_entry_alignment': bool(alignment_cfg.get('enabled', True)),
        'alignment_clearance': float(alignment_cfg.get('clearance', vehicle.front_length)),
        'alignment_yaw_tolerance': math.radians(float(alignment_cfg.get('yaw_tolerance_deg', 3.0))),
        'max_extension_distance': float(extension_cfg.get('max_distance', 20.0)),
        'extension_step': float(extension_cfg.get('step', 0.5)),
        'curvature_tolerance': float(curvature_cfg.get('tolerance', 0.001)),
        'curvature_iterations': int(curvature_cfg.get('max_iterations', 1000)),
        'curvature_relaxation': float(curvature_cfg.get('relaxation', 0.35)),
        'enforce_global_curvature': bool(curvature_cfg.get('enforce_global', True)),
        'num_connection_points': int(smoothing_cfg.get('connection_points', 20)),
        'global_smooth': bool(smoothing_cfg.get('global_enabled', True)),
        'smooth_factor': float(smoothing_cfg.get('smooth_factor', 0.3)),
        'spline_smoothing': float(smoothing_cfg.get('spline_smoothing', 0.1)),
        'path_resample_spacing': float(smoothing_cfg.get('path_resample_spacing', 0.2)),
        'max_candidate_gap_width_factor': float(search_cfg.get('max_gap_width_factor', 3.0)),
        'pair_search_distance': float(search_cfg.get('pair_search_distance', 5.0)),
        'use_projection_method': bool(
            projection_cfg.get('use_projection_method', True)
        ),
        'projection_connect_only': bool(
            projection_cfg.get('connect_only', False)
        ),
        'projection_extension_margin': float(
            projection_cfg.get('extension_margin', 1.0)
        ),
        'projection_search_step': float(
            projection_cfg.get('search_step', 0.5)
        ),
        'projection_max_search_distance': float(
            projection_cfg.get('max_search_distance', 20.0)
        ),
        'projection_skip_curvature_rate_check': bool(
            projection_cfg.get('skip_curvature_rate_check', True)
        ),
        'projection_relaxed_curvature_check': bool(
            projection_cfg.get('relaxed_curvature_check', True)
        ),
        'projection_skip_alignment_check': bool(
            projection_cfg.get('skip_alignment_check', False)
        ),
    }


def create_curvature_smoother(smooth_cfg, checker, vehicle, verbose):
    """创建首轮和通道后终轮共用的曲率平滑器。"""
    max_curvature = smooth_cfg.get(
        'max_curvature',
        math.tan(math.radians(30.0)) / max(vehicle.front_length, 1e-6),
    )
    return CurvatureSmoother(
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
        verbose=verbose,
    )


def shortcut_after_corridors(points, corridor_report, checker, opt_config):
    """对通道处理后的路径再次 Shortcut，并锁定投影中心段。"""
    points = [tuple(point) for point in points]
    if len(points) < 3:
        return points

    intervals = []
    for pair_info in corridor_report.get('pairs', []):
        if not pair_info.get('valid', False):
            continue
        entry = pair_info.get('corridor_entry')
        exit_point = pair_info.get('corridor_exit')
        if entry is None or exit_point is None:
            continue
        entry_index = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - entry[0], points[index][1] - entry[1]
            ),
        )
        exit_index = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - exit_point[0],
                points[index][1] - exit_point[1],
            ),
        )
        intervals.append(tuple(sorted((entry_index, exit_index))))

    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    def optimize_segment(segment):
        if len(segment) < 3:
            return list(segment)
        post_optimizer = ShortcutOptimizer(
            collision_checker=checker,
            max_iterations=opt_config['max_iterations'],
            min_points_distance=opt_config['min_points_distance'],
            enable_angle_filter=opt_config['enable_angle_filter'],
            angle_threshold=opt_config['angle_threshold'],
            random_seed=opt_config['random_seed'],
            verbose=False,
        )
        return post_optimizer.optimize(list(segment))

    # 没有成功投影段时，对整个通道处理结果执行一次 Shortcut。
    if not merged:
        return optimize_segment(points)

    output = []

    def append_unique(segment):
        for point in segment:
            point = tuple(point)
            if not output or math.hypot(
                    output[-1][0] - point[0], output[-1][1] - point[1]
            ) > 1e-8:
                output.append(point)

    cursor = 0
    for start, end in merged:
        append_unique(optimize_segment(points[cursor:start + 1]))
        append_unique(points[start:end + 1])
        cursor = end
    append_unique(optimize_segment(points[cursor:]))
    return output


class PathCollisionChecker:
    """路径优化的碰撞检测器

    检查两点之间的直线路径是否与障碍物碰撞。
    用于 Shortcut 优化器判断是否可以跳过中间节点。
    """

    def __init__(self, vehicle, obstacles, resolution=0.1):
        """初始化碰撞检测器

        Args:
            vehicle: VehicleGeometry 车辆几何参数
            obstacles: List[CircleObstacle] 障碍物列表
            resolution: 碰撞检测采样分辨率（米），默认 0.1m
        """
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.resolution = resolution

    def check_line(self, p1, p2):
        """检查从 p1 到 p2 的直线路径是否无碰撞

        沿直线均匀采样，检查每个采样点的车辆位姿是否与障碍物发生碰撞。

        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)

        Returns:
            bool: True 表示无碰撞（路径可行），False 表示有碰撞（路径不可行）
        """
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_samples = max(2, int(distance / self.resolution))

        # 计算路径方向角（yaw）
        yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # 沿着直线路径采样检查碰撞
        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            pose = Pose(x, y, yaw)

            # 检查该位姿是否与任何障碍物碰撞
            if check_pose_collision(pose, self.vehicle, self.obstacles):
                return False  # 发现碰撞

        return True  # 全程无碰撞


def optimize_global_path(
    result,
    vehicle,
    obstacles,
    config=None,
    verbose=True,
):
    """优化 RRT 生成的全局路径

    使用 Path Shortcut 算法减少路径中的冗余节点，并可选地进行曲率平滑。

    优化流程：
    1. Shortcut 优化：尝试跳过中间节点，用直线连接更远的点
    2. 曲率平滑：对折线拐角进行插补，满足阿克曼底盘的最小转弯半径约束
    3. 速度规划：根据曲率生成速度前瞻曲线

    Args:
        result: RRT 规划结果 (path_x, path_y, path_yaw, directions)
        vehicle: VehicleGeometry 车辆几何参数
        obstacles: List[CircleObstacle] 障碍物列表
        config: 优化配置字典，如果为 None 则使用默认参数
        verbose: 是否打印详细信息

    Returns:
        tuple: 优化后的路径 (opt_x, opt_y, opt_yaw, opt_directions)，格式与输入相同
               如果优化失败则返回 None
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

    # 5. 曲率约束平滑：自动检测折线拐角，执行曲率约束插补并生成速度前瞻曲线
    smooth_cfg = dict(opt_config['curvature_smoothing'])
    # 全局规划器、通道修正器和曲率平滑器统一使用车辆最小转弯半径。
    # 共享配置存在时不再采用可能过期的独立 max_curvature 数值。
    vehicle_cfg = config.get('_vehicle', {})
    if 'min_turning_radius' in vehicle_cfg:
        min_turning_radius = float(vehicle_cfg['min_turning_radius'])
        if min_turning_radius <= 0.0:
            raise ValueError("vehicle.min_turning_radius must be positive")
        smooth_cfg['max_curvature'] = 1.0 / min_turning_radius
        if verbose:
            print(
                f"[曲率约束] 最小转弯半径={min_turning_radius:.2f}m, "
                f"最大曲率={smooth_cfg['max_curvature']:.4f} 1/m"
            )
    corridor_config = config.get('_corridor_alignment', {})
    if (smooth_cfg.get('enabled', True)
            and len(optimized_points) >= 3
            and not corridor_config.get('enabled', False)):
        smoother = create_curvature_smoother(
            smooth_cfg, checker, vehicle, opt_config['verbose']
        )
        smooth_result = smoother.smooth(optimized_points)
        optimized_points = smooth_result.points
        if verbose:
            print(
                f"[曲率平滑] 拐角={len(smooth_result.corner_indices)}, "
                f"插补点={len(smooth_result.points)}, "
                f"最大曲率={max(map(abs, smooth_result.curvatures), default=0.0):.4f} 1/m, "
                f"最高前瞻速度={max(smooth_result.speeds, default=0.0):.2f} m/s"
            )

    # 通道检测放在公共优化入口中。ROS bridge 已调用本函数，因此无需改动
    # bridge 也能获得与独立脚本一致的窄通道检测和对齐结果。
    if corridor_config.get('enabled', False) and len(optimized_points) >= 2:
        corridor_input = np.column_stack((
            [point[0] for point in optimized_points],
            [point[1] for point in optimized_points],
            np.zeros(len(optimized_points), dtype=float),
        ))
        if len(corridor_input) > 1:
            corridor_input[:-1, 2] = np.arctan2(
                np.diff(corridor_input[:, 1]), np.diff(corridor_input[:, 0])
            )
            corridor_input[-1, 2] = corridor_input[-2, 2]
        corridor_path, corridor_report = refine_global_path_corridors(
            corridor_input.copy(),
            obstacles,
            params=build_corridor_params(
                corridor_config, config.get('_vehicle', {}), vehicle
            ),
        )
        optimized_points = [tuple(point) for point in corridor_path[:, :2]]
        # 供独立脚本打印报告和绘制优化前后对比；普通调用者可忽略。
        config['_corridor_applied'] = True
        config['_corridor_report'] = corridor_report
        config['_corridor_input_path'] = corridor_input
        if verbose:
            print(
                f"[通道检测] 候选圆对={corridor_report['candidate_pair_count']}, "
                f"成功优化={corridor_report['optimized_pair_count']}"
            )

        post_shortcut_input_count = len(optimized_points)
        optimized_points = shortcut_after_corridors(
            optimized_points,
            corridor_report,
            checker,
            opt_config,
        )
        if verbose:
            print(
                f"[通道后Shortcut] {post_shortcut_input_count} "
                f"-> {len(optimized_points)} 点（投影中心段已锁定）"
            )

        # 所有结构修改完成后统一执行最终曲率平滑。
        if (smooth_cfg.get('enabled', True)
                and len(optimized_points) >= 3):
            final_smoother = create_curvature_smoother(
                smooth_cfg, checker, vehicle, opt_config['verbose']
            )
            # 通道修正器会生成非常密集的连接点；若直接检测，相邻点的
            # 单步转角可能低于阈值，即使整体曲率已经严重超限。先统一
            # 到配置采样间距，再合并连续的小角度折线，使圆角器能够
            # 使用足够长的入口/出口线段满足最小转弯半径。
            final_input_points = final_smoother._resample(optimized_points)
            final_geometry_filter = ShortcutOptimizer(
                collision_checker=checker,
                max_iterations=0,
                min_points_distance=0.0,
                enable_angle_filter=True,
                angle_threshold=max(
                    final_smoother.corner_threshold,
                    opt_config['angle_threshold'],
                ),
                random_seed=opt_config['random_seed'],
                verbose=False,
            )
            final_input_points = final_geometry_filter.optimize(
                final_input_points
            )
            final_smooth_result = final_smoother.smooth(final_input_points)
            optimized_points = final_smooth_result.points
            if verbose:
                print(
                    f"[通道后平滑] 拐角={len(final_smooth_result.corner_indices)}, "
                    f"插补点={len(final_smooth_result.points)}, "
                    f"最大曲率={max(map(abs, final_smooth_result.curvatures), default=0.0):.4f} 1/m"
                )

    optimized_count = len(optimized_points)

    if verbose:
        print(f"[路径优化] 优化后路径节点数: {optimized_count}")
        print(f"[路径优化] 减少节点数: {original_count - optimized_count}")
        reduction_ratio = (1 - optimized_count/original_count)*100 if original_count > 0 else 0.0
        print(f"[路径优化] 减少比例: {reduction_ratio:.2f}%")
        print(f"{'='*70}\n")

        # 打印详细统计
        optimizer.print_stats()

    # 6. 转换回 RRT 路径格式
    opt_x = [p[0] for p in optimized_points]
    opt_y = [p[1] for p in optimized_points]

    # 重新计算 yaw（航向角）
    opt_yaw = []
    for i in range(len(optimized_points)):
        if i < len(optimized_points) - 1:
            yaw = math.atan2(
                optimized_points[i+1][1] - optimized_points[i][1],
                optimized_points[i+1][0] - optimized_points[i][0]
            )
        else:
            # 最后一个点沿用前一个点的航向
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
    wheel_base=0.9,
    min_turning_radius=3.0,
):
    """使用 OAG-HRRT* 规划全局路径

    OAG-HRRT* = Obstacle-Aware Goal-biased Hybrid RRT*
    - Obstacle-Aware: 障碍物感知的采样策略
    - Goal-biased: 目标导向的采样偏置
    - Hybrid: 混合采样（目标区域 + 走廊 + 矩形区域）
    - RRT*: 渐进最优的快速搜索树

    Args:
        start_pose: Pose 起始位姿 (x, y, yaw)
        goal_pose: Pose 目标位姿 (x, y, yaw)
        env: 环境对象（从 npz 加载），包含障碍物和边界
        vehicle: VehicleGeometry 车辆几何参数
        max_iterations: 最大迭代次数，默认 2500
        seed: 随机种子，用于复现结果
        rectangle_length: 采样矩形长度（米）
        rectangle_width: 采样矩形宽度（米）
        use_goal_connector: 是否使用目标连接器（Reeds-Shepp 曲线）
        allow_reverse: 是否允许倒车
        wheel_base: 车辆轴距（米）
        min_turning_radius: 最小转弯半径（米）

    Returns:
        tuple: (result, planner)
            - result: (path_x, path_y, path_yaw, directions) 或 None（规划失败）
            - planner: AckermannRRTStar 规划器对象，包含规划统计信息
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

    # 全局规划、后处理和车辆控制统一使用同一个最小转弯半径。
    if wheel_base <= 0.0 or min_turning_radius <= 0.0:
        raise ValueError("wheel_base and min_turning_radius must be positive")
    max_steer = math.atan(wheel_base / min_turning_radius)
    curvature = 1.0 / min_turning_radius

    print(f"\n{'='*70}")
    print(f"[规划] OAG-HRRT* 全局路径规划")
    print(f"[规划] 起点: ({start_pose.x:.2f}, {start_pose.y:.2f}, {math.degrees(start_pose.yaw):.1f}°)")
    print(f"[规划] 终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {math.degrees(goal_pose.yaw):.1f}°)")
    print(f"[规划] 最大迭代: {max_iterations}")
    print(f"[规划] 障碍物数: {len(env.obstacles)}")
    print(f"[规划] 轴距: {wheel_base:.2f}m")
    print(f"[规划] 最小转弯半径: {min_turning_radius:.2f}m")
    print(f"[规划] 等效最大转角: {math.degrees(max_steer):.2f}°")
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


def select_poses_interactively(env, yaw_line_length=4.0):
    """空格切换起/终点，鼠标左键拖拽设置所选位姿及航向。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 10))
    selection = {
        "start": None,
        "goal": None,
        "drag_start": None,
        "drag_button": None,
        "completed": False,
        "active_pose": "start",
    }

    # 绘制障碍物 - 在交互选择界面中也显示不同颜色
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

    def set_pose_indicator(point_artist, heading_artist, pose):
        point_artist.set_data([pose.x], [pose.y])
        heading_artist.set_data(
            [pose.x, pose.x + yaw_line_length * math.cos(pose.yaw)],
            [pose.y, pose.y + yaw_line_length * math.sin(pose.yaw)],
        )

    def update_title(message=None):
        if message is None:
            active = "START" if selection["active_pose"] == "start" else "GOAL"
            message = (
                f"Current: {active} | Space: switch start/goal | "
                "Left-drag: set pose+yaw | R: reset | Esc: cancel"
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
            "active_pose": "start",
        })
        start_point.set_data([], [])
        goal_point.set_data([], [])
        start_heading.set_data([], [])
        goal_heading.set_data([], [])
        update_title()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        selection["drag_start"] = (float(event.xdata), float(event.ydata))
        selection["drag_button"] = event.button
        pose_name = selection["active_pose"]
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
        pose_name = selection["active_pose"]
        yaw = math.atan2(event.ydata - sy, event.xdata - sx)
        if math.hypot(event.xdata - sx, event.ydata - sy) < 0.2:
            if pose_name == "start":
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
        if pose_name == "start":
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
            update_title()

    def on_key(event):
        if event.key in (" ", "space"):
            selection["active_pose"] = (
                "goal" if selection["active_pose"] == "start" else "start"
            )
            update_title()
        elif event.key in ("r", "R"):
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


def extract_corridor_info(corridor_report, obstacles):
    """从通道优化报告中提取障碍物索引信息

    Args:
        corridor_report: 通道优化报告字典
        obstacles: 障碍物列表

    Returns:
        tuple: (all_corridor_indices, optimized_corridor_indices)
            - all_corridor_indices: set，所有参与通道检测的障碍物索引
            - optimized_corridor_indices: set，成功优化的通道障碍物索引
    """
    all_indices = set()
    optimized_indices = set()

    if corridor_report is None:
        return all_indices, optimized_indices

    for pair_info in corridor_report.get('pairs', []):
        obs_indices = pair_info.get('obstacle_indices', ())
        if len(obs_indices) == 2:
            all_indices.update(obs_indices)
            if pair_info.get('valid', False):
                optimized_indices.update(obs_indices)

    return all_indices, optimized_indices


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
    """执行与 oag_hrrt_dwa_demo.py 相同的目标点补全和 Chaikin 后处理

    Args:
        result: 路径结果 (path_x, path_y, path_yaw, directions)
        goal_pose: 目标位姿
        smoothing_iterations: Chaikin 平滑迭代次数（仅在曲率平滑禁用时使用）
        curvature_enabled: 是否启用了曲率平滑

    Returns:
        tuple: 最终处理后的路径 (path_x, path_y, path_yaw, directions)
    """
    path_x, path_y, path_yaw = result[:3]
    path = [
        Pose(float(x), float(y), float(yaw))
        for x, y, yaw in zip(path_x, path_y, path_yaw)
    ]
    if not path:
        return result

    # 确保终点与目标点对齐
    last = path[-1]
    distance_to_goal = math.hypot(goal_pose.x - last.x, goal_pose.y - last.y)
    if distance_to_goal < 0.05:
        # 距离很近，直接替换航向
        path[-1] = Pose(last.x, last.y, goal_pose.yaw)
    else:
        # 距离较远，添加目标点
        path.append(Pose(goal_pose.x, goal_pose.y, goal_pose.yaw))

    # Chaikin 平滑（仅在曲率平滑禁用时使用）
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

    # 重新计算航向并保持终点航向
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
    vehicle_config = shared_config.get('vehicle', {})
    optimization_config = shared_config.get('path_optimization', {})
    corridor_config = shared_config.get('corridor_alignment', {})
    corridor_visual_cfg = corridor_config.get('visualization', {})

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
        start_pose, goal_pose = select_poses_interactively(
            env,
            yaw_line_length=float(
                corridor_visual_cfg.get('yaw_line_length', 4.0)
            ),
        )
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
        wheel_base=float(vehicle_config.get('wheel_base', 0.9)),
        min_turning_radius=float(
            vehicle_config.get('min_turning_radius', 3.0)
        ),
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

    # 保存窄通道优化前轨迹，用于与优化结果叠加对比。
    corridor_input_path = opt_config.get(
        '_corridor_input_path',
        np.column_stack((path_x, path_y, path_yaw)),
    )

    # 沿最终全局路径搜索圆形障碍物通道，并在进入前完成方向对齐。
    corridor_report = opt_config.get('_corridor_report')
    if (corridor_config.get('enabled', True)
            and not opt_config.get('_corridor_applied', False)):
        search_cfg = corridor_config.get('search', {})
        adjustment_cfg = corridor_config.get('adjustment', {})
        alignment_cfg = corridor_config.get('alignment', {})
        extension_cfg = corridor_config.get('extension', {})
        curvature_cfg = corridor_config.get('curvature', {})
        smoothing_cfg = corridor_config.get('smoothing', {})
        corridor_params = {
            'min_turning_radius': float(
                vehicle_config.get('min_turning_radius', 3.0)
            ),
            'vehicle_front_length': float(args.front_length),
            'vehicle_rear_length': float(args.rear_length),
            'vehicle_width': float(args.vehicle_width),
            'vehicle_safety_margin': float(args.safety_margin),
            'adjust_before': float(adjustment_cfg.get('before', 5.0)),
            'adjust_after': float(adjustment_cfg.get('after', 5.0)),
            'safe_margin': float(adjustment_cfg.get('safe_margin', 0.0)),
            'enforce_entry_alignment': bool(
                alignment_cfg.get('enabled', True)
            ),
            'alignment_clearance': float(
                alignment_cfg.get('clearance', args.front_length)
            ),
            'alignment_yaw_tolerance': math.radians(float(
                alignment_cfg.get('yaw_tolerance_deg', 3.0)
            )),
            'max_extension_distance': float(
                extension_cfg.get('max_distance', 20.0)
            ),
            'extension_step': float(
                extension_cfg.get('step', 0.5)
            ),
            'curvature_tolerance': float(
                curvature_cfg.get('tolerance', 0.001)
            ),
            'curvature_iterations': int(
                curvature_cfg.get('max_iterations', 1000)
            ),
            'curvature_relaxation': float(
                curvature_cfg.get('relaxation', 0.35)
            ),
            'enforce_global_curvature': bool(
                curvature_cfg.get('enforce_global', True)
            ),
            'num_connection_points': int(
                smoothing_cfg.get('connection_points', 20)
            ),
            'global_smooth': bool(
                smoothing_cfg.get('global_enabled', True)
            ),
            'smooth_factor': float(
                smoothing_cfg.get('smooth_factor', 0.3)
            ),
            'spline_smoothing': float(
                smoothing_cfg.get('spline_smoothing', 0.1)
            ),
            'path_resample_spacing': float(
                smoothing_cfg.get('path_resample_spacing', 0.2)
            ),
            'max_candidate_gap_width_factor': float(
                search_cfg.get('max_gap_width_factor', 3.0)
            ),
            'pair_search_distance': float(
                search_cfg.get('pair_search_distance', 5.0)
            ),
        }
        corridor_path = corridor_input_path.copy()
        corridor_path, corridor_report = refine_global_path_corridors(
            corridor_path,
            env.obstacles,
            params=corridor_params,
        )
        path_x = corridor_path[:, 0].tolist()
        path_y = corridor_path[:, 1].tolist()
        path_yaw = corridor_path[:, 2].tolist()
        directions = [1] * len(corridor_path)

        print(f"\n{'='*70}")
        print(
            f"[通道对齐] 候选圆对={corridor_report['candidate_pair_count']}, "
            f"成功优化={corridor_report['optimized_pair_count']}"
        )

        # 打印每个通道对的详细信息
        if corridor_report.get('pairs'):
            print(f"[通道对齐] 检测到的通道详情:")
            for i, pair_info in enumerate(corridor_report['pairs'], 1):
                obs_indices = pair_info.get('obstacle_indices', ())
                if len(obs_indices) == 2:
                    obs1 = env.obstacles[obs_indices[0]]
                    obs2 = env.obstacles[obs_indices[1]]
                    distance = math.hypot(obs2.x - obs1.x, obs2.y - obs1.y)
                    gap_width = distance - obs1.radius - obs2.radius
                    status = "✓ 成功" if pair_info.get('valid', False) else "✗ 失败"
                    print(
                        f"  通道 {i}: 障碍物 [{obs_indices[0]}, {obs_indices[1]}] | "
                        f"净宽={gap_width:.2f}m | {status}"
                    )
                    if not pair_info.get('valid', False) and 'reason' in pair_info:
                        print(f"    失败原因: {pair_info['reason']}")

        print(
            f"[全局曲率] {corridor_report['global_max_curvature_before']:.4f} "
            f"-> {corridor_report['global_max_curvature_after']:.4f} 1/m | "
            f"{corridor_report['global_curvature_reason']}"
        )
        if not corridor_report['valid']:
            print("[通道对齐] 警告：部分候选通道无法满足全部约束")
        print(f"{'='*70}\n")

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

            # 提取通道障碍物信息
            all_corridor_indices, optimized_corridor_indices = extract_corridor_info(
                corridor_report, env.obstacles
            )

            # 绘制障碍物 - 根据是否参与通道优化使用不同颜色
            for idx, obs in enumerate(env.obstacles):
                if idx in optimized_corridor_indices:
                    # 成功优化的窄通道圆 - 使用绿色
                    circle = Circle(
                        (obs.x, obs.y),
                        obs.radius,
                        facecolor="lightgreen",
                        edgecolor="darkgreen",
                        linewidth=2.5,
                        alpha=0.7,
                        label='Optimized corridor' if idx == min(optimized_corridor_indices) else None
                    )
                elif idx in all_corridor_indices:
                    # 检测到但未成功优化的窄通道圆 - 使用橙色
                    circle = Circle(
                        (obs.x, obs.y),
                        obs.radius,
                        facecolor="orange",
                        edgecolor="darkorange",
                        linewidth=2.5,
                        alpha=0.7,
                        label='Detected corridor (not optimized)' if idx == min(all_corridor_indices - optimized_corridor_indices) else None
                    )
                else:
                    # 普通障碍物 - 使用红色
                    circle = Circle(
                        (obs.x, obs.y),
                        obs.radius,
                        facecolor="lightcoral",
                        edgecolor="red",
                        alpha=0.6,
                        label='Regular obstacle' if idx == 0 and not all_corridor_indices else None
                    )
                ax.add_patch(circle)

            # 窄通道优化前后轨迹对比：原轨迹虚线，优化轨迹实线。
            ax.plot(
                corridor_input_path[:, 0],
                corridor_input_path[:, 1],
                linestyle='--',
                color='royalblue',
                linewidth=float(corridor_visual_cfg.get(
                    'unoptimized_line_width', 1.8
                )),
                alpha=0.85,
                label='Before corridor optimization',
            )
            ax.plot(
                path_x,
                path_y,
                linestyle='-',
                color='green',
                linewidth=float(corridor_visual_cfg.get(
                    'optimized_line_width', 2.5
                )),
                label='After corridor optimization',
            )

            # 绘制通道中心线（连接圆对的中心点）
            if corridor_report is not None and corridor_report.get('pairs'):
                for i, pair_info in enumerate(corridor_report['pairs'], 1):
                    obs_indices = pair_info.get('obstacle_indices', ())
                    if len(obs_indices) == 2:
                        obs1 = env.obstacles[obs_indices[0]]
                        obs2 = env.obstacles[obs_indices[1]]
                        is_optimized = pair_info.get('valid', False)
                        line_color = 'darkgreen' if is_optimized else 'darkorange'
                        line_style = '-' if is_optimized else '--'
                        line_alpha = 0.8 if is_optimized else 0.5

                        # 绘制连接线
                        ax.plot(
                            [obs1.x, obs2.x],
                            [obs1.y, obs2.y],
                            color=line_color,
                            linestyle=line_style,
                            linewidth=1.5,
                            alpha=line_alpha,
                        )

                        # 在中点位置添加文本标注
                        mid_x = (obs1.x + obs2.x) / 2
                        mid_y = (obs1.y + obs2.y) / 2
                        distance = math.hypot(obs2.x - obs1.x, obs2.y - obs1.y)
                        gap_width = distance - obs1.radius - obs2.radius
                        status_symbol = '✓' if is_optimized else '✗'

                        ax.text(
                            mid_x, mid_y,
                            f'{status_symbol} C{i}\n{gap_width:.1f}m',
                            color=line_color,
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va='center',
                            bbox=dict(
                                boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=line_color,
                                alpha=0.8
                            )
                        )

            # 绘制起点和终点
            ax.plot(start_pose.x, start_pose.y, 'go', markersize=12, label='Start')
            ax.plot(goal_pose.x, goal_pose.y, 'r*', markersize=16, label='Goal')

            # 绘制起点和终点的航向
            arrow_len = float(corridor_visual_cfg.get('yaw_line_length', 4.0))
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

            # 构建标题，包含通道信息
            title = 'OAG-HRRT* Global Path Planning'
            if corridor_report is not None:
                candidate_count = corridor_report.get('candidate_pair_count', 0)
                optimized_count = corridor_report.get('optimized_pair_count', 0)
                if candidate_count > 0:
                    title += f' | Corridors: {optimized_count}/{candidate_count} optimized'
            title += ' | R: reset | Esc: close'
            ax.set_title(title)

            # 使用handles去重图例
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            result_window = {"reset_requested": False}

            def on_result_key(event):
                if event.key in ("r", "R"):
                    result_window["reset_requested"] = True
                    plt.close(fig)
                elif event.key == "escape":
                    plt.close(fig)

            fig.canvas.mpl_connect("key_press_event", on_result_key)

            plt.tight_layout()
            plt.show()

            if result_window["reset_requested"]:
                # 删除命令行中固定的起终点参数，确保重启后回到交互选点，
                # 而不是使用旧坐标再次自动规划。
                pose_options = {
                    "--start-x", "--start-y", "--start-yaw",
                    "--goal-x", "--goal-y", "--goal-yaw",
                }
                restart_args = []
                skip_next = False
                for argument in sys.argv[1:]:
                    if skip_next:
                        skip_next = False
                        continue
                    if argument in pose_options:
                        skip_next = True
                        continue
                    if any(
                        argument.startswith(option + "=")
                        for option in pose_options
                    ):
                        continue
                    restart_args.append(argument)
                os.execv(
                    sys.executable,
                    [sys.executable, sys.argv[0], *restart_args],
                )

        except ImportError:
            print("[警告] matplotlib 未安装，跳过可视化")

    print("\n[完成] 路径规划完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
