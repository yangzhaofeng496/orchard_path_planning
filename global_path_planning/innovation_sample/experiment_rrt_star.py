import sys
import os
import math
import numpy as np
import yaml
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端，兼容性好
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.collections import LineCollection

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负数显示问题

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import (
    VehicleGeometry,
    CircleObstacle,
)

from ackermann_rrt_star import AckermannRRTStar
from hybrid_sampler import SamplingCorridor
from orchard_environment import (
    load_environment,
    make_orchard_environment,
    make_complex_environment,
    make_hybrid_benchmark_environment,
    make_density_environment,
    make_overlap_environment,
    make_gap_environment,
    save_environment,
)

# 导入路径优化器
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from path_optimizer import ShortcutOptimizer


DEFAULT_ENVIRONMENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "orchard_environment.npz",
)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config.yaml",
)


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
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


def optimize_rrt_path(
    result,
    vehicle,
    obstacles,
    config=None,
    verbose=True,
):
    """
    优化 RRT 生成的路径

    使用 Path Shortcut 算法减少路径中的冗余节点

    Args:
        result: RRT 规划结果 (path_x, path_y, path_yaw, directions)
        vehicle: VehicleGeometry 车辆几何参数
        obstacles: List[CircleObstacle] 障碍物列表
        config: 优化配置字典，如果为 None 则使用默认参数
        verbose: 是否打印详细信息

    Returns:
        优化后的路径 (path_x, path_y, path_yaw, directions)
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
        }

    # 1. 将 RRT 路径转换为点列表
    path_x, path_y, path_yaw, directions = result
    path_points = list(zip(path_x, path_y))

    original_count = len(path_points)

    if verbose:
        print(f"\n{'='*70}")
        print(f"[路径优化] 开始优化...")
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

    # 假设优化后的路径都是前进（简化处理）
    # 如果需要保留原始方向信息，需要更复杂的映射
    opt_directions = [1] * len(optimized_points)

    return (opt_x, opt_y, opt_yaw, opt_directions)


def get_environment(
    environment_path,
    env_type,
    seed,
    rectangle_length,
    rectangle_width,
):
    """优先读取NPZ；文件不存在时生成一次并保存。"""
    benchmark_prefix = "hybrid_"
    if env_type.startswith(benchmark_prefix):
        return make_hybrid_benchmark_environment(
            scenario=env_type[len(benchmark_prefix):],
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )

    if env_type.startswith("density_"):
        density_parts = env_type.split("_")
        obstacle_count = int(density_parts[1])
        map_seed = int(density_parts[2]) if len(density_parts) > 2 else seed
        return make_density_environment(
            obstacle_count=obstacle_count,
            seed=map_seed,
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )

    if env_type.startswith("overlap_"):
        overlap_parts = env_type.split("_")
        overlap_percent = int(overlap_parts[1])
        map_seed = int(overlap_parts[2]) if len(overlap_parts) > 2 else seed
        return make_overlap_environment(
            overlap_percent=overlap_percent,
            seed=map_seed,
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )

    if env_type.startswith("gap_"):
        gap_parts = env_type.split("_")
        gap_width = float(gap_parts[1])
        map_seed = int(gap_parts[2]) if len(gap_parts) > 2 else seed
        return make_gap_environment(
            gap_width=gap_width,
            seed=map_seed,
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )

    if environment_path and os.path.exists(environment_path):
        environment = load_environment(environment_path)
        # 允许实验入口覆盖NPZ内保存的矩形尺寸，而不改变障碍物地图。
        environment.goal_rectangle.length = float(rectangle_length)
        environment.goal_rectangle.width = float(rectangle_width)
        return environment

    factory = (
        make_complex_environment
        if env_type == "complex"
        else make_orchard_environment
    )
    environment = factory(
        seed=seed,
        grid_size=90,
        cell_size=1.0,
        rectangle_length=rectangle_length,
        rectangle_width=rectangle_width,
    )
    if environment_path:
        save_environment(environment, environment_path)
    return environment


def draw_dynamic_rectangle(ax, sampler, goal):
    """实时绘制以当前树末端节点为锚点、朝向目标的采样矩形。"""
    rectangle = sampler.goal_rectangle
    if rectangle is None:
        return
    target_x, target_y = sampler.sampling_target
    heading = math.atan2(
        target_y - rectangle.anchor_y,
        target_x - rectangle.anchor_x,
    )
    c, s = math.cos(heading), math.sin(heading)
    x0 = rectangle.forward_offset
    x1 = rectangle.forward_offset + rectangle.length
    half_width = rectangle.width / 2.0
    local_corners = [
        (x0, half_width),
        (x1, half_width),
        (x1, -half_width),
        (x0, -half_width),
    ]
    corners = [
        (
            rectangle.anchor_x + local_x * c - local_y * s,
            rectangle.anchor_y + local_x * s + local_y * c,
        )
        for local_x, local_y in local_corners
    ]
    ax.add_patch(Polygon(
        corners,
        closed=True,
        facecolor="gold",
        edgecolor="darkorange",
        linewidth=2,
        linestyle="--",
        alpha=0.16,
        label=(
            f"动态采样矩形 "
            f"{rectangle.length:.0f}×{rectangle.width:.0f} m"
            + ("（已缩小）" if sampler.rectangle_shrunk else "")
        ),
        zorder=0,
    ))
    ax.plot(
        rectangle.anchor_x,
        rectangle.anchor_y,
        "o",
        color="darkorange",
        markersize=7,
        label="当前末端节点",
        zorder=7,
    )
    if sampler.tangent_guidance.active:
        ax.plot(
            target_x,
            target_y,
            marker="X",
            color="purple",
            markersize=10,
            linestyle="None",
            label=f"切向子目标 ({sampler.tangent_guidance.side})",
            zorder=8,
        )
        guidance = sampler.tangent_guidance
        ax.add_patch(Ellipse(
            (guidance.cluster_x, guidance.cluster_y),
            width=2.0 * guidance.ellipse_a,
            height=2.0 * guidance.ellipse_b,
            angle=math.degrees(guidance.ellipse_yaw),
            fill=False,
            edgecolor="purple",
            linewidth=2.2,
            linestyle=":",
            label=f"当前阻挡椭圆簇（{len(guidance.obstacle_indexes)}个）",
            zorder=4,
        ))


def calculate_path_length(result):
    x, y, _, _ = result
    length = 0.0
    for i in range(1, len(x)):
        length += math.hypot(x[i] - x[i-1], y[i] - y[i-1])
    return length


def calculate_reverse_length(result):
    x, y, _, direction = result
    length = 0.0
    for i in range(1, len(x)):
        if direction[i] < 0:
            length += math.hypot(x[i] - x[i-1], y[i] - y[i-1])
    return length


def calculate_switch(result):
    _, _, _, direction = result
    count = 0
    for i in range(1, len(direction)):
        if direction[i] != direction[i-1]:
            count += 1
    return count


def simple_sampling_options(method):
    """返回精简、可解释的采样配置。"""
    if method == "Hybrid":
        return {
            "rectangle_anchor_mode": "closest_to_goal",
            "goal_probability": 0.20,
            "corridor_probability": 0.0,
            "rectangle_probability": 0.35,
            "adaptive_sampling_probabilities": False,
            # 果园以圆形树干/树冠障碍为主：仅保留单圆切向绕障。
            "use_tangent_guidance": True,
            "shrink_probability": 0.0,
            "near_anchor_probability": 0.0,
        }
    if method == "GoalBias":
        return {
            "goal_probability": 0.20,
            "corridor_probability": 0.0,
            "rectangle_probability": 0.0,
            "adaptive_sampling_probabilities": False,
            "use_tangent_guidance": False,
            "shrink_probability": 0.0,
            "near_anchor_probability": 0.0,
        }
    return {}


def run_once(
    method,
    seed,
    env_type="simple",
    rectangle_length=30.0,
    rectangle_width=30.0,
    environment_path=DEFAULT_ENVIRONMENT_PATH,
    allow_reverse=False,
    shrink_probability=0.60,
    shrink_length_factor=0.70,
    shrink_width_factor=0.65,
    shrink_activation_distance=20.0,
    near_anchor_probability=0.60,
    near_anchor_length_ratio=0.40,
    safety_margin=0.15,
    environment=None,
    sampling_options=None,
    max_iterations=3000,
    enable_path_optimization=False,
    optimization_config=None,
):
    """
    不带可视化的单次规划（用于快速测试或多次试验）

    新增参数:
        enable_path_optimization: 是否启用路径优化
        optimization_config: 路径优化配置字典
    """
    # 生成环境
    env = environment or get_environment(
        environment_path, env_type, seed, rectangle_length, rectangle_width
    )
    env.goal_rectangle.length = float(rectangle_length)
    env.goal_rectangle.width = float(rectangle_width)
    
    wheel_base = 2.5
    max_steer = math.radians(30)
    curvature = math.tan(max_steer) / wheel_base
    
    start = Pose(env.start_pos[0], env.start_pos[1], 0)
    goal = Pose(env.goal_pos[0], env.goal_pos[1], math.radians(90))
    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=float(safety_margin),
    )
    obstacles = env.obstacles
    
    # 创建走廊对象
    corridors = []
    for corr in env.corridors:
        corridors.append(SamplingCorridor(
            corr['x1'], corr['y1'], corr['x2'], corr['y2'], corr['width']
        ))
    
    use_hybrid = method != "RRT*"
    planner_options = simple_sampling_options(method)
    if sampling_options:
        planner_options.update(sampling_options)
    planner = AckermannRRTStar(
        start, goal, env.bounds, vehicle, obstacles, curvature,
        expand_length=3.0, step_size=0.02, max_iterations=max_iterations, near_radius=5.0,
        use_hybrid_sampling=use_hybrid,
        corridors=corridors if method == "Hybrid" else [],
        goal_rectangle=env.goal_rectangle if method == "Hybrid" else None,
        allow_reverse=allow_reverse,
        shrink_length_factor=shrink_length_factor,
        shrink_width_factor=shrink_width_factor,
        shrink_activation_distance=shrink_activation_distance,
        near_anchor_length_ratio=near_anchor_length_ratio,
        **planner_options,
        random_seed=seed,
    )

    result = planner.planning()
    if result is None:
        return {"success": 0}, None, None, None, None

    # 🆕 路径优化：如果启用，则对 RRT 生成的路径进行优化
    original_result = result
    if enable_path_optimization:
        result = optimize_rrt_path(
            result=result,
            vehicle=vehicle,
            obstacles=obstacles,
            config=optimization_config,
            verbose=False,  # 批量实验时不打印详细信息
        )
        if result is None:
            result = original_result  # 优化失败，使用原始路径

    metrics = planner.get_metrics(original_result)  # 使用原始结果计算节点数等指标
    metrics.update({
        "success": 1,
        "path_length": calculate_path_length(result),  # 使用优化后的路径计算长度
        "reverse_length": calculate_reverse_length(result),
        "switch_count": calculate_switch(result),
    })
    return metrics, result, planner, start, goal


def run_once_with_visualization(
    method,
    seed,
    env_type="hybrid_row_corridor",
    rectangle_length=30.0,
    rectangle_width=30.0,
    environment_path=DEFAULT_ENVIRONMENT_PATH,
    allow_reverse=False,
    shrink_probability=0.60,
    shrink_length_factor=0.70,
    shrink_width_factor=0.65,
    shrink_activation_distance=20.0,
    near_anchor_probability=0.60,
    near_anchor_length_ratio=0.40,
    visualization_interval=10,
    refresh_pause=0.001,
    enable_path_optimization=False,
    optimization_config=None,
):
    """
    带实时可视化的单次规划

    新增参数:
        enable_path_optimization: 是否启用路径优化
        optimization_config: 路径优化配置字典
    """
    env = get_environment(
        environment_path,
        env_type,
        seed,
        rectangle_length,
        rectangle_width,
    )
    obstacles = env.obstacles
    corridors = [SamplingCorridor(c['x1'], c['y1'], c['x2'], c['y2'], c['width']) 
                 for c in env.corridors]
    
    wheel_base = 2.5
    max_steer = math.radians(30)
    curvature = math.tan(max_steer) / wheel_base
    
    start = Pose(env.start_pos[0], env.start_pos[1], 0)
    goal = Pose(env.goal_pos[0], env.goal_pos[1], math.radians(90))
    vehicle = VehicleGeometry(front_length=3.0, rear_length=1.0, width=1.6, safety_margin=0.15)
    
    planner = AckermannRRTStar(
        start, goal, env.bounds, vehicle, obstacles, curvature,
        expand_length=3.0, step_size=0.02, max_iterations=3000, near_radius=5.0,
        use_hybrid_sampling=method != "RRT*", 
        corridors=corridors if method == "Hybrid" else [], 
        goal_rectangle=env.goal_rectangle if method == "Hybrid" else None,
        allow_reverse=allow_reverse,
        shrink_length_factor=shrink_length_factor,
        shrink_width_factor=shrink_width_factor,
        shrink_activation_distance=shrink_activation_distance,
        near_anchor_length_ratio=near_anchor_length_ratio,
        **simple_sampling_options(method),
        random_seed=seed,
    )

    # 创建实时可视化窗口
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.ion()  # 打开交互模式
    
    # 绘制静态元素
    x_min, x_max, y_min, y_max = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if method == "Hybrid":
        draw_dynamic_rectangle(ax, planner.sampler, goal)
    
    # 绘制走廊
    for i, corridor in enumerate(corridors):
        ax.plot([corridor.x1, corridor.x2], [corridor.y1, corridor.y2], 'g--', linewidth=2)
        dx = corridor.x2 - corridor.x1
        dy = corridor.y2 - corridor.y1
        length = math.hypot(dx, dy)
        perp_x = -dy / length * corridor.width / 2
        perp_y = dx / length * corridor.width / 2
        ax.fill([corridor.x1 + perp_x, corridor.x2 + perp_x, corridor.x2 - perp_x, corridor.x1 - perp_x],
                [corridor.y1 + perp_y, corridor.y2 + perp_y, corridor.y2 - perp_y, corridor.y1 - perp_y],
                'green', alpha=0.1)
    
    # 绘制障碍物
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
        ax.add_patch(circle)
    
    # 绘制起点和目标点
    ax.plot(start.x, start.y, 'go', markersize=12, label='Start', zorder=5)
    ax.plot(goal.x, goal.y, 'r*', markersize=20, label='Goal', zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{method} - 规划过程中...', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    
    # 定义回调函数，每次迭代时更新图表
    def visualization_callback(iteration):
        ax.clear()
        
        # 重新绘制基础元素
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if method == "Hybrid":
            draw_dynamic_rectangle(ax, planner.sampler, goal)
        
        # 绘制走廊
        for i, corridor in enumerate(corridors):
            ax.plot([corridor.x1, corridor.x2], [corridor.y1, corridor.y2], 'g--', linewidth=2)
            dx = corridor.x2 - corridor.x1
            dy = corridor.y2 - corridor.y1
            length = math.hypot(dx, dy)
            perp_x = -dy / length * corridor.width / 2
            perp_y = dx / length * corridor.width / 2
            ax.fill([corridor.x1 + perp_x, corridor.x2 + perp_x, corridor.x2 - perp_x, corridor.x1 - perp_x],
                    [corridor.y1 + perp_y, corridor.y2 + perp_y, corridor.y2 - perp_y, corridor.y1 - perp_y],
                    'green', alpha=0.1)
        
        # 绘制障碍物
        for obs in obstacles:
            circle = Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # 批量绘制搜索树，避免为每个节点分别创建Matplotlib对象。
        segments = []
        node_points = []
        for node in planner.nodes:
            node_points.append((node.pose.x, node.pose.y))
            if node.parent is not None:
                parent = planner.nodes[node.parent]
                segments.append([
                    (parent.pose.x, parent.pose.y),
                    (node.pose.x, node.pose.y),
                ])
        if segments:
            ax.add_collection(LineCollection(
                segments, colors='c', linewidths=0.5, alpha=0.5
            ))
        if node_points:
            point_x, point_y = zip(*node_points)
            ax.scatter(point_x, point_y, c='c', s=9, alpha=0.7, linewidths=0)
        
        # 绘制起点和目标点
        ax.plot(start.x, start.y, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(goal.x, goal.y, 'r*', markersize=20, label='Goal', zorder=5)
        
        # 如果已找到目标，绘制当前最优路径
        if planner.goal_index is not None:
            path = planner.extract_path()
            path_x, path_y, _, _ = path
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Current Path', zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{method} - 迭代: {iteration} | 节点数: {len(planner.nodes)} | 最优成本: {planner.best_cost:.2f}', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(max(0.0001, refresh_pause))
    
    # 开始规划，传入回调函数
    result = planner.planning(
        callback=visualization_callback,
        callback_interval=max(1, int(visualization_interval)),
    )
    
    plt.ioff()  # 关闭交互模式

    # 规划完成后保持窗口显示，等待用户按 ESC 关闭
    if result is not None:
        # 🆕 路径优化：如果启用，则对 RRT 生成的路径进行优化
        original_result = result
        if enable_path_optimization:
            print("\n[可视化] 开始路径优化...")
            result = optimize_rrt_path(
                result=result,
                vehicle=vehicle,
                obstacles=obstacles,
                config=optimization_config,
                verbose=True,  # 可视化模式打印详细信息
            )
            if result is None:
                print("[可视化] 路径优化失败，使用原始路径")
                result = original_result
            else:
                print("[可视化] 路径优化完成")

        # 绘制最终优化后的路径
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if method == "Hybrid":
            draw_dynamic_rectangle(ax, planner.sampler, goal)
        
        # 绘制走廊
        for corridor in corridors:
            ax.plot([corridor.x1, corridor.x2], [corridor.y1, corridor.y2], 'g--', linewidth=2)
            dx = corridor.x2 - corridor.x1
            dy = corridor.y2 - corridor.y1
            length = math.hypot(dx, dy)
            perp_x = -dy / length * corridor.width / 2
            perp_y = dx / length * corridor.width / 2
            ax.fill([corridor.x1 + perp_x, corridor.x2 + perp_x, corridor.x2 - perp_x, corridor.x1 - perp_x],
                    [corridor.y1 + perp_y, corridor.y2 + perp_y, corridor.y2 - perp_y, corridor.y1 - perp_y],
                    'green', alpha=0.1)
        
        # 绘制障碍物
        for obs in obstacles:
            circle = Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # 绘制搜索树
        for node in planner.nodes:
            if node.parent is not None:
                parent = planner.nodes[node.parent]
                ax.plot([parent.pose.x, node.pose.x], [parent.pose.y, node.pose.y], 'c-', linewidth=0.5, alpha=0.5)
            ax.plot(node.pose.x, node.pose.y, 'c.', markersize=3, alpha=0.7)
        
        # 绘制起点和目标点
        ax.plot(start.x, start.y, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(goal.x, goal.y, 'r*', markersize=20, label='Goal', zorder=5)
        
        # 绘制最优路径
        path_x, path_y, _, _ = result
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Final Path', zorder=10)

        # 🆕 如果启用了路径优化，同时绘制原始路径用于对比
        if enable_path_optimization and original_result is not None:
            orig_x, orig_y, _, _ = original_result
            ax.plot(orig_x, orig_y, 'gray', linewidth=1.5, alpha=0.5,
                   linestyle='--', label=f'Original Path ({len(orig_x)} points)', zorder=9)
            ax.plot(path_x, path_y, 'b-', linewidth=2.5,
                   label=f'Optimized Path ({len(path_x)} points)', zorder=10)
        else:
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Final Path', zorder=10)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = f'{method} - 规划完成！按 ESC 关闭窗口'
        if enable_path_optimization:
            title += f' (优化: {len(orig_x)} → {len(path_x)} 节点)'
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        
        # 设置按键事件处理器
        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()  # 阻塞直到窗口关闭
    
    if result is None:
        return {"success": 0}, None, None, None, None
    
    metrics = planner.get_metrics(result)
    metrics.update({
        "success": 1,
        "path_length": calculate_path_length(result),
        "reverse_length": calculate_reverse_length(result),
        "switch_count": calculate_switch(result),
    })
    
    return metrics, result, planner, start, goal


def visualize_path(method, result, planner, start, goal, obstacles, corridors):
    """绘制规划的路径"""
    if result is None:
        print(f"{method}: 规划失败")
        return None
    
    path_x, path_y, path_yaw, directions = result
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置坐标范围
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 23)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制走廊
    for i, corridor in enumerate(corridors):
        ax.plot([corridor.x1, corridor.x2], [corridor.y1, corridor.y2], 'g--', linewidth=2, label='Corridor' if i == 0 else '')
        # 绘制走廊宽度范围
        dx = corridor.x2 - corridor.x1
        dy = corridor.y2 - corridor.y1
        length = math.hypot(dx, dy)
        perp_x = -dy / length * corridor.width / 2
        perp_y = dx / length * corridor.width / 2
        ax.fill([corridor.x1 + perp_x, corridor.x2 + perp_x, corridor.x2 - perp_x, corridor.x1 - perp_x],
                [corridor.y1 + perp_y, corridor.y2 + perp_y, corridor.y2 - perp_y, corridor.y1 - perp_y],
                'green', alpha=0.1)
    
    # 绘制障碍物
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
        ax.add_patch(circle)
    
    # 绘制起点和目标点
    ax.plot(start.x, start.y, 'go', markersize=12, label='Start', zorder=5)
    ax.plot(goal.x, goal.y, 'r*', markersize=20, label='Goal', zorder=5)
    
    # 绘制采样的节点（树）
    for node in planner.nodes:
        if node.parent is not None:
            parent = planner.nodes[node.parent]
            ax.plot([parent.pose.x, node.pose.x], [parent.pose.y, node.pose.y], 'c-', linewidth=0.5, alpha=0.3)
        ax.plot(node.pose.x, node.pose.y, 'c.', markersize=2, alpha=0.5)
    
    # 绘制规划的路径
    ax.plot(path_x, path_y, 'b-', linewidth=3, label='Final Path', zorder=10)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{method} - Path Length: {calculate_path_length(result):.2f}m', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    print(f"{method}: 图表已生成，路径长度 = {calculate_path_length(result):.2f}m")
    
    return fig




def run_experiment(
    visualize=True,
    save_figs=True,
    env_type="complex",
    rectangle_length=30.0,
    rectangle_width=30.0,
    environment_path=DEFAULT_ENVIRONMENT_PATH,
    methods=None,
    allow_reverse=False,
    shrink_probability=0.60,
    shrink_length_factor=0.70,
    shrink_width_factor=0.65,
    shrink_activation_distance=20.0,
    near_anchor_probability=0.60,
    near_anchor_length_ratio=0.40,
    visualization_interval=5,
    refresh_pause=0.001,
    enable_path_optimization=False,
    optimization_config=None,
    config_path=None,
):
    """
    运行路径规划实验

    新增参数:
        enable_path_optimization: 是否启用路径优化
        optimization_config: 路径优化配置字典
        config_path: 配置文件路径，如果提供则从文件加载配置
    """
    # 🆕 如果提供了配置文件路径，从配置文件加载参数
    if config_path:
        config = load_config(config_path)
        if config:
            # 环境配置
            env_config = config.get('environment', {})
            env_type = env_config.get('env_type', env_type)
            environment_path = env_config.get('environment_path', environment_path) or environment_path
            rect_config = env_config.get('rectangle', {})
            rectangle_length = rect_config.get('length', rectangle_length)
            rectangle_width = rect_config.get('width', rectangle_width)

            # 规划器配置
            planner_config = config.get('planner', {})
            methods = planner_config.get('methods', methods)
            allow_reverse = planner_config.get('allow_reverse', allow_reverse)

            # 混合采样配置
            sampling_config = config.get('sampling', {})
            shrink_probability = sampling_config.get('shrink_probability', shrink_probability)
            shrink_length_factor = sampling_config.get('shrink_length_factor', shrink_length_factor)
            shrink_width_factor = sampling_config.get('shrink_width_factor', shrink_width_factor)
            shrink_activation_distance = sampling_config.get('shrink_activation_distance', shrink_activation_distance)
            near_anchor_probability = sampling_config.get('near_anchor_probability', near_anchor_probability)
            near_anchor_length_ratio = sampling_config.get('near_anchor_length_ratio', near_anchor_length_ratio)

            # 可视化配置
            viz_config = config.get('visualization', {})
            visualize = viz_config.get('enabled', visualize)
            save_figs = viz_config.get('save_figures', save_figs)
            visualization_interval = viz_config.get('visualization_interval', visualization_interval)
            refresh_pause = viz_config.get('refresh_pause', refresh_pause)

            # 🆕 路径优化配置
            opt_config = config.get('path_optimization', {})
            enable_path_optimization = opt_config.get('enabled', enable_path_optimization)
            if enable_path_optimization:
                optimization_config = opt_config

            print(f"[实验] 从配置文件加载完成")
            if enable_path_optimization:
                print(f"[实验] 路径优化: 已启用")
                print(f"[实验] 优化参数: max_iterations={opt_config.get('max_iterations', 100)}, "
                      f"min_distance={opt_config.get('min_points_distance', 0.2)}m, "
                      f"angle_threshold={opt_config.get('angle_threshold', 15.0)}°")
            else:
                print(f"[实验] 路径优化: 未启用")

    if methods is None:
        methods = ["Hybrid"]

    valid_methods = {"RRT*", "GoalBias", "Hybrid"}
    # valid_methods = {"Hybrid"}
    unknown_methods = set(methods).difference(valid_methods)
    if unknown_methods:
        raise ValueError(
            f"未知规划方法: {', '.join(sorted(unknown_methods))}"
        )
    trials = 1  # 如果要看实时过程，通常只需要 1 次试验
    results = {}
    
    for method in methods:
        print()
        print("=" * 60)
        print(method)
        print("=" * 60)
        data = []
        for i in range(trials):
            print(f"正在执行 {method} 的规划... (请观看规划过程的实时可视化)")
            if visualize:
                metric, result, planner, start, goal = run_once_with_visualization(
                    method,
                    i,
                    env_type=env_type,
                    rectangle_length=rectangle_length,
                    rectangle_width=rectangle_width,
                    environment_path=environment_path,
                    allow_reverse=allow_reverse,
                    shrink_probability=shrink_probability,
                    shrink_length_factor=shrink_length_factor,
                    shrink_width_factor=shrink_width_factor,
                    shrink_activation_distance=shrink_activation_distance,
                    near_anchor_probability=near_anchor_probability,
                    near_anchor_length_ratio=near_anchor_length_ratio,
                    visualization_interval=visualization_interval,
                    refresh_pause=refresh_pause,
                    enable_path_optimization=enable_path_optimization,  # 🆕 传递优化参数
                    optimization_config=optimization_config,
                )
            else:
                metric, result, planner, start, goal = run_once(
                    method,
                    i,
                    env_type=env_type,
                    rectangle_length=rectangle_length,
                    rectangle_width=rectangle_width,
                    environment_path=environment_path,
                    allow_reverse=allow_reverse,
                    shrink_probability=shrink_probability,
                    shrink_length_factor=shrink_length_factor,
                    shrink_width_factor=shrink_width_factor,
                    shrink_activation_distance=shrink_activation_distance,
                    near_anchor_probability=near_anchor_probability,
                    near_anchor_length_ratio=near_anchor_length_ratio,
                    enable_path_optimization=enable_path_optimization,  # 🆕 传递优化参数
                    optimization_config=optimization_config,
                )
            data.append(metric)
        
        results[method] = data
    
    print_result(results)





def mean_value(data, key):
    values = [x[key] for x in data if x["success"]]
    if len(values) == 0:
        return 0
    return np.mean(values)



def success_rate(data):
    return sum(x["success"] for x in data) / len(data) * 100



def print_result(results):
    print()
    print("=" * 70)
    print("实验结果")
    print("=" * 70)
    for method, data in results.items():
        print()
        print(method)
        print(f"成功率: {success_rate(data):.1f}%")
        print(f"平均时间: {mean_value(data, 'planning_time'):.3f}s")
        print(f"平均节点: {mean_value(data, 'node_count'):.1f}")
        print(f"首次找到: {mean_value(data, 'first_solution_iteration'):.1f}")
        print(f"路径长度: {mean_value(data, 'path_length'):.3f}m")
        print(f"倒车距离: {mean_value(data, 'reverse_length'):.3f}m")
        print(f"换向次数: {mean_value(data, 'switch_count'):.2f}")


if __name__ == "__main__":
    # 方式 1: 使用配置文件（推荐）
    # 从 config.yaml 加载所有配置参数
    run_experiment(config_path=DEFAULT_CONFIG_PATH)

    # 方式 2: 手动指定参数（保留原有方式）
    # run_experiment(
    #     visualize=True,
    #     save_figs=False,
    #     env_type="hybrid_staggered_trees",
    #     rectangle_length=30.0,
    #     rectangle_width=12.0,
    #     environment_path=DEFAULT_ENVIRONMENT_PATH,
    #     methods=["GoalBias","Hybrid"],
    #     allow_reverse=True,
    #     shrink_probability=0.60,
    #     shrink_length_factor=0.90,
    #     shrink_width_factor=0.65,
    #     shrink_activation_distance=20.0,
    #     near_anchor_probability=0.60,
    #     near_anchor_length_ratio=0.40,
    #     enable_path_optimization=True,  # 🆕 启用路径优化
    #     optimization_config={            # 🆕 优化配置
    #         'max_iterations': 100,
    #         'min_points_distance': 0.2,
    #         'enable_angle_filter': True,
    #         'angle_threshold': 15.0,
    #         'random_seed': 42,
    #         'verbose': True,
    #         'collision_check_resolution': 0.1,
    #     },
    # )
