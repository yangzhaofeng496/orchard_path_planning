"""
可视化消融实验的搜索树和路径对比

使用与消融实验完全相同的配置：
1. Baseline (RRT*)
2. GoalBias (RRT*+GoalBias)
3. GoalBias+Tangent (GASKPO-RRT*)
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry
from global_path_planning.innovation_sample.ackermann_rrt_star import AckermannRRTStar
from global_path_planning.innovation_sample.orchard_environment import load_environment


# 算法配置（与消融实验完全一致）
CONFIGS = {
    'Baseline': {
        'label': 'RRT*',
        'color': '#5A5A5A',           # 深灰色
        'path_color': '#2C3E50',      # 深蓝灰
        'tree_alpha': 0.25,           # 提高透明度
        'goal_probability': 0.0,
        'tangent_probability': 0.0,
        'use_tangent_guidance': False,
    },
    'GoalBias': {
        'label': 'RRT*+GoalBias',
        'color': '#3498DB',           # 明亮蓝色
        'path_color': '#2471A3',      # 深蓝色
        'tree_alpha': 0.25,
        'goal_probability': 0.20,     # 降低GoalBias的目标采样概率
        'tangent_probability': 0.0,
        'use_tangent_guidance': False,
    },
    'GoalBias+Tangent': {
        'label': 'GASKPO-RRT* (ours)',
        'color': '#E74C3C',           # 明亮红色
        'path_color': '#C0392B',      # 深红色
        'tree_alpha': 0.30,           # GASKPO的树更明显
        'goal_probability': 0.2,     # 提高goal bias，减少初始无效探索
        'tangent_probability': 0.20,  # 提高切向采样概率，减少无效探索
        'use_tangent_guidance': True,
    },
}


def run_planner_with_ablation_config(env, start_pose, goal_pose, vehicle, config, max_iterations=1500, random_seed=None):
    """
    使用消融实验的配置运行规划器

    Args:
        env: 环境对象
        start_pose: 起点位姿
        goal_pose: 终点位姿
        vehicle: 车辆几何参数
        config: 配置字典
        max_iterations: 最大迭代次数
        random_seed: 随机种子

    Returns:
        tuple: (planner, result, metrics)
    """
    # 判断是否启用混合采样
    use_hybrid_sampling = (
        config['goal_probability'] > 0.0 or config['tangent_probability'] > 0.0
    )
    enable_tangent_connectors = config['use_tangent_guidance']

    # 创建规划器（参数与消融实验完全一致，但降低切向采样的分散度）
    planner = AckermannRRTStar(
        start=start_pose,
        goal=goal_pose,
        bounds=env.bounds,
        vehicle=vehicle,
        obstacles=env.obstacles,
        curvature=1.0 / 3.0,  # 最小转弯半径 3m
        use_ackermann_constraints=False,
        expand_length=3.0,
        step_size=0.08,
        max_iterations=max_iterations,
        near_radius=5.0,
        use_hybrid_sampling=use_hybrid_sampling,
        goal_probability=config['goal_probability'],
        tangent_probability=config['tangent_probability'],
        adaptive_sampling_probabilities=False,
        corridor_probability=0.0,
        rectangle_probability=0.45,
        allow_reverse=enable_tangent_connectors,
        use_tangent_guidance=config['use_tangent_guidance'],
        shrink_probability=0.35,
        shrink_length_factor=0.70,
        shrink_width_factor=0.70,
        shrink_activation_distance=18.0,
        near_anchor_probability=0.55,
        near_anchor_length_ratio=0.40,
        cluster_shape="ellipse",
        rectangle_anchor_mode="closest_to_goal",
        use_goal_connector=enable_tangent_connectors,
        relax_goal_yaw=False,
        random_seed=random_seed if random_seed is not None else int(time.time() * 1000) % 100000,
        # 降低切向采样的分散度
        tangent_extension=0.1,        # 从默认0.3减小到0.1，减少切点延伸
        tangent_along_std=0.15,       # 从默认0.30减半到0.15，沿切线方向高斯标准差
        tangent_lateral_std=0.10,     # 从默认0.20减半到0.10，垂直切线方向高斯标准差
    )

    # 运行规划
    result = planner.planning()

    # 计算指标
    if result is None:
        metrics = {
            'success': False,
            'planning_time': 0.0,
            'node_count': 0,
            'path_length': float('inf'),
        }
    else:
        path_x, path_y, _, _ = result
        path_length = sum(
            np.hypot(path_x[i+1] - path_x[i], path_y[i+1] - path_y[i])
            for i in range(len(path_x) - 1)
        )
        metrics = {
            'success': True,
            'planning_time': planner.planning_time if hasattr(planner, 'planning_time') else 0.0,
            'node_count': len(planner.nodes),
            'path_length': path_length,
        }

    return planner, result, metrics


def visualize_tree_and_path(ax, planner, result, config, show_tree=True):
    """绘制搜索树和路径"""
    # 1. 绘制搜索树（使用虚线，更清晰）
    if show_tree and planner is not None and result is not None:
        for node in planner.nodes[1:]:
            parent_idx = node.parent
            if parent_idx is not None:
                parent_node = planner.nodes[parent_idx]
                ax.plot(
                    [parent_node.pose.x, node.pose.x],
                    [parent_node.pose.y, node.pose.y],
                    color=config['color'],
                    alpha=config['tree_alpha'],
                    linewidth=1.2,        # 加粗线宽
                    linestyle='--',       # 使用虚线
                    zorder=1
                )

    # 2. 绘制切向圆（只对GASKPO-RRT*显示）
    if show_tree and planner is not None and config.get('use_tangent_guidance', False):
        # 检查规划器是否有切向圆信息
        if hasattr(planner, 'tangent_manager') and planner.tangent_manager is not None:
            tangent_mgr = planner.tangent_manager
            if hasattr(tangent_mgr, 'tangent_circles') and tangent_mgr.tangent_circles:
                for tc in tangent_mgr.tangent_circles:
                    # 绘制切向圆
                    circle = Circle(
                        (tc.center.x, tc.center.y),
                        tc.radius,
                        color='#FF6B35',      # 亮橙色
                        alpha=0.2,
                        fill=True,
                        linestyle='-',
                        linewidth=2.0,
                        edgecolor='#FF6B35',
                        zorder=2
                    )
                    ax.add_patch(circle)
                    # 标注圆心
                    ax.plot(tc.center.x, tc.center.y, 'o',
                           color='#FF6B35', markersize=6,
                           markeredgecolor='white', markeredgewidth=1,
                           zorder=3, label='Tangent Circle' if tc == tangent_mgr.tangent_circles[0] else '')

    # 3. 绘制最终路径（使用不同颜色，更粗）
    if result is not None:
        path_x, path_y, _, _ = result
        ax.plot(
            path_x, path_y,
            color=config.get('path_color', config['color']),
            linewidth=3.5,
            label=config['label'],
            zorder=4,
            solid_capstyle='round',
            solid_joinstyle='round'
        )


def draw_environment(ax, env, start_pose, goal_pose):
    """绘制环境"""
    # 绘制障碍物（更深的颜色）
    for obs in env.obstacles:
        circle = Circle(
            (obs.x, obs.y), obs.radius,
            color='#34495E', alpha=0.75, zorder=2,
            edgecolor='#2C3E50', linewidth=1.5
        )
        ax.add_patch(circle)

    # 绘制起点和终点（更大更明显）
    ax.plot(start_pose.x, start_pose.y, 'o', color='#27AE60', markersize=15,
            label='Start', zorder=5, markeredgecolor='white', markeredgewidth=2)
    ax.plot(goal_pose.x, goal_pose.y, '*', color='#E74C3C', markersize=20,
            label='Goal', zorder=5, markeredgecolor='white', markeredgewidth=2)


def create_comparison_figure(map_path, pair_index=0, max_iterations=1500, random_seed=0):
    """
    创建三个算法的对比图

    Args:
        map_path: 地图文件路径
        pair_index: 使用哪一对起点终点（0-4）
        max_iterations: 最大迭代次数
        random_seed: 随机种子
    """
    # 加载环境
    env = load_environment(map_path)

    # 获取起点终点
    if env.start_goal_pairs and len(env.start_goal_pairs) > pair_index:
        start, goal = env.start_goal_pairs[pair_index]
        start_pose = Pose(x=start[0], y=start[1], yaw=0.0)
        goal_pose = Pose(x=goal[0], y=goal[1], yaw=0.0)
    else:
        start_pose = Pose(x=env.start_pos[0], y=env.start_pos[1], yaw=0.0)
        goal_pose = Pose(x=env.goal_pos[0], y=env.goal_pos[1], yaw=0.0)

    # 车辆参数
    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=0.15
    )

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    results = {}

    # 运行三个算法
    for ax, (config_name, config) in zip(axes, CONFIGS.items()):
        print(f"Running {config['label']}...")

        planner, result, metrics = run_planner_with_ablation_config(
            env, start_pose, goal_pose, vehicle, config,
            max_iterations=max_iterations,
            random_seed=random_seed
        )

        if metrics['success']:
            print(f"  Success! Nodes: {metrics['node_count']}, "
                  f"Time: {metrics['planning_time']:.2f}s, "
                  f"Path Length: {metrics['path_length']:.2f}m")

            # 绘制环境
            draw_environment(ax, env, start_pose, goal_pose)

            # 绘制搜索树和路径
            visualize_tree_and_path(ax, planner, result, config, show_tree=True)

            # 添加统计信息
            info_text = (
                f"Nodes: {metrics['node_count']}\n"
                f"Time: {metrics['planning_time']:.2f}s\n"
                f"Path: {metrics['path_length']:.2f}m"
            )
            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

            results[config_name] = (planner, result, metrics)
        else:
            print(f"  Failed!")
            draw_environment(ax, env, start_pose, goal_pose)

        # 设置图表样式
        ax.set_title(config['label'], fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray')
        ax.set_xlim(env.bounds[0], env.bounds[1])
        ax.set_ylim(env.bounds[2], env.bounds[3])
        ax.set_facecolor('#FAFAFA')  # 浅灰背景

    plt.suptitle(
        f'Ablation Study Comparison (Pair {pair_index}, Seed {random_seed})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图片
    output_dir = os.path.dirname(__file__)
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    output_path = os.path.join(
        output_dir,
        f"ablation_tree_comparison_{map_name}_pair{pair_index}_seed{random_seed}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存: {output_path}")

    plt.show()

    return results


def create_path_comparison_figure(map_path, pair_index=0, max_iterations=1500, random_seed=0):
    """创建路径对比图（所有路径在一张图上）"""
    # 加载环境
    env = load_environment(map_path)

    # 获取起点终点
    if env.start_goal_pairs and len(env.start_goal_pairs) > pair_index:
        start, goal = env.start_goal_pairs[pair_index]
        start_pose = Pose(x=start[0], y=start[1], yaw=0.0)
        goal_pose = Pose(x=goal[0], y=goal[1], yaw=0.0)
    else:
        start_pose = Pose(x=env.start_pos[0], y=env.start_pos[1], yaw=0.0)
        goal_pose = Pose(x=env.goal_pos[0], y=env.goal_pos[1], yaw=0.0)

    # 车辆参数
    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=0.15
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制环境
    draw_environment(ax, env, start_pose, goal_pose)

    stats_text = []

    # 运行三个算法
    for config_name, config in CONFIGS.items():
        print(f"Running {config['label']}...")

        planner, result, metrics = run_planner_with_ablation_config(
            env, start_pose, goal_pose, vehicle, config,
            max_iterations=max_iterations,
            random_seed=random_seed
        )

        if metrics['success']:
            # 绘制路径
            path_x, path_y, _, _ = result
            linewidth = 3.0 if config_name == 'GoalBias+Tangent' else 2.5
            ax.plot(
                path_x, path_y,
                color=config['color'],
                linewidth=linewidth,
                alpha=0.9,
                label=config['label'],
                zorder=4 if config_name == 'GoalBias+Tangent' else 3
            )

            stats_text.append(
                f"{config['label']}:\n"
                f"  Nodes: {metrics['node_count']} | "
                f"Time: {metrics['planning_time']:.2f}s | "
                f"Path: {metrics['path_length']:.2f}m"
            )
            print(f"  Success! Nodes: {metrics['node_count']}, "
                  f"Time: {metrics['planning_time']:.2f}s, "
                  f"Path Length: {metrics['path_length']:.2f}m")
        else:
            print(f"  Failed!")
            stats_text.append(f"{config['label']}: Failed")

    # 添加统计信息
    info_text = "\n\n".join(stats_text)
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        zorder=6
    )

    # 设置图表样式
    ax.set_title(
        f'Path Comparison: Ablation Study\n(Pair {pair_index}, Seed {random_seed})',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(env.bounds[0], env.bounds[1])
    ax.set_ylim(env.bounds[2], env.bounds[3])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')

    plt.tight_layout()

    # 保存图片
    output_dir = os.path.dirname(__file__)
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    output_path = os.path.join(
        output_dir,
        f"ablation_path_comparison_{map_name}_pair{pair_index}_seed{random_seed}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存: {output_path}")

    plt.show()


if __name__ == "__main__":
    # 使用density=40的对比地图（更密集）
    map_path = os.path.join(os.path.dirname(__file__), "comparison_map_density40.npz")

    # 生成搜索树对比图（3个子图）
    print("=" * 60)
    print("生成搜索树对比图...")
    print("=" * 60)
    create_comparison_figure(
        map_path=map_path,
        pair_index=0,       # 使用第0对起点终点
        max_iterations=1500,
        random_seed=0
    )

    # 生成路径对比图（单张图）
    print("\n" + "=" * 60)
    print("生成路径对比图...")
    print("=" * 60)
    create_path_comparison_figure(
        map_path=map_path,
        pair_index=0,
        max_iterations=1500,
        random_seed=0
    )
