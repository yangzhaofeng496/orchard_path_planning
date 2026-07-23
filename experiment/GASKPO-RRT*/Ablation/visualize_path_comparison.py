"""
可视化三个算法在density=20地图上的路径对比（单张图）

展示 RRT*, RRT*+GoalBias, GASKPO-RRT* 的路径差异
"""
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np

# 添加正确的路径到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../global_path_planning/innovation_sample")))

from experiment_rrt_star import run_once, get_environment


# 算法配置
METHODS = {
    'RRT*': {'label': 'RRT*', 'color': '#9AA0A6', 'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.9},
    'GoalBias': {'label': 'RRT*+GoalBias', 'color': '#4C78A8', 'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.9},
    'Hybrid': {'label': 'GASKPO-RRT* (ours)', 'color': '#E07A5F', 'linewidth': 3.0, 'linestyle': '-', 'alpha': 1.0},
}


def draw_environment(ax, env):
    """绘制环境（障碍物、起点、终点）"""
    # 绘制障碍物
    for obs in env.obstacles:
        circle = Circle(
            (obs.x, obs.y), obs.radius,
            color='#2F4F4F', alpha=0.6, zorder=2
        )
        ax.add_patch(circle)

    # 绘制起点和终点
    ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=14,
            label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=18,
            label='Goal', zorder=5, markeredgecolor='darkred', markeredgewidth=2)

    # 绘制目标区域
    if hasattr(env, 'goal_rectangle') and env.goal_rectangle is not None:
        rect_x = env.goal_pos[0] - env.goal_rectangle.width / 2
        rect_y = env.goal_pos[1] - env.goal_rectangle.length / 2
        rect = Rectangle(
            (rect_x, rect_y),
            env.goal_rectangle.width,
            env.goal_rectangle.length,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            alpha=0.5,
            zorder=2
        )
        ax.add_patch(rect)


def create_path_comparison_figure(density=20, map_seed=0, search_seed=0, max_iterations=1500):
    """
    创建三个算法的路径对比图（所有路径在一张图上）

    Args:
        density: 障碍物密度
        map_seed: 地图种子
        search_seed: 搜索种子
        max_iterations: 最大迭代次数
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 生成环境
    env_type = f"density_{density}_{map_seed}"
    env = get_environment(
        environment_path=None,
        env_type=env_type,
        seed=map_seed,
        rectangle_length=30.0,
        rectangle_width=20.0,
    )

    # 绘制环境
    draw_environment(ax, env)

    results =
    planners = {}
    stats_text = []

    # 运行三个算法
    for method_name in ['RRT*', 'GoalBias', 'Hybrid']:
        print(f"Running {method_name}...")
        metrics, result, planner, start, goal = run_once(
            method=method_name,
            seed=search_seed,
            env_type=env_type,
            environment_path=None,
            rectangle_length=30.0,
            rectangle_width=20.0,
            allow_reverse=False,
            max_iterations=max_iterations,
            environment=env,
        )

        if metrics['success']:
            results[method_name] = result
            planners[method_name] = planner

            # 绘制路径
            path_x, path_y, _, _ = result
            config = METHODS[method_name]
            ax.plot(
                path_x, path_y,
                color=config['color'],
                linewidth=config['linewidth'],
                linestyle=config['linestyle'],
                alpha=config['alpha'],
                label=config['label'],
                zorder=3 if method_name != 'Hybrid' else 4  # GASKPO-RRT*在最上层
            )

            # 收集统计信息
            stats_text.append(
                f"{config['label']}:\n"
                f"  Nodes: {metrics['node_count']:.0f} | "
                f"Time: {metrics['planning_time']:.2f}s | "
                f"Path: {metrics['path_length']:.2f}m"
            )
            print(f"  Success! Nodes: {metrics['node_count']:.0f}, "
                  f"Time: {metrics['planning_time']:.2f}s, "
                  f"Path Length: {metrics['path_length']:.2f}m")
        else:
            print(f"  Failed!")
            stats_text.append(f"{METHODS[method_name]['label']}: Failed")

    # 添加统计信息文本框
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
        f'Path Comparison: RRT* vs RRT*+GoalBias vs GASKPO-RRT*\n'
        f'(Density={density}, Map={map_seed}, Search={search_seed})',
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

    # 添加图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')

    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join(os.path.dirname(__file__), "density_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"path_comparison_d{density}_m{map_seed}_s{search_seed}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存: {output_path}")

    plt.show()


if __name__ == "__main__":
    # 使用density=20的地图
    create_path_comparison_figure(
        density=20,
        map_seed=0,      # 可以尝试 0-4
        search_seed=0,   # 可以尝试 0-9
        max_iterations=1500
    )
