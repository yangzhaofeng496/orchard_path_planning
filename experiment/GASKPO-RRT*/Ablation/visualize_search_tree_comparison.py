"""
可视化三个算法在density=20地图上的搜索树和路径对比

展示 RRT*, RRT*+GoalBias, GASKPO-RRT* 的搜索差异
"""
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Ellipse, FancyArrowPatch
import numpy as np
import yaml

# 添加正确的路径到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../global_path_planning/innovation_sample")))

from experiment_rrt_star import run_once
from orchard_environment import make_density_environment


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "search_tree_comparison.yaml",
)


def load_config(config_path=CONFIG_PATH):
    """读取搜索树实验和绘图的统一 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    hybrid = config["sampling"]["hybrid"]
    probability_names = (
        "goal_probability",
        "tangent_probability",
        "rectangle_probability",
        "corridor_probability",
    )
    probabilities = {
        name: float(hybrid[name])
        for name in probability_names
    }
    invalid = {
        name: value
        for name, value in probabilities.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(
            f"采样概率必须位于 0.0–1.0，当前非法配置: {invalid}"
        )
    if sum(probabilities.values()) > 0.80 + 1e-9:
        raise ValueError(
            "Hybrid 引导采样概率之和不能超过 0.80，"
            "需要为全局均匀探索保留至少 20%。"
        )
    return config


CONFIG = load_config()
METHODS = CONFIG["methods"]
DISPLAY = CONFIG["display"]
PLANNER_CONFIG = CONFIG["planner"]
HYBRID_CONFIG = CONFIG["sampling"]["hybrid"]
COMMON_PLANNER_OPTIONS = {
    "use_ackermann_constraints": PLANNER_CONFIG.get(
        "use_ackermann_constraints", True
    ),
    # 兼容旧版 YAML：若公共 planner 中缺失，则读取原 hybrid 配置。
    "use_goal_connector": PLANNER_CONFIG.get(
        "use_goal_connector",
        HYBRID_CONFIG.get("use_goal_connector", False),
    ),
    "goal_connect_distance": PLANNER_CONFIG.get(
        "goal_connect_distance",
        HYBRID_CONFIG.get("goal_connect_distance", 7.0),
    ),
    "relax_goal_yaw": PLANNER_CONFIG.get(
        "relax_goal_yaw",
        HYBRID_CONFIG.get("relax_goal_yaw", False),
    ),
}
SAMPLING_OPTIONS = {
    "RRT*": dict(COMMON_PLANNER_OPTIONS),
    "GoalBias": dict(COMMON_PLANNER_OPTIONS),
    "Hybrid": {
        **COMMON_PLANNER_OPTIONS,
        **HYBRID_CONFIG,
    },
}


def visualize_tree_and_path(ax, planner, result, method_config, show_tree=True):
    """
    在给定的ax上绘制搜索树和路径

    Args:
        ax: matplotlib axes对象
        planner: AckermannRRTStar规划器实例
        result: 路径结果 (path_x, path_y, path_yaw, directions)
        method_config: 方法的可视化配置
        show_tree: 是否显示搜索树
    """
    # 1. 绘制搜索树（如果启用）
    if show_tree and planner is not None:
        for node in planner.nodes[1:]:  # 跳过起点
            parent_idx = node.parent
            if parent_idx is not None:
                parent_node = planner.nodes[parent_idx]
                ax.plot(
                    [parent_node.pose.x, node.pose.x],
                    [parent_node.pose.y, node.pose.y],
                    color=method_config['color'],
                    alpha=method_config['tree_alpha'],
                    linewidth=DISPLAY['tree_linewidth'],
                    linestyle=(
                        0,
                        (
                            DISPLAY['tree_dash_on'],
                            DISPLAY['tree_dash_off'],
                        ),
                    ),
                    dash_capstyle='round',
                    zorder=1
                )

    # 2. 绘制切向引导的触发与调整历史
    if (
        show_tree
        and planner is not None
        and method_config['label'].startswith('GASKPO')
    ):
        sampler = planner.sampler
        guidance_history = getattr(sampler, 'tangent_guidance_history', [])

        # 相同障碍簇附近会反复刷新几乎一致的目标；保留完整日志，
        # 但图中只显示具有可见变化的关键调整，避免箭头和编号堆叠。
        visible_guidance = []
        for event in guidance_history:
            if not visible_guidance:
                visible_guidance.append(event)
                continue
            previous = visible_guidance[-1]
            target_shift = np.hypot(
                event['target_x'] - previous['target_x'],
                event['target_y'] - previous['target_y'],
            )
            if (
                event['side'] != previous['side']
                or event['obstacle_indexes'] != previous['obstacle_indexes']
                or target_shift >= DISPLAY['tangent_history_min_shift']
            ):
                visible_guidance.append(event)

        for index, event in enumerate(visible_guidance, start=1):
            side_color = '#F39C12' if event['side'] == 'left' else '#8E44AD'

            # 当次参与切向计算的膨胀障碍簇。
            ellipse = Ellipse(
                (event['cluster_x'], event['cluster_y']),
                width=(
                    2.0 * event['ellipse_a']
                    * DISPLAY['tangent_ellipse_scale']
                ),
                height=(
                    2.0 * event['ellipse_b']
                    * DISPLAY['tangent_ellipse_scale']
                ),
                angle=np.degrees(event['ellipse_yaw']),
                fill=False,
                edgecolor=side_color,
                linewidth=1.2,
                linestyle=(0, (3, 2)),
                alpha=0.55,
                zorder=2,
            )
            ax.add_patch(ellipse)

            # 锚点 -> 切向子目标，直观展示每次引导如何调整探索方向。
            arrow = FancyArrowPatch(
                (event['anchor_x'], event['anchor_y']),
                (event['target_x'], event['target_y']),
                arrowstyle='-|>',
                mutation_scale=12,
                color=side_color,
                linewidth=DISPLAY['tangent_arrow_linewidth'],
                linestyle='-.',
                alpha=0.90,
                zorder=5,
            )
            ax.add_patch(arrow)
            ax.scatter(
                event['target_x'], event['target_y'],
                s=DISPLAY['tangent_target_size'], marker='D', color=side_color,
                edgecolors='white', linewidths=0.8, zorder=6,
            )
            ax.annotate(
                f'T{index}',
                (event['target_x'], event['target_y']),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, fontweight='bold', color=side_color, zorder=7,
            )

        sample_history = getattr(sampler, 'tangent_sample_history', [])
        if sample_history:
            ax.scatter(
                [sample['x'] for sample in sample_history],
                [sample['y'] for sample in sample_history],
                s=DISPLAY['tangent_sample_size'], marker='x', linewidths=1.3,
                color='#FFD166', zorder=6,
                label='Tangent samples',
            )

    # 3. 绘制最终路径
    if result is not None:
        path_x, path_y, _, _ = result
        ax.plot(
            path_x, path_y,
            color=method_config['path_color'],
            linewidth=DISPLAY['path_linewidth'],
            label=method_config['label'],
            zorder=3
        )


def draw_environment(ax, env):
    """绘制环境（障碍物、起点、终点）"""
    # 绘制障碍物
    for obs in env.obstacles:
        circle = Circle(
            (obs.x, obs.y), obs.radius * DISPLAY['obstacle_scale'],
            color=DISPLAY['obstacle_color'],
            alpha=DISPLAY['obstacle_alpha'],
            zorder=2,
        )
        ax.add_patch(circle)

    # 绘制起点和终点
    ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=12,
            label='Start', zorder=4, markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=16,
            label='Goal', zorder=4, markeredgecolor='darkred', markeredgewidth=1.5)

    # 绘制目标区域
    if hasattr(env, 'goal_rectangle') and env.goal_rectangle is not None:
        rect_x = env.goal_pos[0] - env.goal_rectangle.width / 2
        rect_y = env.goal_pos[1] - env.goal_rectangle.length / 2
        rect = Rectangle(
            (rect_x, rect_y),
            env.goal_rectangle.width,
            env.goal_rectangle.length,
            linewidth=DISPLAY['rectangle_linewidth'],
            edgecolor=DISPLAY['rectangle_color'],
            facecolor='none',
            linestyle='--',
            alpha=DISPLAY['rectangle_alpha'],
            zorder=2
        )
        ax.add_patch(rect)


def style_axis(ax, env, title):
    """统一设置静态坐标轴样式。"""
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(env.bounds[0], env.bounds[1])
    ax.set_ylim(env.bounds[2], env.bounds[3])


def make_live_callback(fig, ax, env, method_name):
    """创建规划迭代回调，在对应子图中实时刷新搜索状态。"""
    method_config = METHODS[method_name]

    def callback_factory(planner):
        def update(iteration):
            ax.clear()
            draw_environment(ax, env)

            current_result = None
            if planner.goal_index is not None:
                current_result = planner.extract_path()

            visualize_tree_and_path(
                ax,
                planner,
                current_result,
                method_config,
                show_tree=True,
            )
            style_axis(
                ax,
                env,
                (
                    f"{method_config['label']} | Iteration: {iteration} | "
                    f"Nodes: {len(planner.nodes)}"
                ),
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(max(0.0001, float(DISPLAY['live_pause'])))

        return update

    return callback_factory


def create_comparison_figure(
    density=20,
    map_seed=0,
    search_seed=0,
    max_iterations=1500,
    rectangle_length=30.0,
    rectangle_width=20.0,
    obstacle_clearance=2.0,
):
    """
    创建三个算法的对比图

    Args:
        density: 障碍物密度
        map_seed: 地图种子
        search_seed: 搜索种子
        max_iterations: 最大迭代次数
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DISPLAY['figure_width'], DISPLAY['figure_height']),
    )
    live_enabled = bool(DISPLAY.get('live_enabled', False))
    if live_enabled:
        plt.ion()

    # 生成环境（只生成一次，三个算法使用相同地图）
    env_type = f"density_{density}_{map_seed}"
    env = make_density_environment(
        obstacle_count=density,
        seed=map_seed,
        rectangle_length=rectangle_length,
        rectangle_width=rectangle_width,
        obstacle_clearance=obstacle_clearance,
    )

    results = {}
    planners = {}

    for ax, method_name in zip(axes, ['RRT*', 'GoalBias', 'Hybrid']):
        draw_environment(ax, env)
        style_axis(ax, env, METHODS[method_name]['label'])
    if live_enabled:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.show(block=False)

    # 运行三个算法
    for ax, method_name in zip(axes, ['RRT*', 'GoalBias', 'Hybrid']):
        print(f"Running {method_name}...")
        callback_factory = (
            make_live_callback(fig, ax, env, method_name)
            if live_enabled else None
        )
        metrics, result, planner, start, goal = run_once(
            method=method_name,
            seed=search_seed,
            env_type=env_type,
            environment_path=None,
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
            allow_reverse=PLANNER_CONFIG["allow_reverse"],
            max_iterations=max_iterations,
            environment=env,  # 传入相同的环境
            sampling_options=SAMPLING_OPTIONS.get(method_name),
            callback_factory=callback_factory,
            callback_interval=DISPLAY.get('live_callback_interval', 5),
        )

        if metrics['success']:
            results[method_name] = result
            planners[method_name] = planner
            print(f"  Success! Nodes: {metrics['node_count']:.0f}, "
                  f"Time: {metrics['planning_time']:.2f}s, "
                  f"Path Length: {metrics['path_length']:.2f}m")
        else:
            print(f"  Failed!")
            results[method_name] = None
            planners[method_name] = None

    # 绘制三个子图
    for ax, method_name in zip(axes, ['RRT*', 'GoalBias', 'Hybrid']):
        # 绘制环境
        draw_environment(ax, env)

        # 绘制搜索树和路径
        if results[method_name] is not None:
            visualize_tree_and_path(
                ax,
                planners[method_name],
                results[method_name],
                METHODS[method_name],
                show_tree=True
            )

            # 添加统计信息
            metrics = planners[method_name].get_metrics(results[method_name])
            sampler_stats = planners[method_name].sampler.get_stats()
            info_text = (
                f"Nodes: {metrics['node_count']:.0f}\n"
                f"Time: {metrics['planning_time']:.2f}s\n"
                f"Path: {metrics['path_length']:.2f}m"
            )
            if method_name == 'Hybrid':
                info_text += (
                    f"\nTangent: {sampler_stats.get('tangent', 0):.0f}"
                    f"\nGoal: {sampler_stats.get('goal', 0):.0f}"
                    f"\nUniform: {sampler_stats.get('uniform', 0):.0f}"
                )
            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

        style_axis(ax, env, METHODS[method_name]['label'])

    plt.suptitle(
        f'Search Tree Comparison (Density={density}, Map={map_seed}, Search={search_seed})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图片
    output_dir = os.path.join(os.path.dirname(__file__), "density_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"tree_comparison_d{density}_m{map_seed}_s{search_seed}.png"
    )
    plt.savefig(
        output_path,
        dpi=DISPLAY['dpi'],
        bbox_inches='tight',
        facecolor='white',
    )
    print(f"\n图片已保存: {output_path}")

    if live_enabled:
        plt.ioff()
    plt.show(block=bool(DISPLAY.get('keep_window_open', True)))


if __name__ == "__main__":
    experiment_config = CONFIG["experiment"]
    map_config = CONFIG["map"]
    create_comparison_figure(
        density=experiment_config["density"],
        map_seed=experiment_config["map_seed"],
        search_seed=experiment_config["search_seed"],
        max_iterations=experiment_config["max_iterations"],
        rectangle_length=map_config["rectangle_length"],
        rectangle_width=map_config["rectangle_width"],
        obstacle_clearance=map_config["obstacle_clearance"],
    )
