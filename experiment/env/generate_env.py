"""
果园环境生成工具
生成随机果树障碍物环境，用于路径规划实验
"""
import sys
import os
import numpy as np
import argparse

# 添加路径以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vehicle.vehicle_collision import CircleObstacle
from global_path_planning.innovation_sample.orchard_environment import (
    OrchardEnvironment,
    make_goal_rectangle,
    save_environment,
    plot_environment
)


def generate_random_orchard_env(
    seed=42,
    bounds=(0.0, 90.0, 0.0, 90.0),
    num_obstacles=30,
    tree_diameter_range=(1.5, 3.5),
    start_pos=(5.0, 45.0),
    goal_pos=(85.0, 45.0),
    clear_radius=6.0,
    rectangle_length=30.0,
    rectangle_width=20.0,
    max_attempts=10000
):
    """
    生成随机果树障碍物环境

    Args:
        seed: 随机种子
        bounds: 环境边界 (x_min, x_max, y_min, y_max)
        num_obstacles: 障碍物数量
        tree_diameter_range: 果树直径范围（米），转换为半径
        start_pos: 起点坐标 (x, y)
        goal_pos: 终点坐标 (x, y)
        clear_radius: 起点和终点周围的清空半径
        rectangle_length: 目标矩形长度
        rectangle_width: 目标矩形宽度
        max_attempts: 最大尝试次数

    Returns:
        OrchardEnvironment 对象
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds

    obstacles = []
    attempts = 0

    print(f"开始生成环境 (seed={seed})")
    print(f"目标障碍物数量: {num_obstacles}")
    print(f"果树直径范围: {tree_diameter_range[0]:.2f}m - {tree_diameter_range[1]:.2f}m")

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1

        # 随机生成位置
        x = float(rng.uniform(x_min + 2.0, x_max - 2.0))
        y = float(rng.uniform(y_min + 2.0, y_max - 2.0))

        # 随机生成直径，转换为半径
        diameter = float(rng.uniform(tree_diameter_range[0], tree_diameter_range[1]))
        radius = diameter / 2.0

        # 检查是否与起点或终点重叠
        dist_to_start = np.hypot(x - start_pos[0], y - start_pos[1])
        dist_to_goal = np.hypot(x - goal_pos[0], y - goal_pos[1])

        if dist_to_start < clear_radius + radius:
            continue
        if dist_to_goal < clear_radius + radius:
            continue

        # 检查是否与已有障碍物重叠
        overlaps = False
        for existing_obs in obstacles:
            dist = np.hypot(x - existing_obs.x, y - existing_obs.y)
            min_dist = radius + existing_obs.radius
            if dist < min_dist:
                overlaps = True
                break

        if overlaps:
            continue

        # 添加障碍物
        obstacles.append(CircleObstacle(x, y, radius))

        if len(obstacles) % 10 == 0:
            print(f"已生成 {len(obstacles)}/{num_obstacles} 个障碍物...")

    if len(obstacles) < num_obstacles:
        print(f"警告: 只生成了 {len(obstacles)}/{num_obstacles} 个障碍物 (尝试了 {attempts} 次)")
    else:
        print(f"成功生成 {len(obstacles)} 个障碍物 (尝试了 {attempts} 次)")

    # 创建目标矩形
    goal_rectangle = make_goal_rectangle(
        start_pos,
        goal_pos,
        rectangle_length,
        rectangle_width,
        forward_offset=0.0
    )

    # 创建环境对象
    environment = OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=goal_rectangle,
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=f"随机果园环境 (种子{seed}, {len(obstacles)}个果树障碍物)"
    )

    return environment


def main():
    parser = argparse.ArgumentParser(description='生成果园环境')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num', type=int, default=30, help='障碍物数量')
    parser.add_argument('--min-diameter', type=float, default=1.5, help='最小果树直径(m)')
    parser.add_argument('--max-diameter', type=float, default=3.5, help='最大果树直径(m)')
    parser.add_argument('--output', type=str, default='orchard_env.npz', help='输出NPZ文件名')
    parser.add_argument('--plot', action='store_true', help='是否显示环境图')
    parser.add_argument('--save-image', action='store_true', help='是否保存环境图片')

    args = parser.parse_args()

    # 生成环境
    env = generate_random_orchard_env(
        seed=args.seed,
        num_obstacles=args.num,
        tree_diameter_range=(args.min_diameter, args.max_diameter)
    )

    # 保存环境
    output_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(output_dir, args.output)
    save_environment(env, npz_path)

    # 可选：保存和显示图片
    if args.save_image or args.plot:
        image_path = npz_path.replace('.npz', '.png') if args.save_image else None
        plot_environment(env, image_path=image_path, show=args.plot)

    print(f"\n环境信息:")
    print(f"  描述: {env.description}")
    print(f"  起点: {env.start_pos}")
    print(f"  终点: {env.goal_pos}")
    print(f"  障碍物数: {len(env.obstacles)}")
    print(f"  边界: {env.bounds}")


if __name__ == "__main__":
    main()
