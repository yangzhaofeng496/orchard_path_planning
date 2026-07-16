#!/usr/bin/env python3
"""
从配置文件生成果园环境并保存为 NPZ 文件
"""
import sys
import os

# 独立目录位于项目根目录下；环境生成模块仍复用全局规划公共实现。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
INNOVATION_DIR = os.path.join(
    PROJECT_ROOT, 'global_path_planning', 'innovation_sample'
)
for path in (INNOVATION_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import yaml
from orchard_environment import make_orchard_environment, save_environment


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成果园环境')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(CURRENT_DIR, 'orchard_scene.npz'),
        help='输出 NPZ 文件路径'
    )
    args = parser.parse_args()

    # 加载配置
    print(f"[生成] 加载配置文件: {args.config}")
    config = load_config(args.config)

    # 提取环境配置
    env_cfg = config['environment']['generation']
    obs_cfg = env_cfg['obstacles']

    print(f"[生成] 环境参数:")
    print(f"  - 网格大小: {env_cfg['grid_size']}")
    print(f"  - 单元大小: {env_cfg['cell_size']} 米")
    print(f"  - 障碍物数量: {obs_cfg['count']}")
    print(f"  - 障碍物大小: {obs_cfg['min_size']}-{obs_cfg['max_size']} 单元")
    print(f"  - 障碍物边距: {obs_cfg['margin']} 单元")
    print(f"  - 安全边距: {obs_cfg['safety_margin']} 米")

    # 生成环境
    print("\n[生成] 开始生成环境...")

    # 从 planner 配置中获取矩形参数
    planner_cfg = config['planner']

    environment = make_orchard_environment(
        seed=env_cfg['seed'],
        grid_size=env_cfg['grid_size'],
        cell_size=env_cfg['cell_size'],
        num_obstacles=obs_cfg['count'],
        obstacle_min_size=obs_cfg['min_size'],
        obstacle_max_size=obs_cfg['max_size'],
        obstacle_margin=obs_cfg['margin'],
        obstacle_safety_margin=obs_cfg['safety_margin'],
        rectangle_length=planner_cfg['rectangle']['length'],
        rectangle_width=planner_cfg['rectangle']['width'],
        rectangle_forward_offset=0.0,
    )

    # 保存环境
    output_path = os.path.abspath(args.output)
    print(f"\n[生成] 保存环境到: {output_path}")
    save_environment(environment, output_path)

    print(f"\n✅ 环境生成完成!")
    print(f"   - 障碍物数量: {len(environment.obstacles)}")
    print(f"   - 起点: {environment.start_pos}")
    print(f"   - 终点: {environment.goal_pos}")
    print(f"   - 边界: {environment.bounds}")
    print(f"\n使用方法:")
    print(f"  python oag_hrrt_dwa_demo.py --map {args.output}")


if __name__ == "__main__":
    main()
