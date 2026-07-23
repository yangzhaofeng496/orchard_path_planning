"""
批量生成果园环境地图
"""
import sys
import os
import numpy as np
import argparse
from pathlib import Path

# 添加路径以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from generate_env import generate_random_orchard_env
from global_path_planning.innovation_sample.orchard_environment import (
    save_environment,
    plot_environment
)


def batch_generate_environments(
    num_maps=10,
    base_seed=42,
    num_obstacles_range=(20, 50),
    tree_diameter_range=(1.5, 3.5),
    output_dir='../maps',
    save_images=True
):
    """
    批量生成多个果园环境地图

    Args:
        num_maps: 生成地图数量
        base_seed: 基础随机种子
        num_obstacles_range: 障碍物数量范围 (min, max)
        tree_diameter_range: 果树直径范围（米）
        output_dir: 输出目录
        save_images: 是否保存图片
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"开始批量生成 {num_maps} 张地图...")
    print(f"输出目录: {output_path.absolute()}")
    print(f"障碍物数量范围: {num_obstacles_range[0]} - {num_obstacles_range[1]}")
    print(f"果树直径范围: {tree_diameter_range[0]:.2f}m - {tree_diameter_range[1]:.2f}m")
    print("-" * 60)

    rng = np.random.default_rng(base_seed)
    generated_maps = []

    for i in range(num_maps):
        seed = base_seed + i
        num_obstacles = int(rng.integers(num_obstacles_range[0], num_obstacles_range[1] + 1))

        print(f"\n[{i+1}/{num_maps}] 生成地图 (seed={seed}, obstacles={num_obstacles})...")

        # 生成环境
        env = generate_random_orchard_env(
            seed=seed,
            num_obstacles=num_obstacles,
            tree_diameter_range=tree_diameter_range
        )

        # 保存NPZ文件
        npz_filename = f"map_{i+1:03d}_seed{seed}_obs{len(env.obstacles)}.npz"
        npz_path = output_path / npz_filename
        save_environment(env, str(npz_path))

        # 保存图片
        if save_images:
            image_filename = npz_filename.replace('.npz', '.png')
            image_path = output_path / image_filename
            plot_environment(env, image_path=str(image_path), show=False)

        generated_maps.append({
            'index': i + 1,
            'seed': seed,
            'obstacles': len(env.obstacles),
            'npz_path': str(npz_path),
            'description': env.description
        })

    print("\n" + "=" * 60)
    print(f"批量生成完成！共生成 {len(generated_maps)} 张地图")
    print("=" * 60)

    # 生成索引文件
    index_path = output_path / "maps_index.txt"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(f"# 果园环境地图索引\n")
        f.write(f"# 生成时间: {np.datetime64('now')}\n")
        f.write(f"# 总数量: {len(generated_maps)}\n")
        f.write(f"# 障碍物范围: {num_obstacles_range[0]}-{num_obstacles_range[1]}\n")
        f.write(f"# 果树直径: {tree_diameter_range[0]:.2f}m-{tree_diameter_range[1]:.2f}m\n")
        f.write("\n" + "=" * 60 + "\n\n")

        for map_info in generated_maps:
            f.write(f"[{map_info['index']:03d}] {os.path.basename(map_info['npz_path'])}\n")
            f.write(f"     描述: {map_info['description']}\n")
            f.write(f"     障碍物数: {map_info['obstacles']}\n")
            f.write("\n")

    print(f"\n索引文件已保存: {index_path}")

    return generated_maps


def main():
    parser = argparse.ArgumentParser(description='批量生成果园环境地图')
    parser.add_argument('--num', type=int, default=10, help='生成地图数量')
    parser.add_argument('--base-seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--min-obs', type=int, default=20, help='最小障碍物数量')
    parser.add_argument('--max-obs', type=int, default=50, help='最大障碍物数量')
    parser.add_argument('--min-diameter', type=float, default=1.5, help='最小果树直径(m)')
    parser.add_argument('--max-diameter', type=float, default=3.5, help='最大果树直径(m)')
    parser.add_argument('--output-dir', type=str, default='../map', help='输出目录')
    parser.add_argument('--no-images', action='store_true', help='不保存图片（仅保存NPZ）')

    args = parser.parse_args()

    # 批量生成
    maps = batch_generate_environments(
        num_maps=args.num,
        base_seed=args.base_seed,
        num_obstacles_range=(args.min_obs, args.max_obs),
        tree_diameter_range=(args.min_diameter, args.max_diameter),
        output_dir=args.output_dir,
        save_images=not args.no_images
    )

    print(f"\n所有地图已保存到: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
