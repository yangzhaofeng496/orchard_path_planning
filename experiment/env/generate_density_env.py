#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按照低、中、高三种障碍物密度批量生成随机果园环境。

输出结构：
generated_density_maps/
├── density_low/
├── density_medium/
├── density_high/
└── manifest.csv

特点：
1. 障碍物完全随机分布，不生成规则树墙或走廊。
2. 每张地图随机生成10对起点和终点。
3. 障碍物不会覆盖起点和终点安全区域。
4. 每张地图使用不同随机种子。
5. 自动进行简单栅格连通性检查，过滤明显无解地图。
"""

import argparse
import csv
import math
import os
import secrets
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vehicle.vehicle_collision import CircleObstacle
from global_path_planning.innovation_sample.orchard_environment import (
    OrchardEnvironment,
    make_goal_rectangle,
    plot_environment,
    save_environment,
)


def save_environment_with_pairs(environment, npz_path, start_goal_pairs):
    """
    保存果园环境，包含多对起点终点。

    Args:
        environment: OrchardEnvironment 对象
        npz_path: 输出文件路径
        start_goal_pairs: 起点终点对列表 [(start1, goal1), (start2, goal2), ...]
    """
    npz_path = os.path.abspath(npz_path)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    obstacles = np.asarray(
        [[obs.x, obs.y, obs.radius] for obs in environment.obstacles],
        dtype=float,
    ).reshape(-1, 3)

    # 保存多对起点终点
    start_goal_array = np.asarray(
        [[start[0], start[1], goal[0], goal[1]] for start, goal in start_goal_pairs],
        dtype=float,
    ).reshape(-1, 4)

    rectangle = environment.goal_rectangle

    np.savez_compressed(
        npz_path,
        format_version=np.asarray([2], dtype=np.int32),  # 版本号改为2，标识新格式
        obstacles=obstacles,
        corridors=np.empty((0, 5), dtype=float),  # 空走廊
        start_pos=np.asarray(environment.start_pos, dtype=float),  # 默认起点（向后兼容）
        goal_pos=np.asarray(environment.goal_pos, dtype=float),    # 默认终点（向后兼容）
        start_goal_pairs=start_goal_array,  # 新增：多对起点终点
        bounds=np.asarray(environment.bounds, dtype=float),
        rectangle=np.asarray([
            rectangle.length,
            rectangle.width,
            rectangle.forward_offset,
        ], dtype=float),
        description=np.asarray(environment.description),
    )
    print(f"[环境] NPZ已保存: {npz_path} (包含{len(start_goal_pairs)}对起点终点)")
    return npz_path


def sample_start_goal(rng, bounds, clear_radius, min_distance):
    """随机生成起点和终点，并保证两点距离足够远。"""
    x_min, x_max, y_min, y_max = bounds
    margin = clear_radius + 1.0

    for _ in range(3000):
        start = (
            float(rng.uniform(x_min + margin, x_max - margin)),
            float(rng.uniform(y_min + margin, y_max - margin)),
        )
        goal = (
            float(rng.uniform(x_min + margin, x_max - margin)),
            float(rng.uniform(y_min + margin, y_max - margin)),
        )

        if math.hypot(goal[0] - start[0], goal[1] - start[1]) >= min_distance:
            return start, goal

    raise RuntimeError("无法生成满足最小距离要求的起点和终点")


def can_place_obstacle(obstacles, x, y, radius, bounds, start, goal, clear_radius, overlap_margin):
    """检查障碍物是否可以放置。"""
    x_min, x_max, y_min, y_max = bounds

    if x - radius < x_min + 0.3 or x + radius > x_max - 0.3:
        return False
    if y - radius < y_min + 0.3 or y + radius > y_max - 0.3:
        return False

    if math.hypot(x - start[0], y - start[1]) < radius + clear_radius:
        return False
    if math.hypot(x - goal[0], y - goal[1]) < radius + clear_radius:
        return False

    for obstacle in obstacles:
        distance = math.hypot(x - obstacle.x, y - obstacle.y)
        if distance < radius + obstacle.radius + overlap_margin:
            return False

    return True


def generate_random_obstacles(
    rng,
    obstacle_count,
    bounds,
    diameter_range,
    start,
    goal,
    clear_radius,
    overlap_margin,
    max_attempts=200000,
):
    """在全地图随机生成指定数量的障碍物。"""
    x_min, x_max, y_min, y_max = bounds
    obstacles = []
    attempts = 0

    while len(obstacles) < obstacle_count and attempts < max_attempts:
        attempts += 1

        diameter = float(rng.uniform(diameter_range[0], diameter_range[1]))
        radius = diameter / 2.0
        x = float(rng.uniform(x_min + radius + 0.4, x_max - radius - 0.4))
        y = float(rng.uniform(y_min + radius + 0.4, y_max - radius - 0.4))

        if not can_place_obstacle(
            obstacles, x, y, radius, bounds, start, goal,
            clear_radius, overlap_margin
        ):
            continue

        obstacles.append(CircleObstacle(x, y, radius))

    if len(obstacles) != obstacle_count:
        raise RuntimeError(
            f"无法生成指定数量的障碍物：{len(obstacles)}/{obstacle_count}，"
            f"尝试次数={attempts}"
        )

    return obstacles


def grid_path_exists(bounds, obstacles, start, goal, clearance, resolution=0.5):
    """使用二维栅格检查起点和终点是否连通。"""
    x_min, x_max, y_min, y_max = bounds
    width = int(math.ceil((x_max - x_min) / resolution))
    height = int(math.ceil((y_max - y_min) / resolution))

    rows, cols = np.ogrid[:height, :width]
    grid_x = x_min + (cols + 0.5) * resolution
    grid_y = y_min + (rows + 0.5) * resolution
    occupied = np.zeros((height, width), dtype=bool)

    for obstacle in obstacles:
        radius = obstacle.radius + clearance
        occupied |= (grid_x - obstacle.x) ** 2 + (grid_y - obstacle.y) ** 2 <= radius ** 2

    def position_to_grid(position):
        col = int((position[0] - x_min) / resolution)
        row = int((position[1] - y_min) / resolution)
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        return row, col

    start_cell = position_to_grid(start)
    goal_cell = position_to_grid(goal)

    if occupied[start_cell] or occupied[goal_cell]:
        return False

    queue = deque([start_cell])
    visited = np.zeros_like(occupied)
    visited[start_cell] = True

    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    while queue:
        row, col = queue.popleft()

        if (row, col) == goal_cell:
            return True

        for dr, dc in directions:
            next_row = row + dr
            next_col = col + dc

            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if occupied[next_row, next_col] or visited[next_row, next_col]:
                continue

            visited[next_row, next_col] = True
            queue.append((next_row, next_col))

    return False


def generate_environment(
    seed,
    density_name,
    obstacle_count,
    bounds,
    diameter_range,
    clear_radius,
    min_start_goal_distance,
    rectangle_length,
    rectangle_width,
    vehicle_width,
    safety_margin,
    overlap_margin,
    num_start_goal_pairs=10,
    min_pair_difference=5.0,
    max_attempts=300,
):
    """生成一张满足要求的随机环境，包含多对起点终点。"""
    last_error = None
    maximum_seed = np.iinfo(np.int64).max

    for attempt in range(max_attempts):
        actual_seed = int((seed + attempt * 1000003) % maximum_seed)
        rng = np.random.default_rng(actual_seed)

        try:
            # 首先生成障碍物（不依赖于起点终点）
            # 使用第一对起点终点来初始化障碍物生成
            temp_start, temp_goal = sample_start_goal(
                rng, bounds, clear_radius, min_start_goal_distance
            )

            obstacles = generate_random_obstacles(
                rng=rng,
                obstacle_count=obstacle_count,
                bounds=bounds,
                diameter_range=diameter_range,
                start=temp_start,
                goal=temp_goal,
                clear_radius=clear_radius,
                overlap_margin=overlap_margin,
            )

            vehicle_clearance = vehicle_width / 2.0 + safety_margin

            # 生成多对起点终点，确保彼此有足够差异
            start_goal_pairs = []

            for pair_idx in range(num_start_goal_pairs):
                # 每对使用不同的随机种子（使用质数间隔避免相似性）
                pair_seed = actual_seed + pair_idx * 7919  # 使用质数间隔
                pair_rng = np.random.default_rng(pair_seed)

                # 尝试生成有效且不重复的起点终点对
                pair_found = False
                for attempt in range(200):
                    start, goal = sample_start_goal(
                        pair_rng, bounds, clear_radius, min_start_goal_distance
                    )

                    # 检查起点终点是否与障碍物碰撞
                    start_valid = all(
                        math.hypot(start[0] - obs.x, start[1] - obs.y) >= obs.radius + clear_radius
                        for obs in obstacles
                    )
                    goal_valid = all(
                        math.hypot(goal[0] - obs.x, goal[1] - obs.y) >= obs.radius + clear_radius
                        for obs in obstacles
                    )

                    if not (start_valid and goal_valid):
                        continue

                    # 检查与已有的起点终点对是否有足够差异
                    too_similar = False
                    for existing_start, existing_goal in start_goal_pairs:
                        start_dist = math.hypot(start[0] - existing_start[0], start[1] - existing_start[1])
                        goal_dist = math.hypot(goal[0] - existing_goal[0], goal[1] - existing_goal[1])

                        # 如果起点和终点都太接近已有的某一对，则认为太相似
                        if start_dist < min_pair_difference and goal_dist < min_pair_difference:
                            too_similar = True
                            break

                    if too_similar:
                        continue

                    # 检查连通性
                    if grid_path_exists(
                        bounds=bounds,
                        obstacles=obstacles,
                        start=start,
                        goal=goal,
                        clearance=vehicle_clearance,
                        resolution=0.5,
                    ):
                        start_goal_pairs.append((start, goal))
                        pair_found = True
                        break

                if not pair_found:
                    raise RuntimeError(f"无法为第{pair_idx+1}对生成有效的起点终点")

            # 使用第一对起点终点作为默认值（向后兼容）
            default_start, default_goal = start_goal_pairs[0]

            goal_rectangle = make_goal_rectangle(
                default_start,
                default_goal,
                rectangle_length,
                rectangle_width,
                forward_offset=0.0,
            )

            description = (
                f"{density_name}随机果园环境；"
                f"seed={actual_seed}；"
                f"obstacles={obstacle_count}；"
                f"{num_start_goal_pairs}对起点终点"
            )

            environment = OrchardEnvironment(
                obstacles=obstacles,
                corridors=[],
                goal_rectangle=goal_rectangle,
                start_pos=default_start,
                goal_pos=default_goal,
                bounds=bounds,
                description=description,
            )

            return environment, actual_seed, start_goal_pairs

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"环境生成失败：density={density_name}，"
        f"obstacles={obstacle_count}，"
        f"最后错误={last_error}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="按照不同障碍物密度生成随机果园环境")

    parser.add_argument("--output-root", type=str, default="generated_density_maps")
    parser.add_argument("--seed", type=int, default=None, help="不设置时每次运行自动生成主种子")
    parser.add_argument("--maps-per-density", type=int, default=30, help="每种密度生成地图数量")
    parser.add_argument("--num-start-goal-pairs", type=int, default=10, help="每张地图生成的起点终点对数量")
    parser.add_argument("--min-pair-difference", type=float, default=5.0, help="不同起点终点对之间的最小距离（米）")

    parser.add_argument("--low-count", type=int, default=2, help="低密度障碍物数量")
    parser.add_argument("--medium-count", type=int, default=2, help="中密度障碍物数量")
    parser.add_argument("--high-count", type=int, default=2, help="高密度障碍物数量")

    parser.add_argument("--map-width", type=float, default=45.0)
    parser.add_argument("--map-height", type=float, default=30.0)
    parser.add_argument("--min-diameter", type=float, default=2.6)
    parser.add_argument("--max-diameter", type=float, default=4.0)
    parser.add_argument("--clear-radius", type=float, default=3.0)
    parser.add_argument("--min-start-goal-distance", type=float, default=25.0)

    parser.add_argument("--rectangle-length", type=float, default=14.0)
    parser.add_argument("--rectangle-width", type=float, default=9.0)
    parser.add_argument("--vehicle-width", type=float, default=1.96)
    parser.add_argument("--safety-margin", type=float, default=0.18)
    parser.add_argument("--overlap-margin", type=float, default=0.08)

    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--plot-first", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.maps_per_density <= 0:
        raise ValueError("--maps-per-density 必须大于0")
    if args.min_diameter <= 0 or args.max_diameter < args.min_diameter:
        raise ValueError("障碍物直径范围无效")

    bounds = (0.0, args.map_width, 0.0, args.map_height)
    diameter_range = (args.min_diameter, args.max_diameter)

    densities = {
        "density_low": args.low_count,
        "density_medium": args.medium_count,
        "density_high": args.high_count,
    }

    for density_name, obstacle_count in densities.items():
        if obstacle_count <= 0:
            raise ValueError(f"{density_name} 的障碍物数量必须大于0")

    output_root = Path(args.output_root)

    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif output_root.is_file():
            raise FileExistsError(f"输出路径是文件：{output_root}")
        elif any(output_root.iterdir()):
            raise FileExistsError(
                f"输出目录非空：{output_root}\n"
                f"重新生成时添加 --overwrite"
            )

    output_root.mkdir(parents=True, exist_ok=True)

    master_seed = args.seed if args.seed is not None else secrets.randbits(63)
    master_rng = np.random.default_rng(master_seed)

    map_area = args.map_width * args.map_height

    print(f"本次主随机种子：{master_seed}")
    print(f"地图尺寸：{args.map_width:.1f}m × {args.map_height:.1f}m")
    print(f"障碍物直径：{args.min_diameter:.1f}～{args.max_diameter:.1f}m")
    print(f"每种密度地图数量：{args.maps_per_density}")

    manifest_rows = []

    for density_name, obstacle_count in densities.items():
        density_dir = output_root / density_name
        density_dir.mkdir(parents=True, exist_ok=True)

        count_per_100m2 = obstacle_count / map_area * 100.0

        print("\n" + "=" * 70)
        print(f"生成密度：{density_name}")
        print(f"障碍物数量：{obstacle_count}")
        print(f"密度：{count_per_100m2:.2f} 个/100m²")
        print("=" * 70)

        for map_index in range(args.maps_per_density):
            initial_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))

            env, actual_seed, start_goal_pairs = generate_environment(
                seed=initial_seed,
                density_name=density_name,
                obstacle_count=obstacle_count,
                bounds=bounds,
                diameter_range=diameter_range,
                clear_radius=args.clear_radius,
                min_start_goal_distance=args.min_start_goal_distance,
                rectangle_length=args.rectangle_length,
                rectangle_width=args.rectangle_width,
                vehicle_width=args.vehicle_width,
                safety_margin=args.safety_margin,
                overlap_margin=args.overlap_margin,
                num_start_goal_pairs=args.num_start_goal_pairs,
                min_pair_difference=args.min_pair_difference,
            )

            file_stem = f"map_{map_index:04d}_seed_{actual_seed}"
            npz_path = density_dir / f"{file_stem}.npz"

            # 保存环境（需要修改save_environment以支持多对起点终点）
            save_environment_with_pairs(env, str(npz_path), start_goal_pairs)

            image_path = density_dir / f"{file_stem}.png" if args.save_images else None
            show_plot = args.plot_first and map_index == 0

            if args.save_images or show_plot:
                plot_environment(
                    env,
                    image_path=str(image_path) if image_path else None,
                    show=show_plot,
                )

            # 为每对起点终点添加一条记录到manifest
            for pair_idx, (start, goal) in enumerate(start_goal_pairs):
                manifest_rows.append({
                    "density": density_name,
                    "map_index": map_index,
                    "pair_index": pair_idx,
                    "seed": actual_seed,
                    "obstacle_count": obstacle_count,
                    "obstacles_per_100m2": count_per_100m2,
                    "start_x": start[0],
                    "start_y": start[1],
                    "goal_x": goal[0],
                    "goal_y": goal[1],
                    "npz_path": str(npz_path.resolve()),
                })

            print(
                f"[{map_index + 1:03d}/{args.maps_per_density:03d}] "
                f"{npz_path.name}，生成了{len(start_goal_pairs)}对起点终点"
            )

    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "density",
            "map_index",
            "pair_index",
            "seed",
            "obstacle_count",
            "obstacles_per_100m2",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "npz_path",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print("全部地图生成完成")
    print(f"输出目录：{output_root}")
    print(f"地图总数：{len(manifest_rows) // args.num_start_goal_pairs}")
    print(f"起点终点对总数：{len(manifest_rows)}")
    print(f"地图清单：{manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()