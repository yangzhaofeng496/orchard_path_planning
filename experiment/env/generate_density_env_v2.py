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
2. 每张地图按阻挡类型生成10对起点和终点。
3. 障碍物不会覆盖起点和终点安全区域。
4. 每张地图使用不同随机种子。
5. 栅格连通性检查禁止对角穿过相邻障碍物。
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
)


def save_environment_with_pairs(environment, npz_path, start_goal_pairs, pair_metadata):
    """
    保存果园环境，包含多对起点终点及其分类信息。

    Args:
        environment: OrchardEnvironment 对象
        npz_path: 输出文件路径
        start_goal_pairs: 起点终点对列表 [(start1, goal1), ...]
        pair_metadata: 每对的元数据字典列表
    """
    npz_path = os.path.abspath(npz_path)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    obstacles = np.asarray(
        [[obs.x, obs.y, obs.radius] for obs in environment.obstacles],
        dtype=float,
    ).reshape(-1, 3)

    start_goal_array = np.asarray(
        [[start[0], start[1], goal[0], goal[1]] for start, goal in start_goal_pairs],
        dtype=float,
    ).reshape(-1, 4)

    # 提取分类信息
    pair_categories = np.asarray([m['category'] for m in pair_metadata], dtype='U30')
    pair_category_codes = np.asarray([m['category_code'] for m in pair_metadata], dtype=np.int32)
    pair_blocking_obstacle_counts = np.asarray([m['blocking_obstacle_count'] for m in pair_metadata], dtype=np.int32)
    pair_blocking_cluster_counts = np.asarray([m['blocking_cluster_count'] for m in pair_metadata], dtype=np.int32)
    pair_direct_distances = np.asarray([m['direct_distance'] for m in pair_metadata], dtype=float)
    pair_seeds = np.asarray([m['pair_seed'] for m in pair_metadata], dtype=np.int64)

    rectangle = environment.goal_rectangle

    np.savez_compressed(
        npz_path,
        format_version=np.asarray([3], dtype=np.int32),  # 版本号改为3
        obstacles=obstacles,
        corridors=np.empty((0, 5), dtype=float),
        start_pos=np.asarray(environment.start_pos, dtype=float),
        goal_pos=np.asarray(environment.goal_pos, dtype=float),
        start_goal_pairs=start_goal_array,
        pair_categories=pair_categories,
        pair_category_codes=pair_category_codes,
        pair_blocking_obstacle_counts=pair_blocking_obstacle_counts,
        pair_blocking_cluster_counts=pair_blocking_cluster_counts,
        pair_direct_distances=pair_direct_distances,
        pair_seeds=pair_seeds,
        bounds=np.asarray(environment.bounds, dtype=float),
        rectangle=np.asarray([
            rectangle.length,
            rectangle.width,
            rectangle.forward_offset,
        ], dtype=float),
        description=np.asarray(environment.description),
    )
    return npz_path



def point_segment_distance(point, start, end):
    """计算点到线段的最短距离。"""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy

    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)

    ratio = ((px - ax) * dx + (py - ay) * dy) / length2
    ratio = max(0.0, min(1.0, ratio))
    closest_x = ax + ratio * dx
    closest_y = ay + ratio * dy

    return math.hypot(px - closest_x, py - closest_y)


def classify_obstacle_clusters(obstacles, classification_inflation=1.0):
    """
    使用并查集算法对障碍物进行聚类。

    Returns:
        obstacle_to_cluster: dict, 障碍物索引 -> 簇ID
    """
    n = len(obstacles)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 合并相邻障碍物
    for i in range(n):
        for j in range(i + 1, n):
            obs_i = obstacles[i]
            obs_j = obstacles[j]
            center_distance = math.hypot(obs_i.x - obs_j.x, obs_i.y - obs_j.y)
            threshold = obs_i.radius + obs_j.radius + 2 * classification_inflation

            if center_distance <= threshold:
                union(i, j)

    # 构建障碍物到簇ID的映射
    obstacle_to_cluster = {}
    for i in range(n):
        obstacle_to_cluster[i] = find(i)

    return obstacle_to_cluster



def classify_start_goal_pair(start, goal, obstacles, obstacle_to_cluster, classification_inflation=1.0):
    """
    分类起点终点对的阻挡类型。

    Returns:
        tuple: (category_name, category_code, blocking_obstacle_indexes, blocking_cluster_count)
            category_code: 0=unblocked, 1=single_cluster_blocked, 2=multi_cluster_blocked
    """
    # 检测直线阻挡障碍物
    blocking_indexes = []
    for i, obstacle in enumerate(obstacles):
        distance = point_segment_distance(
            (obstacle.x, obstacle.y),
            start,
            goal
        )
        threshold = obstacle.radius + classification_inflation

        if distance <= threshold:
            blocking_indexes.append(i)

    # 无阻挡
    if len(blocking_indexes) == 0:
        return "unblocked", 0, blocking_indexes, 0

    # 统计阻挡障碍物所属的簇
    cluster_ids = set()
    for idx in blocking_indexes:
        cluster_ids.add(obstacle_to_cluster[idx])

    cluster_count = len(cluster_ids)

    if cluster_count == 1:
        return "single_cluster_blocked", 1, blocking_indexes, 1
    else:
        return "multi_cluster_blocked", 2, blocking_indexes, cluster_count


def sample_start_goal(rng, bounds, clear_radius, min_distance):
    """随机生成起点和终点，保证两点距离足够远。"""
    x_min, x_max, y_min, y_max = bounds
    margin = clear_radius

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



def can_place_obstacle(obstacles, x, y, radius, bounds, overlap_margin):
    """检查障碍物是否可以放置（不考虑起终点）。"""
    x_min, x_max, y_min, y_max = bounds

    if x - radius < x_min + 0.3 or x + radius > x_max - 0.3:
        return False
    if y - radius < y_min + 0.3 or y + radius > y_max - 0.3:
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
            obstacles, x, y, radius, bounds, overlap_margin
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
    """
    使用二维栅格检查起点和终点是否连通。

    改进：
    1. 标记距离边界小于clearance的栅格为占用
    2. 对角移动时检查两个相邻正交栅格必须同时为空
    """
    x_min, x_max, y_min, y_max = bounds
    width = int(math.ceil((x_max - x_min) / resolution))
    height = int(math.ceil((y_max - y_min) / resolution))

    rows, cols = np.ogrid[:height, :width]
    grid_x = x_min + (cols + 0.5) * resolution
    grid_y = y_min + (rows + 0.5) * resolution
    occupied = np.zeros((height, width), dtype=bool)

    # 标记障碍物占用
    for obstacle in obstacles:
        radius = obstacle.radius + clearance
        occupied |= (grid_x - obstacle.x) ** 2 + (grid_y - obstacle.y) ** 2 <= radius ** 2

    # 标记边界区域为占用
    for row in range(height):
        for col in range(width):
            cell_x = x_min + (col + 0.5) * resolution
            cell_y = y_min + (row + 0.5) * resolution

            if (cell_x - x_min < clearance or x_max - cell_x < clearance or
                cell_y - y_min < clearance or y_max - cell_y < clearance):
                occupied[row, col] = True

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

    # 8邻域方向
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),  # 正交
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # 对角
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

            # 对角移动时，检查两个相邻正交栅格必须同时为空
            if dr != 0 and dc != 0:
                adj1_occupied = occupied[row + dr, col]
                adj2_occupied = occupied[row, col + dc]
                if adj1_occupied or adj2_occupied:
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
    category_requirements,
    classification_inflation=1.0,
    num_start_goal_pairs=10,
    min_pair_difference=5.0,
    max_attempts=50,
):
    """
    生成一张满足要求的随机环境，包含按类别分层的起点终点对。

    category_requirements: dict, {"unblocked": 5, "single_cluster_blocked": 4, "multi_cluster_blocked": 1}
    """
    last_error = None
    maximum_seed = np.iinfo(np.int64).max

    for attempt in range(max_attempts):
        actual_seed = int((seed + attempt * 1000003) % maximum_seed)
        rng = np.random.default_rng(actual_seed)

        try:
            # 1. 先生成障碍物（不依赖起终点）
            obstacles = generate_random_obstacles(
                rng=rng,
                obstacle_count=obstacle_count,
                bounds=bounds,
                diameter_range=diameter_range,
                overlap_margin=overlap_margin,
            )

            # 2. 对障碍物进行聚类
            obstacle_to_cluster = classify_obstacle_clusters(obstacles, classification_inflation)

            vehicle_clearance = vehicle_width / 2.0 + safety_margin

            # 3. 按类别生成起点终点对
            start_goal_pairs = []
            pair_metadata = []
            category_counts = {cat: 0 for cat in category_requirements}

            # 为每个类别生成所需数量的起点终点对
            for target_category, target_count in category_requirements.items():
                for _ in range(target_count):
                    pair_seed = actual_seed + len(start_goal_pairs) * 7919
                    pair_rng = np.random.default_rng(pair_seed)

                    pair_found = False
                    for pair_attempt in range(500):
                        start, goal = sample_start_goal(
                            pair_rng, bounds, clear_radius, min_start_goal_distance
                        )

                        # 检查起点终点距离边界
                        x_min, x_max, y_min, y_max = bounds
                        if (start[0] - x_min < clear_radius or x_max - start[0] < clear_radius or
                            start[1] - y_min < clear_radius or y_max - start[1] < clear_radius):
                            continue
                        if (goal[0] - x_min < clear_radius or x_max - goal[0] < clear_radius or
                            goal[1] - y_min < clear_radius or y_max - goal[1] < clear_radius):
                            continue

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

                        # 分类起点终点对
                        category, code, blocking_indexes, cluster_count = classify_start_goal_pair(
                            start, goal, obstacles, obstacle_to_cluster, classification_inflation
                        )

                        # 检查是否匹配目标类别
                        if category != target_category:
                            continue

                        # 检查与已有的起点终点对是否有足够差异
                        too_similar = False
                        for existing_start, existing_goal in start_goal_pairs:
                            start_dist = math.hypot(start[0] - existing_start[0], start[1] - existing_start[1])
                            goal_dist = math.hypot(goal[0] - existing_goal[0], goal[1] - existing_goal[1])

                            if start_dist < min_pair_difference and goal_dist < min_pair_difference:
                                too_similar = True
                                break

                        if too_similar:
                            continue

                        # 检查连通性
                        if not grid_path_exists(
                            bounds=bounds,
                            obstacles=obstacles,
                            start=start,
                            goal=goal,
                            clearance=vehicle_clearance,
                            resolution=0.5,
                        ):
                            continue

                        # 成功生成一对
                        direct_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
                        start_goal_pairs.append((start, goal))
                        pair_metadata.append({
                            'category': category,
                            'category_code': code,
                            'blocking_obstacle_count': len(blocking_indexes),
                            'blocking_cluster_count': cluster_count,
                            'direct_distance': direct_distance,
                            'pair_seed': pair_seed,
                        })
                        category_counts[category] += 1
                        pair_found = True
                        break

                    if not pair_found:
                        raise RuntimeError(
                            f"无法为类别 {target_category} 生成足够的起点终点对 "
                            f"(已生成{category_counts[target_category]}/{target_count}，"
                            f"尝试次数={pair_attempt+1})"
                        )

            # 验证类别数量
            for category, required_count in category_requirements.items():
                actual_count = category_counts[category]
                if actual_count != required_count:
                    raise RuntimeError(
                        f"类别 {category} 数量不匹配: {actual_count} != {required_count}"
                    )

            # 使用第一对起点终点作为默认值
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

            return environment, actual_seed, start_goal_pairs, pair_metadata, category_counts

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

    parser.add_argument("--low-count", type=int, default=10, help="低密度障碍物数量")
    parser.add_argument("--medium-count", type=int, default=18, help="中密度障碍物数量")
    parser.add_argument("--high-count", type=int, default=26, help="高密度障碍物数量")

    parser.add_argument("--map-width", type=float, default=45.0)
    parser.add_argument("--map-height", type=float, default=30.0)
    parser.add_argument("--min-diameter", type=float, default=2.6)
    parser.add_argument("--max-diameter", type=float, default=4.0)
    parser.add_argument("--clear-radius", type=float, default=3.0)
    parser.add_argument("--min-start-goal-distance", type=float, default=25.0)
    parser.add_argument("--classification-inflation", type=float, default=1.0)

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

    # 定义每种密度的类别要求
    category_requirements_map = {
        "density_low": {
            "unblocked": 5,
            "single_cluster_blocked": 4,
            "multi_cluster_blocked": 1,
        },
        "density_medium": {
            "unblocked": 3,
            "single_cluster_blocked": 4,
            "multi_cluster_blocked": 3,
        },
        "density_high": {
            "unblocked": 0,  # 高密度不要求无遮挡场景
            "single_cluster_blocked": 5,
            "multi_cluster_blocked": 5,
        },
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
    print(f"分类inflation：{args.classification_inflation:.2f}m")

    manifest_rows = []


    for density_name, obstacle_count in densities.items():
        density_dir = output_root / density_name
        density_dir.mkdir(parents=True, exist_ok=True)

        count_per_100m2 = obstacle_count / map_area * 100.0
        category_requirements = category_requirements_map[density_name]

        # 计算障碍物总面积和占比
        avg_diameter = (args.min_diameter + args.max_diameter) / 2.0
        avg_radius = avg_diameter / 2.0
        avg_obstacle_area = math.pi * avg_radius ** 2
        total_obstacle_area = obstacle_count * avg_obstacle_area
        obstacle_area_ratio = total_obstacle_area / map_area

        # 膨胀后的面积占比
        inflated_radius = avg_radius + args.classification_inflation
        inflated_obstacle_area = obstacle_count * math.pi * inflated_radius ** 2
        inflated_obstacle_area_ratio = inflated_obstacle_area / map_area

        print("\n" + "=" * 70)
        print(f"生成密度：{density_name}")
        print(f"障碍物数量：{obstacle_count}")
        print(f"密度：{count_per_100m2:.2f} 个/100m²")
        print(f"障碍物面积占比：{obstacle_area_ratio:.2%}")
        print(f"膨胀后面积占比：{inflated_obstacle_area_ratio:.2%}")
        print(f"类别要求：无遮挡={category_requirements['unblocked']}，"
              f"单簇阻挡={category_requirements['single_cluster_blocked']}，"
              f"多簇阻挡={category_requirements['multi_cluster_blocked']}")
        print("=" * 70)

        for map_index in range(args.maps_per_density):
            initial_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))

            env, actual_seed, start_goal_pairs, pair_metadata, category_counts = generate_environment(
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
                category_requirements=category_requirements,
                classification_inflation=args.classification_inflation,
                num_start_goal_pairs=args.num_start_goal_pairs,
                min_pair_difference=args.min_pair_difference,
            )

            file_stem = f"map_{map_index:04d}_seed_{actual_seed}"
            npz_path = density_dir / f"{file_stem}.npz"

            save_environment_with_pairs(env, str(npz_path), start_goal_pairs, pair_metadata)

            image_path = density_dir / f"{file_stem}.png" if args.save_images else None
            show_plot = args.plot_first and map_index == 0

            if args.save_images or show_plot:
                plot_environment(
                    env,
                    image_path=str(image_path) if image_path else None,
                    show=show_plot,
                )


            # 为每对起点终点添加一条记录到manifest
            for pair_idx, ((start, goal), metadata) in enumerate(zip(start_goal_pairs, pair_metadata)):
                manifest_rows.append({
                    "density": density_name,
                    "map_index": map_index,
                    "pair_index": pair_idx,
                    "seed": actual_seed,
                    "obstacle_count": obstacle_count,
                    "obstacles_per_100m2": count_per_100m2,
                    "obstacle_area_ratio": obstacle_area_ratio,
                    "inflated_obstacle_area_ratio": inflated_obstacle_area_ratio,
                    "start_x": start[0],
                    "start_y": start[1],
                    "goal_x": goal[0],
                    "goal_y": goal[1],
                    "pair_category": metadata['category'],
                    "blocking_obstacle_count": metadata['blocking_obstacle_count'],
                    "blocking_cluster_count": metadata['blocking_cluster_count'],
                    "direct_distance": metadata['direct_distance'],
                    "pair_seed": metadata['pair_seed'],
                    "npz_path": str(npz_path.resolve()),
                })

            print(
                f"[{map_index + 1:03d}/{args.maps_per_density:03d}] "
                f"{npz_path.name} | "
                f"[类别分布] 无遮挡={category_counts['unblocked']}，"
                f"单簇阻挡={category_counts['single_cluster_blocked']}，"
                f"多簇阻挡={category_counts['multi_cluster_blocked']}"
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
            "obstacle_area_ratio",
            "inflated_obstacle_area_ratio",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "pair_category",
            "blocking_obstacle_count",
            "blocking_cluster_count",
            "direct_distance",
            "pair_seed",
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

