#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def convert_scene_to_map(
    npz_path,
    output_prefix,
    resolution=0.05,
    inflation=0.0,
):
    npz_path = Path(npz_path)
    output_prefix = Path(output_prefix)

    with np.load(npz_path, allow_pickle=False) as data:
        obstacles = np.asarray(data["obstacles"], dtype=float)
        bounds = np.asarray(data["bounds"], dtype=float)

    if obstacles.ndim != 2 or obstacles.shape[1] != 3:
        raise ValueError(
            "obstacles必须为(N,3)，每行表示(x, y, radius)"
        )

    if bounds.shape != (4,):
        raise ValueError(
            "bounds必须包含4个数：[xmin, xmax, ymin, ymax]"
        )

    # 假设格式为[x_min, x_max, y_min, y_max]
    x_min, x_max, y_min, y_max = bounds

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "bounds无效。请确认它是否采用"
            "[xmin, xmax, ymin, ymax]格式。"
        )

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    if width <= 0 or height <= 0:
        raise ValueError("计算得到的地图尺寸无效")

    # ROS地图常用灰度：
    # 254=空闲，0=占据，205=未知
    grid = np.full((height, width), 254, dtype=np.uint8)

    # 计算每个像素中心在世界坐标系中的位置
    x_coordinates = (
        x_min + (np.arange(width) + 0.5) * resolution
    )
    y_coordinates = (
        y_min + (np.arange(height) + 0.5) * resolution
    )

    xx, yy = np.meshgrid(x_coordinates, y_coordinates)

    # 将圆形障碍物栅格化
    for obstacle_index, (x, y, radius) in enumerate(obstacles):
        effective_radius = radius + inflation

        if effective_radius < 0:
            raise ValueError(
                f"障碍物{obstacle_index}半径无效：{radius}"
            )

        occupied = (
            (xx - x) ** 2 + (yy - y) ** 2
            <= effective_radius ** 2
        )

        grid[occupied] = 0

    # NumPy第0行对应y_min，图像第0行显示在顶部，
    # 因此写PGM前上下翻转。
    pgm_grid = np.flipud(grid)

    pgm_path = output_prefix.with_suffix(".pgm")
    yaml_path = output_prefix.with_suffix(".yaml")

    pgm_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(pgm_grid, mode="L").save(pgm_path)

    map_metadata = {
        "image": pgm_path.name,
        "mode": "trinary",
        "resolution": float(resolution),
        "origin": [
            float(x_min),
            float(y_min),
            0.0,
        ],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            map_metadata,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"Bounds: {bounds}")
    print(f"Resolution: {resolution} m/pixel")
    print(f"Map size: {width} x {height}")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Inflation: {inflation} m")
    print(f"PGM saved to: {pgm_path}")
    print(f"YAML saved to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将NPZ圆形障碍物场景转换为ROS地图"
    )
    parser.add_argument("input", help="输入NPZ文件")
    parser.add_argument(
        "output_prefix",
        help="输出文件前缀",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="地图分辨率，单位m/pixel",
    )
    parser.add_argument(
        "--inflation",
        type=float,
        default=0.0,
        help="障碍物额外膨胀半径，单位m",
    )
    args = parser.parse_args()

    convert_scene_to_map(
        npz_path=args.input,
        output_prefix=args.output_prefix,
        resolution=args.resolution,
        inflation=args.inflation,
    )


if __name__ == "__main__":
    main()