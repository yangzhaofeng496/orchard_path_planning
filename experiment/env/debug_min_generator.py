#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""调试 min() 生成器表达式问题"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from global_path_planning.innovation_sample.hybrid_sampler import HybridSampler
from vehicle.vehicle_collision import CircleObstacle


class SimplePose:
    def __init__(self, x, y, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw


bounds = (0.0, 100.0, 0.0, 100.0)
goal = SimplePose(90.0, 50.0, 0.0)
pose = SimplePose(10.0, 50.0, 0.0)
obstacles = [CircleObstacle(20.0, 50.0, 3.0)]

sampler = HybridSampler(
    bounds=bounds,
    goal=goal,
    obstacles=obstacles,
    use_tangent_guidance=True,
    goal_probability=0.20,
    tangent_probability=0.10,
)

# 获取直接阻挡索引
direct_blocking = sampler.direct_blocking_obstacles(pose)
print(f"直接阻挡索引: {direct_blocking}")

# 手动计算每个阻挡障碍物的边缘距离
distances = []
for index in direct_blocking:
    obstacle = sampler.obstacles[index]
    distance = sampler.obstacle_edge_distance(pose, obstacle)
    print(f"障碍物 {index}: 边缘距离 = {distance:.2f}m")
    distances.append(distance)

# 使用 min() 计算
if distances:
    min_distance = min(distances)
    print(f"\nmin(distances) = {min_distance:.2f}m")

# 使用生成器表达式
if direct_blocking:
    min_gen = min(
        (sampler.obstacle_edge_distance(pose, sampler.obstacles[index])
         for index in direct_blocking),
        default=math.inf,
    )
    print(f"min(生成器表达式) = {min_gen:.2f}m")

# 测试空列表
empty_min = min(
    (x for x in []),
    default=math.inf,
)
print(f"min(空生成器, default=inf) = {empty_min}")
