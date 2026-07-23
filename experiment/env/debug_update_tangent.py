#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""直接测试 update_tangent_guidance 方法"""

import sys
import os

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

print(f"初始状态:")
print(f"  use_tangent_guidance: {sampler.use_tangent_guidance}")
print(f"  has_feasible_path: {sampler.has_feasible_path}")
print(f"  obstacles: {len(sampler.obstacles)}")

# 直接调用 update_tangent_guidance
print(f"\n调用 update_tangent_guidance(pose)...")
sampler.update_tangent_guidance(pose)

print(f"\n调用后状态:")
print(f"  direct_blocking_indexes: {sampler.direct_blocking_indexes}")
print(f"  direct_blocking_cluster_count: {sampler.direct_blocking_cluster_count}")
print(f"  nearest_blocking_distance: {sampler.nearest_blocking_distance}")
print(f"  tangent_guidance.active: {sampler.tangent_guidance.active}")

# 手动计算验证
if sampler.direct_blocking_indexes:
    distances = [
        sampler.obstacle_edge_distance(pose, sampler.obstacles[idx])
        for idx in sampler.direct_blocking_indexes
    ]
    print(f"\n手动验证:")
    print(f"  边缘距离列表: {distances}")
    print(f"  最小值: {min(distances)}")
