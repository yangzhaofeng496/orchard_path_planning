#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""调试切向距离计算"""

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


# 创建一个简单的测试场景
bounds = (0.0, 100.0, 0.0, 100.0)
goal = SimplePose(90.0, 50.0, 0.0)
pose = SimplePose(10.0, 50.0, 0.0)

# 在直线路径上放置一个障碍物
obstacle = CircleObstacle(20.0, 50.0, 3.0)
obstacles = [obstacle]

sampler = HybridSampler(
    bounds=bounds,
    goal=goal,
    obstacles=obstacles,
    use_tangent_guidance=True,
    goal_probability=0.20,
    tangent_probability=0.10,
    rectangle_probability=0.0,
    corridor_probability=0.0,
)

print(f"起点: ({pose.x}, {pose.y})")
print(f"终点: ({goal.x}, {goal.y})")
print(f"障碍物: ({obstacle.x}, {obstacle.y}), 半径={obstacle.radius}")
print(f"膨胀半径: {sampler.inflated_radius(obstacle)}")

# 手动计算距离
center_distance = math.hypot(obstacle.x - pose.x, obstacle.y - pose.y)
edge_distance = center_distance - sampler.inflated_radius(obstacle)
print(f"\n手动计算:")
print(f"  中心距离: {center_distance:.2f}m")
print(f"  边缘距离: {edge_distance:.2f}m")

# 使用方法计算
method_distance = sampler.obstacle_edge_distance(pose, obstacle)
print(f"  方法计算边缘距离: {method_distance:.2f}m")

# 检查直线是否被阻挡
print(f"\n检查直线阻挡:")
direct_blocking = sampler.direct_blocking_obstacles(pose)
print(f"  直接阻挡障碍物索引: {direct_blocking}")
print(f"  直接阻挡障碍物数量: {len(direct_blocking)}")

# 更新切向引导
print(f"\n更新切向引导:")
sampler.update_rectangle_anchor(pose)
print(f"  直接阻挡索引: {sampler.direct_blocking_indexes}")
print(f"  直接阻挡簇数量: {sampler.direct_blocking_cluster_count}")
print(f"  最近阻挡距离: {sampler.nearest_blocking_distance:.2f}m")
print(f"  切向引导激活: {sampler.tangent_guidance.active}")

# 检查切向采样条件
print(f"\n切向采样条件检查:")
print(f"  use_tangent_guidance: {sampler.use_tangent_guidance}")
print(f"  tangent_probability > 0: {sampler.tangent_probability > 0}")
print(f"  has_feasible_path: {sampler.has_feasible_path}")
print(f"  tangent_guidance.active: {sampler.tangent_guidance.active}")
print(f"  direct_blocking_indexes非空: {bool(sampler.direct_blocking_indexes)}")
print(f"  nearest_blocking_distance <= 12.0: {sampler.nearest_blocking_distance <= 12.0}")
print(f"  should_use_tangent_sampling(): {sampler.should_use_tangent_sampling()}")

# 检查有效概率
probs = sampler.effective_sampling_probabilities()
print(f"\n有效采样概率:")
print(f"  goal: {probs['goal']:.4f}")
print(f"  tangent: {probs['tangent']:.4f}")
print(f"  uniform: {probs['uniform']:.4f}")
print(f"  总和: {sum(probs.values()):.6f}")
