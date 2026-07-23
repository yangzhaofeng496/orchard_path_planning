#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""综合测试切向采样的所有关键场景"""

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


def test_scenario(name, pose, goal, obstacles, expected_tangent_prob_range):
    """测试一个场景"""
    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"{'='*60}")

    bounds = (0.0, 100.0, 0.0, 100.0)
    sampler = HybridSampler(
        bounds=bounds,
        goal=goal,
        obstacles=obstacles,
        use_tangent_guidance=True,
        goal_probability=0.20,
        tangent_probability=0.10,
    )

    sampler.update_rectangle_anchor(pose)

    probs = sampler.effective_sampling_probabilities()
    tangent_prob = probs['tangent']

    print(f"起点: ({pose.x}, {pose.y})")
    print(f"终点: ({goal.x}, {goal.y})")
    print(f"障碍物数量: {len(obstacles)}")
    print(f"直接阻挡数量: {len(sampler.direct_blocking_indexes)}")
    print(f"阻挡簇数量: {sampler.direct_blocking_cluster_count}")
    print(f"最近阻挡距离: {sampler.nearest_blocking_distance:.2f}m")
    print(f"切向引导激活: {sampler.tangent_guidance.active}")
    print(f"切向采样概率: {tangent_prob:.4f}")

    # 验证概率范围
    min_prob, max_prob = expected_tangent_prob_range
    if min_prob <= tangent_prob <= max_prob:
        print(f"✅ 通过：切向概率在预期范围 [{min_prob}, {max_prob}]")
        return True
    else:
        print(f"❌ 失败：切向概率 {tangent_prob:.4f} 不在预期范围 [{min_prob}, {max_prob}]")
        return False


# 测试场景1：无遮挡
print("\n" + "="*60)
print("测试场景1：无遮挡 -> 切向概率应为 0.0")
print("="*60)
result1 = test_scenario(
    "无遮挡",
    pose=SimplePose(10.0, 50.0),
    goal=SimplePose(90.0, 50.0),
    obstacles=[],
    expected_tangent_prob_range=(0.0, 0.0)
)

# 测试场景2：单障碍物阻挡，距离近
result2 = test_scenario(
    "单障碍物阻挡（近）",
    pose=SimplePose(10.0, 50.0),
    goal=SimplePose(90.0, 50.0),
    obstacles=[CircleObstacle(20.0, 50.0, 3.0)],
    expected_tangent_prob_range=(0.02, 0.02)
)

# 测试场景3：单障碍物阻挡，距离远（>12m）
result3 = test_scenario(
    "单障碍物阻挡（远）",
    pose=SimplePose(10.0, 50.0),
    goal=SimplePose(90.0, 50.0),
    obstacles=[CircleObstacle(30.0, 50.0, 3.0)],  # 距离约16m
    expected_tangent_prob_range=(0.0, 0.0)
)

# 测试场景4：多障碍物阻挡
result4 = test_scenario(
    "多障碍物阻挡",
    pose=SimplePose(10.0, 50.0),
    goal=SimplePose(90.0, 50.0),
    obstacles=[
        CircleObstacle(20.0, 48.0, 2.0),
        CircleObstacle(25.0, 52.0, 2.0),
    ],
    expected_tangent_prob_range=(0.05, 0.05)
)

# 测试场景5：已找到可行路径
print("\n" + "="*60)
print("测试场景5：已找到可行路径 -> 切向概率应为 0.0")
print("="*60)
bounds = (0.0, 100.0, 0.0, 100.0)
goal = SimplePose(90.0, 50.0)
pose = SimplePose(10.0, 50.0)
obstacles = [CircleObstacle(20.0, 50.0, 3.0)]

sampler = HybridSampler(
    bounds=bounds,
    goal=goal,
    obstacles=obstacles,
    use_tangent_guidance=True,
    goal_probability=0.20,
    tangent_probability=0.10,
)

sampler.set_feasible_path_found(True)  # 设置已找到路径
sampler.update_rectangle_anchor(pose)

probs = sampler.effective_sampling_probabilities()
result5 = probs['tangent'] == 0.0

print(f"已找到可行路径: {sampler.has_feasible_path}")
print(f"切向采样概率: {probs['tangent']:.4f}")
if result5:
    print("✅ 通过：已找到路径后切向概率为 0.0")
else:
    print("❌ 失败：已找到路径后切向概率应为 0.0")

# 汇总结果
print("\n" + "="*60)
print("测试结果汇总")
print("="*60)
all_results = [result1, result2, result3, result4, result5]
passed = sum(all_results)
total = len(all_results)
print(f"通过: {passed}/{total}")
if passed == total:
    print("✅ 所有测试通过！")
else:
    print(f"❌ {total - passed} 个测试失败")
