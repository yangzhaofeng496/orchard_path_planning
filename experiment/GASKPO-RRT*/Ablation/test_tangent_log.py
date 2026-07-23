import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry
from global_path_planning.innovation_sample.ackermann_rrt_star import AckermannRRTStar
from global_path_planning.innovation_sample.orchard_environment import load_environment

# 加载环境
env = load_environment('comparison_map_density40.npz')
start_pose = Pose(x=env.start_pos[0], y=env.start_pos[1], yaw=0.0)
goal_pose = Pose(x=env.goal_pos[0], y=env.goal_pos[1], yaw=0.0)

vehicle = VehicleGeometry(
    front_length=3.0,
    rear_length=1.0,
    width=1.6,
    safety_margin=0.15
)

print('测试 GASKPO-RRT* 切向引导...')
print(f'起点: ({start_pose.x:.2f}, {start_pose.y:.2f})')
print(f'终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f})')
print(f'障碍物数量: {len(env.obstacles)}')
print()

planner = AckermannRRTStar(
    start=start_pose,
    goal=goal_pose,
    bounds=env.bounds,
    vehicle=vehicle,
    obstacles=env.obstacles,
    curvature=1.0 / 3.0,
    use_ackermann_constraints=False,
    expand_length=3.0,
    step_size=0.08,
    max_iterations=1500,
    near_radius=5.0,
    use_hybrid_sampling=True,
    goal_probability=0.2,
    tangent_probability=0.20,
    adaptive_sampling_probabilities=False,
    corridor_probability=0.0,
    rectangle_probability=0.45,
    allow_reverse=True,
    use_tangent_guidance=True,
    random_seed=0,
)

result = planner.planning()

if result:
    print(f'\n成功！节点数: {len(planner.nodes)}')
else:
    print('\n失败！')
