"""
测试 TEB 算法，调试规划结果
"""
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), 'configs', 'teb_config.yaml')
config = load_teb_config(config_path)
print('=' * 70)
print('TEB 配置')
print('=' * 70)
print(f'num_samples: {config.num_samples}')
print(f'max_iterations: {config.max_iterations}')
print(f'w_time: {config.w_time}')
print(f'w_obstacle: {config.w_obstacle}')
print(f'w_kinematics: {config.w_kinematics}')
print(f'w_path: {config.w_path}')
print()

# 创建规划器
bounds = (0.0, 20.0, 0.0, 12.0)
planner = TEBPlanner(config, bounds)

# 创建简单的全局路径（从起点到终点的直线）
print('=' * 70)
print('创建全局路径')
print('=' * 70)
global_path = []
start_x, start_y = 2.0, 6.0
goal_x, goal_y = 18.0, 6.0

for i in range(20):
    ratio = i / 19.0
    x = start_x + ratio * (goal_x - start_x)
    y = start_y + ratio * (goal_y - start_y)
    yaw = math.atan2(goal_y - start_y, goal_x - start_x)
    global_path.append(Pose(x, y, yaw))

print(f'全局路径: {len(global_path)} 个点')
print(f'  起点: ({global_path[0].x:.2f}, {global_path[0].y:.2f})')
print(f'  终点: ({global_path[-1].x:.2f}, {global_path[-1].y:.2f})')

planner.set_global_path(global_path)
print()

# 创建初始状态
print('=' * 70)
print('车辆初始状态')
print('=' * 70)
state = VehicleState(x=2.0, y=6.0, yaw=0.0, speed=0.5, steering=0.0)
print(f'位置: ({state.x:.2f}, {state.y:.2f})')
print(f'航向: {math.degrees(state.yaw):.1f}°')
print(f'速度: {state.speed:.2f} m/s')
print()

# 创建障碍物
print('=' * 70)
print('障碍物')
print('=' * 70)
obstacles = [CircleObstacle(10.0, 6.0, 1.5)]
for i, obs in enumerate(obstacles):
    print(f'障碍物 {i}: 中心=({obs.x:.2f}, {obs.y:.2f}), 半径={obs.radius:.2f}')
print()

# 执行规划
print('=' * 70)
print('执行 TEB 规划')
print('=' * 70)
result = planner.plan(state, obstacles)
print()

if result and result.success:
    print('✓ 规划成功！')
    print()
    print(f'TEB 轨迹节点数: {len(result.trajectory)}')
    print()
    print('节点详细信息:')
    print(f'{"索引":<6} {"X (m)":<10} {"Y (m)":<10} {"航向(°)":<12} {"距离起点":<12}')
    print('-' * 60)

    for i, pose in enumerate(result.trajectory):
        dist_from_start = math.hypot(pose.x - start_x, pose.y - start_y)
        print(f'{i:<6} {pose.x:<10.2f} {pose.y:<10.2f} {math.degrees(pose.yaw):<12.1f} {dist_from_start:<12.2f}')

    print()

    # 检查 TEB 节点
    if hasattr(planner, 'teb_nodes') and planner.teb_nodes:
        print('TEB 节点速度信息:')
        print(f'{"索引":<6} {"速度 (m/s)":<12} {"时间间隔 (s)":<15}')
        print('-' * 40)

        for i, node in enumerate(planner.teb_nodes[:-1]):
            next_node = planner.teb_nodes[i + 1]
            dist = math.hypot(next_node.x - node.x, next_node.y - node.y)
            speed = dist / node.dt if node.dt > 0.001 else 0.0
            print(f'{i:<6} {speed:<12.2f} {node.dt:<15.3f}')
        print()

    # 检查是否有不合理的轨迹点
    print('轨迹检查:')
    for i in range(len(result.trajectory) - 1):
        p1 = result.trajectory[i]
        p2 = result.trajectory[i + 1]
        dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
        if dist > 5.0:
            print(f'  ⚠ 节点 {i} -> {i+1} 距离过大: {dist:.2f} m')

        # 检查是否穿过障碍物
        for obs in obstacles:
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            dist_to_obs = math.hypot(mid_x - obs.x, mid_y - obs.y)
            if dist_to_obs < obs.radius:
                print(f'  ⚠ 节点 {i} -> {i+1} 可能穿过障碍物！距离: {dist_to_obs:.2f} m')

    print('轨迹检查完成')

else:
    print('✗ 规划失败')
    print(f'失败原因: {planner.last_failure_reason}')

print()
print('=' * 70)
