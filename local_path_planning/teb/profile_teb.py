"""
TEB 性能分析工具
帮助定位性能瓶颈
"""
import sys
import os
import time
import cProfile
import pstats
from io import StringIO

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle


def profile_teb_planning():
    """性能分析 TEB 规划"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '../configs/teb_config.yaml')
    config = load_teb_config(config_path)

    print("=" * 70)
    print("TEB 性能分析")
    print("=" * 70)
    print(f"配置:")
    print(f"  节点数: {config.num_samples}")
    print(f"  最大迭代: {config.max_iterations}")
    print(f"  碰撞检查分辨率: {config.collision_check_resolution}")
    print(f"  时间步长: {config.dt}")
    print("=" * 70)

    # 创建规划器
    bounds = (0.0, 20.0, 0.0, 12.0)
    planner = TEBPlanner(config, bounds)

    # 设置全局路径
    global_path = [
        Pose(2.0, 6.0, 0.0),
        Pose(6.0, 6.0, 0.0),
        Pose(10.0, 6.0, 0.0),
        Pose(14.0, 6.0, 0.0),
        Pose(18.0, 6.0, 0.0),
    ]
    planner.set_global_path(global_path)

    # 规划参数
    start_state = VehicleState(2.0, 6.0, 0.0, 0.5, 0.0)
    obstacles = [CircleObstacle(10.0, 6.0, 1.5)]

    print("\n开始性能分析...")

    # 使用 cProfile 分析
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    result = planner.plan(start_state, obstacles)
    elapsed = time.time() - start_time

    profiler.disable()

    print(f"\n规划结果:")
    if result:
        print(f"  成功: {result.success}")
        print(f"  轨迹点数: {len(result.trajectory)}")
    else:
        print(f"  失败: 返回 None")
    print(f"  总耗时: {elapsed*1000:.1f} ms")

    # 输出性能统计
    print("\n" + "=" * 70)
    print("性能热点 (按累计时间排序):")
    print("=" * 70)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # 显示前30个最耗时的函数

    profile_output = s.getvalue()

    # 只显示相关的行
    for line in profile_output.split('\n'):
        if any(keyword in line for keyword in [
            '_objective_function',
            '_obstacle_cost',
            '_vehicle_obstacle_clearance',
            '_kinematics_cost',
            '_acceleration_cost',
            'minimize',
            'ncalls',
            'tottime',
            'cumtime',
        ]):
            print(line)

    print("\n" + "=" * 70)
    print("性能建议:")
    print("=" * 70)

    if elapsed > 1.0:
        print("⚠️  规划时间过长 (>1秒)")
        print("   建议:")
        print("   1. 减少 num_samples (当前: {})".format(config.num_samples))
        print("   2. 减少 max_iterations (当前: {})".format(config.max_iterations))
        print("   3. 增大 collision_check_resolution (当前: {})".format(config.collision_check_resolution))
        print("   4. 增大 dt 时间步长 (当前: {})".format(config.dt))
    elif elapsed > 0.5:
        print("⚠️  规划时间较长 (0.5-1秒)")
        print("   建议适当调整参数以提升实时性")
    else:
        print("✅ 规划速度良好")


if __name__ == "__main__":
    profile_teb_planning()
