"""
调试 TEB 失败恢复 - 详细日志
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

def test_single_recovery():
    """测试单次失败恢复的详细过程"""
    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = True
    config.debug_log = True
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    print("=" * 70)
    print("详细调试：单次失败恢复")
    print("=" * 70)

    # 第 1 次：正常规划
    print("\n[周期 1] 正常规划")
    print("-" * 70)
    state1 = VehicleState(3.0, 6.0, 0.0, 0.5, 0.0)
    result1 = planner.plan(state1, [])
    print(f"结果: {'成功' if result1 else '失败'}")
    print(f"consecutive_failures = {planner.consecutive_failures}")
    print(f"teb_nodes = {len(planner.teb_nodes)}")

    # 第 2 次：添加障碍物导致失败
    print("\n[周期 2] 添加大障碍物，预期失败")
    print("-" * 70)
    obstacles = [CircleObstacle(10.0, 6.0, 5.0)]
    state2 = VehicleState(4.0, 6.0, 0.0, 0.5, 0.0)
    result2 = planner.plan(state2, obstacles)
    print(f"结果: {'成功' if result2 else '失败'}")
    print(f"失败原因: {planner.last_failure_reason}")
    print(f"consecutive_failures = {planner.consecutive_failures}")
    print(f"teb_nodes = {len(planner.teb_nodes)}")

    # 第 3 次：移除障碍物，验证恢复
    print("\n[周期 3] 移除障碍物，预期恢复成功")
    print("-" * 70)
    state3 = VehicleState(5.0, 6.0, 0.0, 0.5, 0.0)
    result3 = planner.plan(state3, [])
    print(f"结果: {'成功' if result3 else '失败'}")
    if result3 is None:
        print(f"失败原因: {planner.last_failure_reason}")
    print(f"consecutive_failures = {planner.consecutive_failures}")
    print(f"teb_nodes = {len(planner.teb_nodes)}")

    print("\n" + "=" * 70)
    if result3 is not None:
        print("✅ 恢复成功")
    else:
        print("❌ 恢复失败")
    print("=" * 70)
    
    return result3 is not None

if __name__ == '__main__':
    success = test_single_recovery()
    sys.exit(0 if success else 1)
