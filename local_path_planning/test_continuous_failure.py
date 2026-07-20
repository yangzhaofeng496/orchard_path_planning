"""
测试 TEB 连续失败场景
验证失败后是否能重新进入优化
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

def test_continuous_failure():
    """测试连续失败场景"""
    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = True  # 打开详细日志
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    # 设置一个正常的路径
    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    print("=" * 70)
    print("测试场景：连续失败后是否能重新进入优化")
    print("=" * 70)

    # 场景 1: 让规划器连续失败 10 次（车辆超出边界）
    print("\n📊 阶段 1: 连续失败 10 次（车辆位置异常）")
    print("-" * 70)
    
    for i in range(1, 11):
        state = VehicleState(100.0, 100.0, 0.0, 0.0, 0.0)  # 远离边界
        result = planner.plan(state, [])
        
        print(f"周期 {i}: 结果={'成功' if result else '失败'}, "
              f"连续失败={planner.consecutive_failures}, "
              f"TEB节点={len(planner.teb_nodes)}")
        
        if result is not None:
            print(f"  ❌ 预期失败但实际成功！")
            return False

    # 场景 2: 车辆回到正常位置，验证是否能恢复
    print("\n📊 阶段 2: 车辆回到正常位置，验证恢复")
    print("-" * 70)
    
    for i in range(11, 14):
        state = VehicleState(3.0, 6.0, 0.0, 0.5, 0.0)  # 正常位置
        result = planner.plan(state, [])
        
        print(f"周期 {i}: 结果={'成功' if result else '失败'}, "
              f"连续失败={planner.consecutive_failures}, "
              f"TEB节点={len(planner.teb_nodes)}")
        
        if result is None:
            print(f"  ❌ 预期成功但实际失败！失败原因: {planner.last_failure_reason}")
            return False
        else:
            print(f"  ✅ 成功恢复！")
            break

    # 场景 3: 添加障碍物导致碰撞失败
    print("\n📊 阶段 3: 添加障碍物导致碰撞失败")
    print("-" * 70)
    
    # 在路径中间放一个大障碍物
    obstacles = [CircleObstacle(10.0, 6.0, 5.0)]
    
    for i in range(14, 20):
        state = VehicleState(3.0 + (i-14)*0.5, 6.0, 0.0, 0.5, 0.0)
        result = planner.plan(state, obstacles)
        
        print(f"周期 {i}: 结果={'成功' if result else '失败'}, "
              f"连续失败={planner.consecutive_failures}, "
              f"TEB节点={len(planner.teb_nodes)}")

    # 场景 4: 移除障碍物，验证是否能恢复
    print("\n📊 阶段 4: 移除障碍物，验证恢复")
    print("-" * 70)
    
    state = VehicleState(5.0, 6.0, 0.0, 0.5, 0.0)
    result = planner.plan(state, [])  # 无障碍物
    
    print(f"周期 20: 结果={'成功' if result else '失败'}, "
          f"连续失败={planner.consecutive_failures}, "
          f"TEB节点={len(planner.teb_nodes)}")
    
    if result is None:
        print(f"  ❌ 预期成功但实际失败！失败原因: {planner.last_failure_reason}")
        return False
    else:
        print(f"  ✅ 成功恢复！")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！TEB 失败恢复机制工作正常")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = test_continuous_failure()
    sys.exit(0 if success else 1)
