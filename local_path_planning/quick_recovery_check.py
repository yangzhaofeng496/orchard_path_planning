#!/usr/bin/env python3
"""
TEB 失败恢复机制快速检查脚本
用于验证 TEB 规划器在失败后能否正常恢复
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose

def quick_check():
    """快速检查 TEB 失败恢复是否正常工作"""
    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = False  # 关闭详细日志
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    # 测试：成功-失败-恢复
    states = [
        (VehicleState(2.0, 6.0, 0.0, 0.0, 0.0), True, "初始规划"),
        (VehicleState(50.0, 50.0, 0.0, 0.0, 0.0), False, "触发失败"),
        (VehicleState(3.0, 6.0, 0.0, 0.5, 0.0), True, "自动恢复"),
    ]

    print("TEB 失败恢复机制快速检查")
    print("=" * 50)

    all_passed = True
    for i, (state, expected, desc) in enumerate(states, 1):
        result = planner.plan(state, [])
        actual = result is not None
        status = "✅" if actual == expected else "❌"

        print(f"{status} 周期 {i}: {desc}")
        print(f"   预期={expected}, 实际={actual}, 连续失败={planner.consecutive_failures}")

        if actual != expected:
            all_passed = False
            print(f"   ❌ 失败原因: {planner.last_failure_reason}")

    print("=" * 50)
    if all_passed:
        print("✅ 失败恢复机制工作正常！")
        return 0
    else:
        print("❌ 失败恢复机制异常，请运行完整测试：")
        print("   python test_recovery.py")
        return 1

if __name__ == '__main__':
    sys.exit(quick_check())
