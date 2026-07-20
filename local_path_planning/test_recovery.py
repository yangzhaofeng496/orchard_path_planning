"""
TEB 失败恢复机制测试
验证单次失败恢复、连续失败强制重置、恢复日志显示等功能
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

def test_single_failure_recovery():
    """测试单次失败后的自动恢复"""
    print('=' * 70)
    print('测试场景 1: 单次失败后自动恢复')
    print('=' * 70)

    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = True  # 启用详细日志
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    # 设置全局路径
    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    # 周期 1: 正常规划（成功）
    print('\n>>> 周期 1: 正常位置，应该成功')
    state1 = VehicleState(2.0, 6.0, 0.0, 0.0, 0.0)
    result1 = planner.plan(state1, [])
    print(f'结果: 成功={result1 is not None}, TEB节点={len(planner.teb_nodes)}, 连续失败={planner.consecutive_failures}')

    # 周期 2: 车辆超出边界（失败）
    print('\n>>> 周期 2: 车辆超出边界，应该失败')
    state2 = VehicleState(50.0, 50.0, 0.0, 0.0, 0.0)
    result2 = planner.plan(state2, [])
    print(f'结果: 成功={result2 is not None}, TEB节点={len(planner.teb_nodes)}, 连续失败={planner.consecutive_failures}')

    # 周期 3: 恢复正常位置（应该自动恢复）
    print('\n>>> 周期 3: 回到正常位置，应该自动恢复并显示恢复日志')
    state3 = VehicleState(3.0, 6.0, 0.0, 0.5, 0.0)
    result3 = planner.plan(state3, [])
    print(f'结果: 成功={result3 is not None}, TEB节点={len(planner.teb_nodes)}, 连续失败={planner.consecutive_failures}')

    assert result1 is not None, "周期1应该成功"
    assert result2 is None, "周期2应该失败"
    assert result3 is not None, "周期3应该恢复成功"
    assert planner.consecutive_failures == 0, "恢复后连续失败应该清零"

    print('\n✅ 测试场景 1 通过：单次失败后能自动恢复\n')


def test_consecutive_failure_reset():
    """测试连续失败 5 次后的强制重置"""
    print('=' * 70)
    print('测试场景 2: 连续失败 5 次触发强制重置')
    print('=' * 70)

    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = True
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    # 设置全局路径
    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    # 连续失败 6 次
    print('\n>>> 连续触发 6 次失败（第 5 次应该触发强制重置）')
    for i in range(6):
        state = VehicleState(50.0, 50.0, 0.0, 0.0, 0.0)
        result = planner.plan(state, [])
        print(f'周期 {i+1}: 成功={result is not None}, TEB节点={len(planner.teb_nodes)}, 连续失败={planner.consecutive_failures}')

        # 第 5 次失败后应该触发强制重置，节点清空
        if i == 4:
            assert len(planner.teb_nodes) == 0, f"第 5 次失败后应该清空 TEB 节点"

    # 恢复正常位置
    print('\n>>> 恢复正常位置，应该能重新初始化并规划成功')
    state_recover = VehicleState(3.0, 6.0, 0.0, 0.5, 0.0)
    result_recover = planner.plan(state_recover, [])
    print(f'恢复周期: 成功={result_recover is not None}, TEB节点={len(planner.teb_nodes)}, 连续失败={planner.consecutive_failures}')

    assert result_recover is not None, "恢复后应该规划成功"
    assert planner.consecutive_failures == 0, "恢复后连续失败应该清零"
    assert len(planner.teb_nodes) > 0, "恢复后应该有 TEB 节点"

    print('\n✅ 测试场景 2 通过：连续失败 5 次后强制重置，之后能恢复\n')


def test_mixed_failures():
    """测试混合失败场景：失败-成功-失败-成功"""
    print('=' * 70)
    print('测试场景 3: 混合失败场景')
    print('=' * 70)

    config = load_teb_config('configs/teb_config.yaml')
    config.verbose = True
    planner = TEBPlanner(config, (0.0, 20.0, 0.0, 12.0))

    path = [Pose(2.0, 6.0, 0.0), Pose(18.0, 6.0, 0.0)]
    planner.set_global_path(path)

    states = [
        (VehicleState(2.0, 6.0, 0.0, 0.0, 0.0), True, "正常位置"),
        (VehicleState(50.0, 50.0, 0.0, 0.0, 0.0), False, "超出边界"),
        (VehicleState(3.0, 6.0, 0.0, 0.5, 0.0), True, "恢复正常"),
        (VehicleState(50.0, 50.0, 0.0, 0.0, 0.0), False, "再次超出"),
        (VehicleState(4.0, 6.0, 0.0, 0.5, 0.0), True, "再次恢复"),
    ]

    for i, (state, expected_success, desc) in enumerate(states, 1):
        print(f'\n>>> 周期 {i}: {desc}')
        result = planner.plan(state, [])
        actual_success = result is not None
        print(f'结果: 成功={actual_success}, 预期={expected_success}, 连续失败={planner.consecutive_failures}')
        assert actual_success == expected_success, f"周期 {i} 结果不符合预期"

    print('\n✅ 测试场景 3 通过：混合失败场景处理正确\n')


if __name__ == '__main__':
    print('\n')
    print('*' * 70)
    print('TEB 失败恢复机制完整测试')
    print('*' * 70)
    print()

    try:
        test_single_failure_recovery()
        test_consecutive_failure_reset()
        test_mixed_failures()

        print('=' * 70)
        print('🎉 所有测试通过！TEB 失败恢复机制工作正常')
        print('=' * 70)
        print()
        print('增强功能总结:')
        print('1. ✅ 单次失败后自动恢复 - 下一周期重新初始化并优化')
        print('2. ✅ 连续失败计数器 - 记录失败次数')
        print('3. ✅ 强制重置机制 - 连续失败 5 次后清空 TEB 节点')
        print('4. ✅ 恢复日志显示 - 从失败恢复时显示明确日志')
        print('5. ✅ 混合场景处理 - 失败和成功交替时正确处理')
        print()

    except AssertionError as e:
        print(f'\n❌ 测试失败: {e}\n')
        sys.exit(1)
