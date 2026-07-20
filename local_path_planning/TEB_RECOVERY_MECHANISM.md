# TEB 局部规划器失败恢复机制

## 📋 概述

TEB 规划器已增强失败恢复机制，确保**单次规划失败只影响当前控制周期，不会导致规划器永久停止**。

## ✅ 增强功能

### 1. **单次失败后自动恢复**
- **行为**：失败后不清空 `teb_nodes`，下一周期自动从当前位置重建 TEB 并重新优化
- **实现**：`_update_teb()` 每周期调用 `_rebuild_teb()`，完全基于当前状态重新构建轨迹
- **验证**：
  ```
  周期1: 成功=True  ✅
  周期2: 成功=False ❌ 失败但节点保留
  周期3: 成功=True  ✅ 自动恢复
  ```

### 2. **连续失败计数器**
- **属性**：`self.consecutive_failures`（在 `__init__` 中初始化）
- **更新逻辑**：
  - 规划失败时：`consecutive_failures += 1`
  - 规划成功时：`consecutive_failures = 0`（重置）
  - 设置新路径时：`consecutive_failures = 0`（重置）

### 3. **强制重置机制**
- **触发条件**：连续失败 ≥ 5 次
- **执行动作**：
  - 清空 TEB 节点：`self.teb_nodes = []`
  - 记录日志：`⚠️ 连续失败 N 次，强制重置 TEB 以恢复`
  - 下一周期将从当前位置重新初始化完整的 TEB
- **恢复能力**：即使连续失败 6+ 次，回到正常位置后仍能恢复

### 4. **恢复日志显示**
- **触发条件**：从失败状态（`consecutive_failures > 0`）恢复到成功
- **日志内容**：`✅ 从失败中恢复！周期 N`
- **用途**：帮助用户监控规划器健康状态

### 5. **混合场景处理**
- **支持场景**：失败-成功-失败-成功交替
- **验证结果**：每次失败后都能正确恢复，连续失败计数器正确重置

## 🔍 关键设计原则

### 原则 1: 失败不传播
```python
# ❌ 错误做法（旧版本可能存在）
if not success:
    self.teb_nodes = []  # 立即清空，下次无法热启动
    return None

# ✅ 正确做法（当前实现）
if not success:
    self.consecutive_failures += 1  # 只记录失败
    # 保留 teb_nodes，为下次规划提供热启动基础
    return None
```

### 原则 2: 每周期强制重建
```python
def _update_teb(self, current_state: VehicleState):
    """每周期从当前位置重新构建 TEB，不依赖上次结果"""
    self._rebuild_teb(current_state)
```

### 原则 3: 独立优化器状态
- 每次 `_optimize_teb()` 都从新的初值 `x0` 开始
- 失败不会污染优化器内部状态
- 每周期的优化是独立的

### 原则 4: 渐进式恢复策略
```
失败 1-4 次：保留节点，热启动优化
失败 5 次：   强制重置，冷启动优化
恢复成功：   重置计数器，正常运行
```

## 📊 测试验证

### 测试场景 1: 单次失败恢复
```python
周期1: 位置=(2.0, 6.0)  → 成功  ✅
周期2: 位置=(50.0, 50.0) → 失败  ❌ (超出边界)
周期3: 位置=(3.0, 6.0)  → 成功  ✅ (自动恢复)
```
**结果**：✅ 通过

### 测试场景 2: 连续失败强制重置
```python
周期1-4: 位置=(50.0, 50.0) → 连续失败 4 次
周期5:   位置=(50.0, 50.0) → 失败，触发强制重置，TEB 节点清空
周期6:   位置=(50.0, 50.0) → 继续失败
恢复:    位置=(3.0, 6.0)  → 重新初始化，规划成功 ✅
```
**结果**：✅ 通过

### 测试场景 3: 混合失败场景
```python
周期1: 成功 → 失败=0
周期2: 失败 → 失败=1
周期3: 成功 → 失败=0 ✅ 恢复
周期4: 失败 → 失败=1
周期5: 成功 → 失败=0 ✅ 再次恢复
```
**结果**：✅ 通过

## 🎯 使用建议

### 外层控制循环的正确写法

```python
# ✅ 推荐做法
planner = TEBPlanner(config, bounds)
planner.set_global_path(global_path)

while not reached_goal:
    result = planner.plan(current_state, obstacles)
    
    if result is not None:
        # 规划成功，执行控制
        execute_control(result.control)
    else:
        # 规划失败，使用后备策略（停车、减速等）
        execute_fallback_control()
        
        # 检查是否陷入长期失败
        if planner.consecutive_failures >= 10:
            # 可能需要重新规划全局路径
            replan_global_path()
    
    # 关键：无论成功或失败，下一周期继续调用 plan()
    update_state()
```

### 不要做的事情

```python
# ❌ 错误做法 1: 失败后不再调用
result = planner.plan(state, obstacles)
if result is None:
    return  # 错误：永久停止规划

# ❌ 错误做法 2: 失败后清空全局路径
result = planner.plan(state, obstacles)
if result is None:
    planner.set_global_path([])  # 错误：导致后续规划永久失败

# ❌ 错误做法 3: 失败后手动清空 TEB
result = planner.plan(state, obstacles)
if result is None:
    planner.teb_nodes = []  # 错误：破坏恢复机制
```

## 🔧 配置参数

### 强制重置阈值
```python
# 在 teb.py 中可以调整（目前硬编码为 5）
if self.consecutive_failures >= 5:
    # 触发强制重置
```

如果需要调整，可以修改为配置参数：
```yaml
# configs/teb_config.yaml
teb:
  max_consecutive_failures: 5  # 建议值：3-10
```

## 📈 监控指标

在日志中关注以下指标：

1. **连续失败次数**
   - `consecutive_failures < 3`：正常
   - `3 <= consecutive_failures < 5`：需要注意
   - `consecutive_failures >= 5`：触发强制重置

2. **恢复成功标志**
   - 看到 `✅ 从失败中恢复！` 表示恢复机制正常工作

3. **失败原因**
   - `TEB节点不足`：路径问题
   - `数值优化失败`：约束过严或场景过难
   - `轨迹碰撞`：障碍物太近或边界问题

## 🐛 故障排查

### 现象：规划器一次失败后永久停止

**可能原因**：
1. 外层控制循环在失败后停止调用 `plan()`
2. 全局路径被意外清空
3. 车辆持续处于失败条件（超出边界、目标太远等）

**排查步骤**：
1. 在失败后添加日志，确认是否继续调用 `plan()`
2. 检查 `planner.global_path` 是否为空
3. 检查 `planner.consecutive_failures` 的值
4. 运行 `test_recovery.py` 验证恢复机制本身是否正常

### 现象：连续失败后仍无法恢复

**可能原因**：
1. 车辆位置持续超出边界
2. 目标距离过远（>30m）
3. 障碍物完全阻塞路径

**解决方法**：
1. 检查车辆位置是否在合理范围内
2. 使用全局路径分段，降低局部规划距离
3. 移除或调整障碍物位置

## 📝 更新日志

**2024-XX-XX**
- ✅ 添加连续失败计数器 `consecutive_failures`
- ✅ 添加强制重置机制（连续失败 ≥ 5 次）
- ✅ 添加恢复日志显示
- ✅ 完善失败恢复测试用例
- ✅ 创建完整文档

## 🧪 测试

运行完整测试：
```bash
cd local_path_planning
python test_recovery.py
```

预期输出：
```
🎉 所有测试通过！TEB 失败恢复机制工作正常

增强功能总结:
1. ✅ 单次失败后自动恢复
2. ✅ 连续失败计数器
3. ✅ 强制重置机制
4. ✅ 恢复日志显示
5. ✅ 混合场景处理
```

## 📚 相关文件

- **核心实现**：`teb/teb.py` - `TEBPlanner` 类
- **测试脚本**：`test_recovery.py` - 完整的恢复机制测试
- **配置文件**：`configs/teb_config.yaml` - TEB 参数配置
- **本文档**：`TEB_RECOVERY_MECHANISM.md` - 恢复机制说明

## 💡 总结

TEB 规划器的失败恢复机制确保了鲁棒性和可靠性：

- **自动恢复**：单次失败不影响后续规划
- **渐进式重置**：连续失败时自动采取更激进的恢复策略
- **可观测性**：清晰的日志帮助监控和调试
- **无需干预**：外层控制循环只需持续调用 `plan()`，规划器会自动处理失败和恢复

这确保了 TEB 规划器在复杂和动态环境中的稳定运行。
