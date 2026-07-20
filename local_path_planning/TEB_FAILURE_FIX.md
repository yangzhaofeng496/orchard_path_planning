# TEB 持续失败问题修复方案

## 🔍 问题总结

**现象**：TEB 规划器在简单场景下也持续失败，失败后无法恢复

**根因**：
1. **SLSQP 约束容差过严**（1e-4）导致优化结果被错误拒绝
2. **约束检查的容差设置不一致**（日志用 1.5，判断用 1.5e-2）
3. **失败后清空节点的阈值过高**（连续 5 次才清空）
4. **缺乏明确的失败后强制重新初始化机制**

## ✅ 修复方案

### 修复 1: 放宽 SLSQP 约束容差
```python
# 修改前（第 543 行）
tolerance = 1.5e-2 if self.active_solver == 'g2o' else 1e-4

# 修改后
tolerance = 1.5e-2  # 统一使用 0.015，对 SLSQP 和 g2o 都合理
```

**理由**：
- 非线性优化很难达到 1e-4 的数值精度
- 0.015 的容差足以保证轨迹可行性
- 与 ROS teb_local_planner 的容差设置一致

### 修复 2: 统一约束检查日志的容差
```python
# 修改前（第 459 行）
tolerance = 1.5 if self.active_solver == 'g2o' else 1e-4

# 修改后
tolerance = 1.5e-2  # 与实际判断容差一致
```

**理由**：
- 避免日志显示的容差与实际判断不一致，造成困惑
- 统一使用 0.015

### 修复 3: 降低强制重置阈值，更快清空无效节点
```python
# 修改前（第 192 行）
if self.consecutive_failures >= 5:

# 修改后
if self.consecutive_failures >= 3:
```

**理由**：
- 3 次失败后就清空节点，避免继续使用无效配置
- 更快触发冷启动，提高恢复成功率

### 修复 4: 添加 need_reinit 标志，失败后强制重新初始化
```python
# 在 __init__ 中添加
self.need_reinit = False

# 在 plan() 开头添加强制重新初始化逻辑
if self.need_reinit:
    self._log("强制重新初始化 TEB", force=True)
    self.teb_nodes = []
    self.need_reinit = False

# 在失败时设置标志
if not success or len(self.teb_nodes) < 2:
    ...
    self.need_reinit = True  # 下次强制重新初始化
    return None

if collision is not None:
    ...
    self.need_reinit = True  # 下次强制重新初始化
    return None
```

**理由**：
- 明确的状态标志，确保失败后下一周期重新初始化
- 避免继续使用失败轨迹的热启动
- 符合用户期望的恢复流程

### 修复 5: 增强日志，验证恢复流程
```python
# 在 _initialize_teb 中
self._log(f"🔄 initTrajectoryToGoal: TEB节点={len(self.teb_nodes)}", force=True)

# 在 _optimize_teb 开始时
self._log("🔧 optimizeTEB begin", force=True)

# 在失败时
self._log(f"❌ cycle={self.plan_count} optimizeTEB failed", force=True)
self._log(f"🧹 cycle={self.plan_count} clear TEB, need_reinit=true", force=True)
```

## 📊 预期效果

修复后的恢复流程：
```
cycle=1  正常规划成功
cycle=2  添加障碍物导致失败
         ❌ optimizeTEB failed
         🧹 clear TEB, need_reinit=true
         输出零速度，返回失败
cycle=3  移除障碍物，继续调用规划器
         🔄 initTrajectoryToGoal
         🔧 optimizeTEB begin
         ✅ 规划成功，恢复
```

## 🎯 修改文件

只需修改一个文件：`teb/teb.py`

关键修改点：
- 第 54 行：添加 `self.need_reinit = False`
- 第 165-169 行：添加强制重新初始化逻辑
- 第 192 行：降低阈值 5 → 3
- 第 201-202 行：设置 `need_reinit = True`
- 第 241 行：设置 `need_reinit = True`
- 第 281 行：增强日志
- 第 394 行：增强日志
- 第 459 行：统一容差 1.5 → 1.5e-2
- 第 543 行：放宽容差 1e-4 → 1.5e-2

## ✅ 验证清单

- [ ] 简单场景（无障碍物）能够成功规划
- [ ] 失败后能够立即恢复（1-2 个周期内）
- [ ] 连续失败 3 次后触发强制重置
- [ ] 日志显示完整的恢复流程
- [ ] 约束容差放宽后成功率提高
