# Tangent 采样优化总结

## 问题背景

GoalBias+Tangent 配置虽然规划时间更短，但路径长度比 GoalBias 高约 15%。

## 优化目标

将路径增幅从 15% 降低到 5% 以内，同时保持规划速度优势。

## 核心修改

### 1. 二维几何绕行比评价（最重要）

**原理**：使用纯几何距离评价切向候选，避免引入阿克曼约束或曲率代价。

```python
direct_length = distance(current_pose, goal)
candidate_length = distance(current_pose, tangent_target) + distance(tangent_target, goal)
detour_ratio = candidate_length / max(direct_length, 1e-6)

cost = (
    candidate_length
    + tangent_detour_weight * max(0.0, detour_ratio - 1.0)
    + remaining_blocker_weight * remaining_blocking_cluster_count
    + side_switch_penalty * side_switched
)
```

**参数**：
- `tangent_detour_weight = 20.0`：重度惩罚绕行
- `remaining_blocker_weight = 3.0`：惩罚未清除的障碍簇
- `side_switch_penalty = 2.0`：轻微惩罚换侧

### 2. 多候选延伸距离

**修改前**：每侧只生成 1 个固定延伸距离的切向目标  
**修改后**：每侧测试 4 个延伸距离 `(0.0, 0.2, 0.4, 0.6)`，从所有合法候选中选择综合代价最小的。

### 3. 候选点合法性检查

必须满足：
- ✅ 位于地图边界内
- ✅ 不位于任何膨胀障碍物内
- ✅ current_pose 到候选点的线段不被其他障碍物阻挡
- ✅ 数值有限（非 NaN/inf）

**优先规则**：优先选择"候选点到最终目标的线段已经不再穿过当前障碍簇"的候选。

### 4. 限制绕行比

**单障碍簇**：`max_detour_ratio = 1.10`（仅允许 10% 绕行）  
**多障碍簇**：`max_detour_ratio = 1.20`（仅允许 20% 绕行）

**策略**：
- 优先从不超过阈值的候选中选择
- 若所有候选都超过阈值，保留绕行比最小的合法候选
- **不能直接关闭切向引导**（避免陷入死循环）

### 5. 条件触发切向采样（最关键）

**修改前**：固定 10% 概率使用切向采样  
**修改后**：根据场景动态调整

| 场景 | 切向概率 |
|------|---------|
| 无遮挡 | 0.00 |
| 单障碍簇 + 近距离（≤12m） | 0.02 |
| 多障碍簇 + 近距离（≤12m） | 0.05 |
| 障碍物远（>12m） | 0.00 |
| 已找到可行路径 | 0.00 |

**关键**：取消的切向概率只能转给 Uniform，**不能提高 Goal Bias 概率**。

### 6. 减少切向引导持续时间

**最大引导次数**：`max_guidance_updates = 15`  
**目标容差**：`tangent_target_tolerance = 0.8`

**清除条件**（满足任一立即清除）：
- ✅ 当前节点到目标已经无遮挡
- ✅ 已通过当前障碍簇
- ✅ 到达切向目标附近
- ✅ 已找到首次可行路径
- ✅ 引导次数超过 15
- ✅ 切向目标越界或进入障碍物

### 7. 全局规划不评价 yaw

**修改**：采样点的 yaw 仅设置为指向最终目标：

```python
yaw = atan2(goal.y - sample_y, goal.x - sample_x)
```

**不添加**：
- ❌ 转向代价
- ❌ 航向差拒绝
- ❌ 曲率代价
- ❌ Dubins/Reeds-Shepp 路径长度

### 8. 修复 nearest_blocking_distance 覆盖 bug

**问题**：`maybe_shrink_rectangle()` 方法会覆盖 `update_tangent_guidance()` 中基于直线阻挡计算的 `nearest_blocking_distance`。

**解决**：让 `maybe_shrink_rectangle()` 只计算局部变量 `nearest_distance`，不覆盖 `self.nearest_blocking_distance`。

## 测试验证

### ✅ 综合测试通过（5/5）

1. **无遮挡** → 切向概率 = 0.00 ✅
2. **单障碍物阻挡（近距离 5.8m）** → 切向概率 = 0.02 ✅
3. **单障碍物阻挡（远距离 15.8m）** → 切向概率 = 0.00 ✅
4. **多障碍物阻挡（7m）** → 切向概率 = 0.05 ✅
5. **已找到可行路径** → 切向概率 = 0.00 ✅

### ✅ 代码验证

- ✅ `python -m py_compile` 通过
- ✅ 概率总和始终等于 1.0
- ✅ 无遮挡场景切向采样次数为 0
- ✅ 首次解后不再进行切向采样

## 预期效果

1. **路径质量提升**：路径长度增幅从 15% 降低到 5% 以内
2. **保持速度优势**：切向采样只在真正需要时启用（频率大幅降低）
3. **避免过度绕行**：严格限制绕行比（单簇 10%，多簇 20%）
4. **快速退出**：找到首次可行路径后立即停止切向采样

## 文件修改

- **主文件**：`global_path_planning/innovation_sample/hybrid_sampler.py`
- **测试脚本**：`experiment/env/test_tangent_final.py`

## 后续建议

1. 运行完整消融实验，对比 GoalBias vs GoalBias+Tangent 的路径长度和规划时间
2. 如果路径增幅仍 >5%，可以进一步降低多障碍簇的切向概率（0.05 → 0.03）
3. 如果规划时间明显增加，可以适当放宽绕行比限制（1.10 → 1.15）

## 关键设计原则

1. **几何优先**：全局规划只考虑二维几何距离，不引入运动学约束
2. **条件触发**：切向采样只在真正被阻挡且距离较近时启用
3. **严格限制**：通过绕行比上限避免生成过长路径
4. **快速退出**：找到首次可行路径后立即停止优化采样
5. **概率守恒**：取消的切向概率转给 Uniform，保持 Goal Bias 稳定
