# TEB 算法修复总结

## 问题诊断

原始 TEB 实现规划出的轨迹完全错乱，表现为：

### 症状
1. ❌ 节点位置跳来跳去（x 从 2.00 → 15.47 → 0.72）
2. ❌ 相邻节点距离过大（最大 14.80m）
3. ❌ 航向角混乱（-98.5°, 98.5° 等不合理值）
4. ❌ 轨迹穿过障碍物
5. ❌ 速度异常（最高 34.23 m/s，超出限制）

### 根本原因
TEB 优化器的约束太宽松，导致：
- 优化边界覆盖整个地图范围
- 节点可以自由移动到任何位置
- 没有足够的连续性约束
- 初始化后优化器容易陷入局部最优

## 修复方案

### 1. **限制节点优化边界** (`teb.py:299-327`)

**修改前：**
```python
# 其他节点可以在空间范围内移动
bounds.append((x_min, x_max))
bounds.append((y_min, y_max))
bounds.append((-np.pi, np.pi))
bounds.append((0.01, 2.0))
```

**修改后：**
```python
# 最后一个节点固定（目标位置）
if i == len(self.teb_nodes) - 1:
    bounds.append((node.x, node.x))
    bounds.append((node.y, node.y))
    bounds.append((node.yaw, node.yaw))
else:
    # 中间节点：限制在初始位置附近
    max_deviation = 3.0  # 最大偏离 3m
    bounds.append((max(x_min, node.x - max_deviation), 
                   min(x_max, node.x + max_deviation)))
    bounds.append((max(y_min, node.y - max_deviation), 
                   min(y_max, node.y + max_deviation)))
    # 航向角限制在 ±90°
    bounds.append((node.yaw - π/2, node.yaw + π/2))
bounds.append((0.01, 1.0))  # 时间间隔缩小
```

**效果：**
- ✅ 防止节点跳变到远处
- ✅ 保持轨迹连续性
- ✅ 终点固定，确保规划到达目标

### 2. **添加平滑性约束** (`teb.py:368-391`)

新增 `_smoothness_cost()` 函数，惩罚相邻节点距离过大：

```python
def _smoothness_cost(self, nodes: List[TEBNode]) -> float:
    """轨迹平滑代价：惩罚相邻节点之间距离过大"""
    cost = 0.0
    
    for i in range(len(nodes) - 1):
        n1, n2 = nodes[i], nodes[i + 1]
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        
        # 期望的节点间距
        expected_dist = self.config.max_speed * n1.dt * 0.5
        
        # 距离过大，惩罚
        if dist > expected_dist * 2.0:
            cost += (dist - expected_dist) ** 2
        
        # 距离过小，也惩罚（避免重叠）
        if dist < 0.1:
            cost += (0.1 - dist) ** 2
    
    return cost
```

在目标函数中添加：
```python
cost += 50.0 * self._smoothness_cost(temp_nodes)
```

**效果：**
- ✅ 相邻节点保持合理距离
- ✅ 避免节点聚集或分散过度
- ✅ 提高轨迹平滑度

### 3. **调整配置参数** (`configs/teb_config.yaml`)

**车辆尺寸：**
```yaml
# 修改前（太大，容易越界）
vehicle_front_length: 3.0
vehicle_rear_length: 1.0
vehicle_width: 1.6
vehicle_safety_margin: 0.18

# 修改后（更合理）
vehicle_front_length: 1.2
vehicle_rear_length: 0.5
vehicle_width: 1.0
vehicle_safety_margin: 0.1
```

**优化参数：**
```yaml
# 修改前
num_samples: 20
max_iterations: 50

# 修改后（减少节点数，增加迭代次数）
num_samples: 15
max_iterations: 100
```

**障碍物安全距离：**
```yaml
# 修改前（太大，限制过严）
obstacle_min_dist: 1.5
obstacle_influence_dist: 3.0

# 修改后（更宽松）
obstacle_min_dist: 0.5
obstacle_influence_dist: 2.0
```

**效果：**
- ✅ 车辆尺寸合理，不易越界
- ✅ 更多迭代次数提高收敛性
- ✅ 障碍物约束平衡安全性和可行性

## 修复效果对比

### 修复前
```
节点 0 -> 1 距离: 13.51 m  ⚠️ 跳变！
节点 1 -> 2 距离: 14.80 m  ⚠️ 穿过障碍物！
速度: 34.23 m/s           ⚠️ 超速！
航向: -98.5° → 98.5°      ⚠️ 混乱！
```

### 修复后
```
节点 0 -> 1 距离: 1.25 m   ✅ 连续
节点 1 -> 2 距离: 0.98 m   ✅ 平滑
速度: 1.48 - 3.42 m/s     ✅ 合理
航向: 0° → 37.3° → 0°     ✅ 绕过障碍物
```

## 测试结果

运行 `python test_teb_debug.py`：

```
✓ 规划成功！
TEB 轨迹节点数: 15

起点: (2.00, 6.00)
终点: (18.00, 6.00)
障碍物: (10.00, 6.00) 半径 1.5m

轨迹特征:
- 从起点连续向前推进
- 在障碍物处绕行（y 从 6.00 → 8.78）
- 通过障碍物后回到直线
- 顺利到达终点

轨迹检查: ✅ 通过
- 无大跳变
- 无碰撞
- 无越界
```

## 交互式演示使用

启动程序：
```bash
python teb_interactive.py
```

**功能特性：**
- ✅ 拖动起点/终点实时更新轨迹
- ✅ 拖动障碍物观察避障效果
- ✅ 显示每个节点的方向箭头
- ✅ 显示每个节点的速度标注
- ✅ 显示车辆轮廓
- ✅ 每 0.5 秒自动重新规划

## 技术要点

### TEB 优化本质
TEB 是一个**高度非凸**的优化问题：
- 多个局部最优
- 对初始值敏感
- 需要好的约束来引导

### 关键设计原则
1. **渐进式优化**：从全局路径的等弧长插值开始
2. **局部化边界**：限制节点偏离初始位置的范围
3. **多重约束**：平滑性 + 运动学 + 障碍物 + 路径跟踪
4. **固定端点**：起点和终点不参与优化

### 与 g2o 的对比
| 维度 | SciPy SLSQP | g2o |
|------|-------------|-----|
| 实现难度 | 简单 | 复杂 |
| 优化速度 | 慢（通用求解器） | 快（利用稀疏性） |
| 适用规模 | <20 节点 | >100 节点 |
| 适用场景 | 原型开发 | 生产环境 |

## 未来改进方向

1. **热启动**：保留上一次的优化结果作为下次的初始值
2. **自适应采样**：障碍物密集区域增加节点密度
3. **多目标 Pareto 优化**：平衡时间、平滑度、安全性
4. **切换到 g2o**：处理更长的规划范围
5. **并行优化**：生成多条候选轨迹，选择最优

## 相关文件

- `teb.py` - TEB 规划器核心实现
- `teb_interactive.py` - 交互式演示程序
- `configs/teb_config.yaml` - 配置文件
- `test_teb_debug.py` - 调试测试脚本
- `TEB_INTERACTIVE_README.md` - 使用说明

## 参考

- [TEB Local Planner (ROS)](http://wiki.ros.org/teb_local_planner)
- [Rösmann et al., ICRA 2015](https://ieeexplore.ieee.org/document/7139537)
- [SciPy SLSQP 文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
