# Adaptive Ackermann-TEB with DWA Feedback

该模块只编排项目中已有的 `TEBPlanner`、`DWAPlanner`、Ackermann 车辆模型和
RRT* 路径，不复制其核心算法。

## 流程

```text
RRT* global_path
        │
        ▼
AdaptiveWindowSelector ── density/free-space/curvature/speed
        │ local_reference_path
        ▼
existing TEBPlanner
        │ candidate trajectory
        ▼
existing DWAPlanner as evaluator
        │ score/risk/tracking/kinematic feasibility
        ▼
ParameterManager ── bounded weight and window update
        │
        └── repeat TEB → DWA (maximum N iterations)
```

## 输入接口

```python
planner.set_global_path(list[Pose])
result = planner.plan(robot_state, local_costmap)
```

`local_costmap` 支持以下任意形式：

- `list[CircleObstacle]`
- 实现 `get_obstacles()` 的对象
- 具有 `.obstacles` 属性的对象

这样可通过一个很薄的适配器连接已有 costmap，不要求修改融合规划器。

## 输出

`AdaptivePlannerResult.trajectory` 中每个元素为：

```text
x, y, yaw, v, steering
```

结果同时包含动态窗口指标、DWA 评价、每轮参数调整、反馈次数和计算时间。
累计统计位于 `planner.statistics`。

## 演示

```bash
cd /Users/yangzhaofeng/VsCodeProject/orchard_path_planning
python3 -m local_path_planning.adaptive_teb_dwa.demo --seed 1
```

无界面验证：

```bash
MPLBACKEND=Agg python3 -m local_path_planning.adaptive_teb_dwa.demo \
  --seed 1 --obstacles 30 --no-show
```

参数统一位于 `config.yaml`。
