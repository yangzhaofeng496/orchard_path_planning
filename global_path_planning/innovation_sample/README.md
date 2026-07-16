# innovation_sample 文件分类

## 公共核心

根目录中的 `ackermann_rrt_star.py`、`hybrid_sampler.py`、`orchard_environment.py`、`experiment_rrt_star.py` 以及环境 NPZ/PNG 是所有实验共享的算法和环境文件。

## 实验目录

- `experiment_three_planners/`：RRT*、GoalBias、Hybrid 三规划器典型场景实验。
- `experiment_density/`：10/20/30/40 个圆形障碍物的密度敏感性实验。
- `experiment_overlap/`：圆形障碍重叠率敏感性实验（0/20/40/60%）。
- `experiment_gap/`：果园行间净间隙敏感性实验（2.5–7.0 m）。
- `experiment_safety/`：车辆安全裕度敏感性实验（0.00–0.45 m）。
- `oag_hrrt_dwa/`：OAG-HRRT*-DWA 交互式系统，将果园 Hybrid RRT* 全局规划与完整 Ackermann-DWA 动态避障结合。
- `archive/legacy_results/`：早期实验结果、场景图和未完成实验的归档。

每个正式实验目录都包含对应脚本、README、CSV数据和图表结果。
