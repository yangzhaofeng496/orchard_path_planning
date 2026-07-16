# 障碍物密度实验

设置10、20、30、40个圆形障碍，每种密度生成5张地图，每张地图使用10个搜索种子，对比 RRT*、GoalBias 和 Hybrid，共600次实验。

- `benchmark_density.py`：密度实验与绘图脚本
- `density_results/`：逐次CSV、汇总CSV和密度敏感性图表
- 默认最大迭代数为1500；脚本支持断点文件 `density_detail_checkpoint.csv`

脚本依赖上一级 `experiment_rrt_star.py`、`orchard_environment.py` 等项目模块。
