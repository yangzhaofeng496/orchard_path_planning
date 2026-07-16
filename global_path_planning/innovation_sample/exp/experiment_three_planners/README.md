# 三规划器典型场景实验

对比 RRT*、GoalBias 和 Hybrid，在单圆封锁、交错果树、行间通道三个场景上各运行30个搜索随机种子，共270次实验。

- `test_three_planners.py`：测试与绘图脚本
- `three_planner_results/`：正式实验CSV、PNG、PDF、SVG和分析报告
- `three_planner_results_smoke/`：2个种子的冒烟测试结果

脚本依赖上一级 `experiment_rrt_star.py`、`orchard_environment.py` 等项目模块。
