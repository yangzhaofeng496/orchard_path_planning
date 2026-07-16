# 圆形障碍重叠率实验

固定每张地图30个圆形障碍，仅改变障碍之间的重叠程度：0%、20%、40%、60%。每级5张地图、10个搜索种子，对比RRT*、GoalBias和Hybrid，共600次规划。

- `benchmark_overlap.py`：实验与绘图脚本
- `overlap_results/`：逐次结果、汇总结果和敏感性图表
- `overlap_detail_checkpoint.csv`：实验中断时可用于续跑的断点文件
