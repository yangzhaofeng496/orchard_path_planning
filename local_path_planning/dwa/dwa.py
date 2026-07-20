"""
DWA (Dynamic Window Approach) 局部规划器
"""

import sys
import os

# 添加 vehicle 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vehicle'))

from ackerman_dwa import (
    AckermannDWA,
    DWAConfig as OriginalDWAConfig,
    DWAState as OriginalDWAState,
    VehicleGeometry,
    Pose as OriginalPose,
    CircleObstacle as OriginalCircleObstacle,
)

# 条件导入：支持作为模块和独立脚本运行
if __name__ == "__main__":
    # 独立运行时使用绝对导入
    from base import (
        LocalPlanner,
        LocalPlannerConfig,
        LocalPlanResult,
        VehicleState,
        Pose,
        Control,
        CircleObstacle,
    )
    from config import DWAConfig
else:
    # 作为模块导入时使用相对导入
    from ..base import (
        LocalPlanner,
        LocalPlannerConfig,
        LocalPlanResult,
        VehicleState,
        Pose,
        Control,
        CircleObstacle,
    )
    from ..config import DWAConfig

from typing import List, Optional
import math


class DWAPlanner(LocalPlanner):
    """DWA 局部规划器"""

    def __init__(self, config: DWAConfig, bounds: tuple):
        super().__init__(config)
        self.bounds = bounds
        self.dwa_instance: Optional[AckermannDWA] = None

    def set_global_path(self, path: List[Pose]):
        """设置全局路径并初始化 DWA"""
        super().set_global_path(path)

        # 转换为 DWA 使用的格式
        dwa_path = [
            OriginalPose(p.x, p.y, p.yaw)
            for p in path
        ]

        # 构建 DWA 配置
        dwa_config = OriginalDWAConfig(
            wheel_base=self.config.wheel_base,
            max_speed=self.config.max_speed,
            max_accel=self.config.max_accel,
            max_decel=self.config.max_decel,
            max_steer=math.radians(self.config.max_steer_deg),
            max_steer_rate=math.radians(self.config.max_steer_rate_deg),
            speed_sample_count=self.config.speed_samples,
            steering_sample_count=self.config.steer_samples,
            dt=self.config.dt,
            predict_time=self.config.predict_time,
            lookahead_distance=self.config.lookahead_distance,
            goal_tolerance=self.config.goal_tolerance,
            goal_cost_weight=self.config.goal_cost_weight,
            path_cost_weight=self.config.path_cost_weight,
            heading_cost_weight=self.config.heading_cost_weight,
            obstacle_cost_weight=self.config.obstacle_cost_weight,
            speed_cost_weight=self.config.speed_cost_weight,
            steering_cost_weight=self.config.steering_cost_weight,
            steering_change_cost_weight=self.config.steering_change_cost_weight,
            progress_reward_weight=self.config.progress_reward_weight,
        )

        # 车辆几何
        vehicle_geom = VehicleGeometry(
            front_length=self.config.vehicle_front_length,
            rear_length=self.config.vehicle_rear_length,
            width=self.config.vehicle_width,
            safety_margin=self.config.vehicle_safety_margin,
        )

        # 创建 DWA 实例
        self.dwa_instance = AckermannDWA(
            config=dwa_config,
            vehicle=vehicle_geom,
            global_path=dwa_path,
            bounds=self.bounds,
        )

    def plan(
        self,
        state: VehicleState,
        obstacles: List[CircleObstacle],
    ):
        """执行 DWA 规划，返回原始结果用于可视化"""
        if self.dwa_instance is None:
            # 返回空结果
            class EmptyResult:
                best = None
                candidates = []
                local_goal = Pose(0, 0, 0)
                nearest_path_index = 0
            return EmptyResult()

        # 转换状态格式
        dwa_state = OriginalDWAState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            speed=state.speed,
            steering=state.steering,
        )

        # 转换障碍物格式
        dwa_obstacles = [
            OriginalCircleObstacle(obs.x, obs.y, obs.radius)
            for obs in obstacles
        ]

        # 调用 DWA 规划，直接返回原始结果
        return self.dwa_instance.plan(dwa_state, dwa_obstacles)


def main():
    """DWA 规划器演示"""
    import sys
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # 添加路径
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from local_path_planning import load_dwa_config, DWAPlanner
    from local_path_planning import VehicleState, Pose, CircleObstacle

    print("\n" + "=" * 70)
    print("DWA 局部规划器演示")
    print("=" * 70)

    # 1. 加载配置
    print("\n[1] 加载配置文件...")
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'dwa_config.yaml')
    config = load_dwa_config(config_path)
    print(f"    ✅ 配置加载成功")
    print(f"    - 最大速度: {config.max_speed} m/s")
    print(f"    - 最大转向角: {config.max_steer_deg}°")
    print(f"    - 速度采样数: {config.speed_samples}")
    print(f"    - 转向采样数: {config.steer_samples}")

    # 2. 创建规划器
    print("\n[2] 创建 DWA 规划器...")
    bounds = (0, 50, 0, 50)  # (x_min, x_max, y_min, y_max)
    planner = DWAPlanner(config, bounds)
    print(f"    ✅ 规划器创建成功")

    # 3. 设置全局路径
    print("\n[3] 设置全局路径...")
    global_path = [
        Pose(5, 5, 0),
        Pose(15, 10, 0.3),
        Pose(25, 15, 0.1),
        Pose(35, 20, -0.2),
        Pose(45, 25, 0),
    ]
    planner.set_global_path(global_path)
    print(f"    ✅ 全局路径设置完成 ({len(global_path)} 个路径点)")

    # 4. 设置当前状态
    print("\n[4] 设置车辆状态...")
    current_state = VehicleState(
        x=5.0,
        y=5.0,
        yaw=0.0,
        speed=0.5,
        steering=0.0,
    )
    print(f"    - 位置: ({current_state.x:.1f}, {current_state.y:.1f})")
    print(f"    - 航向: {np.degrees(current_state.yaw):.1f}°")
    print(f"    - 速度: {current_state.speed:.2f} m/s")

    # 5. 设置障碍物
    print("\n[5] 设置障碍物...")
    obstacles = [
        CircleObstacle(20, 12, 2.0),
        CircleObstacle(30, 18, 1.5),
    ]
    print(f"    ✅ 添加 {len(obstacles)} 个障碍物")

    # 6. 执行规划
    print("\n[6] 执行 DWA 规划...")
    result = planner.plan(current_state, obstacles)

    if result.best is not None:
        print(f"    ✅ 规划成功!")
        print(f"    - 最优速度: {result.best.control.speed:.2f} m/s")
        print(f"    - 最优转向: {np.degrees(result.best.control.steering):.1f}°")
        print(f"    - 预测轨迹点数: {len(result.best.trajectory)}")
        print(f"    - 候选轨迹数: {len(result.candidates)}")
    else:
        print(f"    ❌ 规划失败!")
        return

    # 7. 可视化
    print("\n[7] 可视化结果...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制全局路径
    path_x = [p.x for p in global_path]
    path_y = [p.y for p in global_path]
    ax.plot(path_x, path_y, 'b--', linewidth=2, label='全局路径', alpha=0.6)
    ax.plot(path_x, path_y, 'bo', markersize=8)

    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.3)
        ax.add_patch(circle)
        ax.plot(obs.x, obs.y, 'rx', markersize=12, markeredgewidth=3)

    # 绘制候选轨迹（灰色）
    for candidate in result.candidates:
        traj_x = [p.x for p in candidate.trajectory]
        traj_y = [p.y for p in candidate.trajectory]
        ax.plot(traj_x, traj_y, 'gray', linewidth=0.5, alpha=0.3)

    # 绘制最优轨迹
    best_x = [p.x for p in result.best.trajectory]
    best_y = [p.y for p in result.best.trajectory]
    ax.plot(best_x, best_y, 'g-', linewidth=3, label='最优轨迹')

    # 绘制当前车辆位置
    ax.plot(current_state.x, current_state.y, 'go', markersize=15,
            label='当前位置', markeredgecolor='darkgreen', markeredgewidth=2)

    # 绘制局部目标
    ax.plot(result.local_goal.x, result.local_goal.y, 'r*',
            markersize=20, label='局部目标')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('DWA 局部规划器演示', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(os.path.dirname(__file__), 'dwa_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✅ 结果已保存到: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
