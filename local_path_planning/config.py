"""
局部规划器配置类
"""
from dataclasses import dataclass


@dataclass
class DWAConfig:
    """DWA 规划器配置"""
    # 车辆参数
    wheel_base: float = 2.5
    max_speed: float = 2.0
    max_accel: float = 1.0
    max_decel: float = 1.0
    max_steer_deg: float = 30.0
    max_steer_rate_deg: float = 70.0

    # 采样参数
    speed_samples: int = 7
    steer_samples: int = 19

    # 控制参数
    dt: float = 0.05
    predict_time: float = 2.8
    lookahead_distance: float = 3.2
    goal_tolerance: float = 0.45

    # 车辆几何
    vehicle_front_length: float = 3.0
    vehicle_rear_length: float = 1.0
    vehicle_width: float = 1.6
    vehicle_safety_margin: float = 0.18

    # 代价函数权重
    goal_cost_weight: float = 3.0
    path_cost_weight: float = 2.0
    heading_cost_weight: float = 0.8
    obstacle_cost_weight: float = 3.5
    speed_cost_weight: float = 0.45
    steering_cost_weight: float = 0.2
    steering_change_cost_weight: float = 0.5
    progress_reward_weight: float = 0.04


@dataclass
class TEBConfig:
    """TEB 规划器配置"""
    # 车辆参数
    wheel_base: float = 2.5
    max_speed: float = 2.0
    min_speed: float = 0.20
    preferred_speed: float = 1.20
    max_accel: float = 1.0
    max_decel: float = 1.0
    max_steer_deg: float = 30.0

    # 优化参数
    dt: float = 0.1
    num_samples: int = 20
    max_iterations: int = 50
    goal_tolerance: float = 0.5
    goal_yaw_tolerance_deg: float = 5.0
    lookahead_distance: float = 6.0
    max_dt: float = 1.5

    # 非线性求解器：slsqp、g2o 或 auto。
    # g2o 需要项目级 Python/C++ 适配器，Homebrew 的 CLI 本身不能求解自定义 TEB 边。
    solver: str = "slsqp"
    solver_fallback: bool = True

    # 车辆几何
    vehicle_front_length: float = 3.0
    vehicle_rear_length: float = 1.0
    vehicle_width: float = 1.6
    vehicle_safety_margin: float = 0.18

    # 优化权重
    w_time: float = 1.0              # 时间最优权重
    w_obstacle: float = 50.0         # 障碍物代价权重
    w_kinematics: float = 20.0       # 运动学约束权重
    w_acceleration: float = 5.0      # 加速度平滑权重
    w_omega: float = 10.0            # 角速度约束权重
    w_path: float = 1.0              # 路径跟踪权重
    weight_path_yaw: float = 1.0     # 路径航向角跟踪权重
    weight_goal_yaw: float = 15.0    # 真实终点航向角权重
    w_velocity: float = 8.0          # 期望速度权重，避免首段速度退化为零
    w_steering: float = 3.0          # 转向使用权重，避免无障碍时持续打满方向

    # 障碍物安全参数
    obstacle_min_dist: float = 1.5           # 最小安全距离（米）
    obstacle_influence_dist: float = 3.0     # 障碍物影响范围（米）

    # 诊断日志
    debug_log: bool = False
    log_interval: int = 10

    # 连续轨迹硬碰撞检测的空间采样间隔（米）
    collision_check_resolution: float = 0.10

    # 仅此前瞻路径距离内的碰撞触发立即停车（米）
    collision_stop_horizon: float = 3.0
