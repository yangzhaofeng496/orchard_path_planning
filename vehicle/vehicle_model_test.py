import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class VehicleState:
    """
    阿克曼车辆状态。

    x, y:
        后轴中心在世界坐标系中的位置，单位 m。

    yaw:
        车辆航向角，单位 rad。
        yaw = 0 表示车辆朝向世界坐标系 x 轴正方向。
    """
    x: float
    y: float
    yaw: float


def normalize_angle(angle: float) -> float:
    """
    将角度限制到 [-pi, pi)。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bicycle_model_step(
    state: VehicleState,
    velocity: float,
    steering_angle: float,
    wheel_base: float,
    dt: float,
) -> VehicleState:
    """
    使用运动学自行车模型推进一个时间步。

    参数
    ----------
    state:
        当前车辆状态。

    velocity:
        车辆纵向速度，单位 m/s。
        velocity > 0 表示前进。
        velocity < 0 表示倒车。

    steering_angle:
        前轮转角，单位 rad。
        steering_angle > 0 表示向左转。
        steering_angle < 0 表示向右转。

    wheel_base:
        车辆轴距，单位 m。

    dt:
        积分时间步长，单位 s。

    返回
    ----------
    下一时刻的车辆状态。
    """

    if wheel_base <= 0.0:
        raise ValueError("wheel_base 必须大于 0")

    if dt <= 0.0:
        raise ValueError("dt 必须大于 0")

    # 当前状态
    x = state.x
    y = state.y
    yaw = state.yaw

    # 欧拉积分
    next_x = x + velocity * math.cos(yaw) * dt
    next_y = y + velocity * math.sin(yaw) * dt

    yaw_rate = velocity / wheel_base * math.tan(steering_angle)
    next_yaw = yaw + yaw_rate * dt
    next_yaw = normalize_angle(next_yaw)

    return VehicleState(
        x=next_x,
        y=next_y,
        yaw=next_yaw,
    )


def simulate_vehicle(
    initial_state: VehicleState,
    velocity: float,
    steering_angle: float,
    wheel_base: float,
    dt: float,
    simulation_time: float,
):
    """
    在恒定速度、恒定前轮转角下模拟车辆运动。
    """

    state = VehicleState(
        x=initial_state.x,
        y=initial_state.y,
        yaw=initial_state.yaw,
    )

    states = [state]

    step_count = int(simulation_time / dt)

    for _ in range(step_count):
        state = bicycle_model_step(
            state=state,
            velocity=velocity,
            steering_angle=steering_angle,
            wheel_base=wheel_base,
            dt=dt,
        )
        states.append(state)

    return states


def get_vehicle_corners(
    state: VehicleState,
    front_length: float,
    rear_length: float,
    vehicle_width: float,
):
    """
    根据后轴中心状态计算车辆矩形的四个角点。

    这里暂时只用于绘制，不做碰撞检测。
    """

    half_width = vehicle_width / 2.0

    # 车辆局部坐标系中的四个角点
    local_corners = [
        (front_length, half_width),
        (front_length, -half_width),
        (-rear_length, -half_width),
        (-rear_length, half_width),
    ]

    cos_yaw = math.cos(state.yaw)
    sin_yaw = math.sin(state.yaw)

    world_corners = []

    for local_x, local_y in local_corners:
        world_x = (
            state.x
            + local_x * cos_yaw
            - local_y * sin_yaw
        )

        world_y = (
            state.y
            + local_x * sin_yaw
            + local_y * cos_yaw
        )

        world_corners.append((world_x, world_y))

    # 闭合矩形
    world_corners.append(world_corners[0])

    return world_corners


def calculate_theoretical_turning_radius(
    wheel_base: float,
    steering_angle: float,
) -> float:
    """
    计算理论转弯半径。

    R = L / tan(delta)
    """

    if abs(steering_angle) < 1e-10:
        return math.inf

    return wheel_base / math.tan(steering_angle)


def main():
    # =========================
    # 车辆参数
    # =========================
    wheel_base = 2.5

    vehicle_width = 1.6

    # 后轴中心到车头距离
    front_length = 3.0

    # 后轴中心到车尾距离
    rear_length = 1.0

    # 最大前轮转角
    max_steering_angle = math.radians(30.0)

    # =========================
    # 仿真参数
    # =========================
    velocity = 1.0

    # 第一轮测试：固定使用最大左转角
    steering_angle = -max_steering_angle

    dt = 0.02
    simulation_time = 12.0

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
    )

    states = simulate_vehicle(
        initial_state=initial_state,
        velocity=velocity,
        steering_angle=steering_angle,
        wheel_base=wheel_base,
        dt=dt,
        simulation_time=simulation_time,
    )

    x_list = [state.x for state in states]
    y_list = [state.y for state in states]
    yaw_list = [state.yaw for state in states]

    theoretical_radius = calculate_theoretical_turning_radius(
        wheel_base=wheel_base,
        steering_angle=steering_angle,
    )

    theoretical_yaw_rate = (
        velocity
        / wheel_base
        * math.tan(steering_angle)
    )

    print("=" * 50)
    print("车辆运动学模型测试")
    print("=" * 50)

    print(f"轴距 L: {wheel_base:.3f} m")
    print(
        f"前轮转角 delta: "
        f"{math.degrees(steering_angle):.3f} deg"
    )
    print(f"纵向速度 v: {velocity:.3f} m/s")

    print(
        f"理论转弯半径 R = L / tan(delta): "
        f"{theoretical_radius:.3f} m"
    )

    print(
        f"理论航向角速度 yaw_rate: "
        f"{theoretical_yaw_rate:.3f} rad/s"
    )

    print(
        f"理论航向角速度: "
        f"{math.degrees(theoretical_yaw_rate):.3f} deg/s"
    )

    print(
        f"最终位置: "
        f"x={states[-1].x:.3f}, "
        f"y={states[-1].y:.3f}"
    )

    print(
        f"最终航向角: "
        f"{math.degrees(states[-1].yaw):.3f} deg"
    )

    # =========================
    # 绘图
    # =========================
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Kinematic Bicycle Model Test")

    # 完整轨迹
    ax.plot(
        x_list,
        y_list,
        linestyle="--",
        label="rear axle trajectory",
    )

    # 起点
    ax.scatter(
        initial_state.x,
        initial_state.y,
        marker="o",
        s=80,
        label="start",
    )

    # 理论圆心
    if math.isfinite(theoretical_radius):
        center_x = initial_state.x
        center_y = initial_state.y + theoretical_radius

        ax.scatter(
            center_x,
            center_y,
            marker="x",
            s=100,
            label="theoretical center",
        )

        theoretical_circle = plt.Circle(
            (center_x, center_y),
            abs(theoretical_radius),
            fill=False,
            linestyle=":",
            label="theoretical circle",
        )

        ax.add_patch(theoretical_circle)

    # 动画元素
    trajectory_line, = ax.plot([], [], linewidth=2)

    vehicle_line, = ax.plot([], [], linewidth=2)

    heading_line, = ax.plot([], [], linewidth=2)

    ax.legend()

    margin = 2.0

    ax.set_xlim(
        min(x_list) - margin,
        max(x_list) + margin,
    )

    ax.set_ylim(
        min(y_list) - margin,
        max(y_list) + margin,
    )

    def update(frame_index):
        current_state = states[frame_index]

        # 已经行驶过的轨迹
        trajectory_line.set_data(
            x_list[:frame_index + 1],
            y_list[:frame_index + 1],
        )

        # 车辆矩形
        corners = get_vehicle_corners(
            state=current_state,
            front_length=front_length,
            rear_length=rear_length,
            vehicle_width=vehicle_width,
        )

        corner_x = [corner[0] for corner in corners]
        corner_y = [corner[1] for corner in corners]

        vehicle_line.set_data(corner_x, corner_y)

        # 航向箭头
        heading_length = 1.5

        heading_end_x = (
            current_state.x
            + heading_length * math.cos(current_state.yaw)
        )

        heading_end_y = (
            current_state.y
            + heading_length * math.sin(current_state.yaw)
        )

        heading_line.set_data(
            [current_state.x, heading_end_x],
            [current_state.y, heading_end_y],
        )

        return (
            trajectory_line,
            vehicle_line,
            heading_line,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=range(0, len(states), 3),
        interval=20,
        repeat=True,
        blit=True,
    )

    plt.show()


if __name__ == "__main__":
    main()