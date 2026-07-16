import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from matplotlib import font_manager

# Matplotlib 中文字体配置
font_candidates = [
    "PingFang SC",       # macOS
    "Heiti SC",          # macOS
    "Songti SC",         # macOS
    "Microsoft YaHei",   # Windows
    "SimHei",            # Windows
    "Noto Sans CJK SC",  # Linux
    "WenQuanYi Micro Hei",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name]
        print(f"Matplotlib 使用中文字体：{font_name}")
        break
else:
    print("警告：没有找到可用中文字体，中文可能显示为方框")

plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 数据结构
# ============================================================

@dataclass
class Pose:
    x: float
    y: float
    yaw: float

@dataclass
class DWAState:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    steering: float = 0.0

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.yaw)


@dataclass
class DWAControl:
    speed: float
    steering: float


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float


@dataclass
class VehicleGeometry:
    front_length: float
    rear_length: float
    width: float
    safety_margin: float = 0.0


@dataclass
class DWAConfig:
    wheel_base: float = 1.2

    max_speed: float = 1.2
    max_accel: float = 1.0
    max_decel: float = 1.5

    max_steer: float = math.radians(30.0)
    max_steer_rate: float = math.radians(70.0)

    speed_sample_count: int = 6
    steering_sample_count: int = 17

    dt: float = 0.1
    predict_time: float = 2.8
    lookahead_distance: float = 3.0

    goal_tolerance: float = 0.35

    goal_cost_weight: float = 3.0
    path_cost_weight: float = 2.0
    heading_cost_weight: float = 0.8
    obstacle_cost_weight: float = 3.5
    speed_cost_weight: float = 0.45
    steering_cost_weight: float = 0.2
    steering_change_cost_weight: float = 0.5
    progress_reward_weight: float = 0.04


@dataclass
class CandidateTrajectory:
    control: DWAControl
    trajectory: list[Pose]
    valid: bool
    clearance: float
    cost: float = math.inf


@dataclass
class DWAPlanResult:
    best: Optional[CandidateTrajectory]
    candidates: list[CandidateTrajectory]
    local_goal: Pose
    nearest_path_index: int


# ============================================================
# 基础函数
# ============================================================

def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(angle1: float, angle2: float) -> float:
    return normalize_angle(angle1 - angle2)


def generate_global_path() -> list[Pose]:
    """生成一条平滑 S 形全局路径。"""
    x = np.linspace(1.5, 18.5, 600)
    y = 6.0 + 1.35 * np.sin((x - 1.5) * 0.43)

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)

    return [
        Pose(float(px), float(py), float(pyaw))
        for px, py, pyaw in zip(x, y, yaw)
    ]


def rrt_result_to_global_path(rrt_result) -> list[Pose]:
    """
    后续接入 Ackermann-RRT 时使用。

    rrt_result 格式：
        path_x, path_y, path_yaw, directions
    """
    path_x, path_y, path_yaw, _ = rrt_result

    return [
        Pose(float(x), float(y), float(yaw))
        for x, y, yaw in zip(path_x, path_y, path_yaw)
    ]


def get_vehicle_corners(
    pose: Pose,
    vehicle: VehicleGeometry,
) -> list[tuple[float, float]]:
    margin = vehicle.safety_margin

    front = vehicle.front_length + margin
    rear = vehicle.rear_length + margin
    half_width = vehicle.width / 2.0 + margin

    local_corners = [
        (front, half_width),
        (front, -half_width),
        (-rear, -half_width),
        (-rear, half_width),
    ]

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    corners = []

    for local_x, local_y in local_corners:
        world_x = pose.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = pose.y + local_x * sin_yaw + local_y * cos_yaw
        corners.append((world_x, world_y))

    return corners


def circle_to_vehicle_clearance(
    pose: Pose,
    vehicle: VehicleGeometry,
    obstacle: CircleObstacle,
) -> float:
    """
    计算圆形障碍物与车辆矩形之间的距离。

    返回值：
        > 0：未碰撞
        = 0：刚好接触
        < 0：发生碰撞
    """
    dx = obstacle.x - pose.x
    dy = obstacle.y - pose.y

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    margin = vehicle.safety_margin
    x_min = -vehicle.rear_length - margin
    x_max = vehicle.front_length + margin
    y_min = -vehicle.width / 2.0 - margin
    y_max = vehicle.width / 2.0 + margin

    closest_x = min(max(local_x, x_min), x_max)
    closest_y = min(max(local_y, y_min), y_max)

    distance = math.hypot(local_x - closest_x, local_y - closest_y)
    return distance - obstacle.radius


def pose_inside_bounds(
    pose: Pose,
    vehicle: VehicleGeometry,
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds

    return all(
        x_min <= x <= x_max and y_min <= y <= y_max
        for x, y in get_vehicle_corners(pose, vehicle)
    )


# ============================================================
# Ackermann-DWA
# ============================================================

class AckermannDWA:
    def __init__(
        self,
        config: DWAConfig,
        vehicle: VehicleGeometry,
        global_path: list[Pose],
        bounds: tuple[float, float, float, float],
    ):
        self.config = config
        self.vehicle = vehicle
        self.global_path = global_path
        self.bounds = bounds
        self.progress_index = 0

        self.path_x = np.array([pose.x for pose in global_path])
        self.path_y = np.array([pose.y for pose in global_path])

    def reset(self):
        self.progress_index = 0

    def step(
        self,
        state: DWAState,
        control: DWAControl,
    ) -> DWAState:
        """执行一个控制周期。"""
        dt = self.config.dt
        speed = control.speed
        steering = control.steering

        yaw_rate = speed / self.config.wheel_base * math.tan(steering)

        if abs(yaw_rate) < 1e-10:
            x = state.x + speed * math.cos(state.yaw) * dt
            y = state.y + speed * math.sin(state.yaw) * dt
            yaw = state.yaw
        else:
            new_yaw = state.yaw + yaw_rate * dt
            radius = speed / yaw_rate

            x = state.x + radius * (
                math.sin(new_yaw) - math.sin(state.yaw)
            )

            y = state.y - radius * (
                math.cos(new_yaw) - math.cos(state.yaw)
            )

            yaw = normalize_angle(new_yaw)

        return DWAState(x, y, yaw, speed, steering)

    def predict_trajectory(
        self,
        state: DWAState,
        control: DWAControl,
    ) -> list[Pose]:
        simulated_state = DWAState(
            state.x,
            state.y,
            state.yaw,
            state.speed,
            state.steering,
        )

        trajectory = [simulated_state.pose]
        step_count = max(1, int(self.config.predict_time / self.config.dt))

        for _ in range(step_count):
            simulated_state = self.step(simulated_state, control)
            trajectory.append(simulated_state.pose)

        return trajectory

    def find_nearest_path_index(self, state: DWAState) -> int:
        start = max(0, self.progress_index - 15)
        end = min(len(self.global_path), self.progress_index + 260)

        dx = self.path_x[start:end] - state.x
        dy = self.path_y[start:end] - state.y

        nearest_index = start + int(np.argmin(dx * dx + dy * dy))
        self.progress_index = max(self.progress_index, nearest_index)

        return nearest_index

    def find_local_goal(
        self,
        nearest_index: int,
    ) -> tuple[Pose, int]:
        distance = 0.0
        index = nearest_index

        while index < len(self.global_path) - 1:
            current = self.global_path[index]
            next_pose = self.global_path[index + 1]

            distance += math.hypot(
                next_pose.x - current.x,
                next_pose.y - current.y,
            )

            index += 1

            if distance >= self.config.lookahead_distance:
                break

        return self.global_path[index], index

    def dynamic_window(
        self,
        state: DWAState,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = self.config.dt

        speed_min = max(
            0.0,
            state.speed - self.config.max_decel * dt,
        )

        speed_max = min(
            self.config.max_speed,
            state.speed + self.config.max_accel * dt,
        )

        steer_delta = self.config.max_steer_rate * dt

        steer_min = max(
            -self.config.max_steer,
            state.steering - steer_delta,
        )

        steer_max = min(
            self.config.max_steer,
            state.steering + steer_delta,
        )

        speeds = np.linspace(
            speed_min,
            speed_max,
            self.config.speed_sample_count,
        )

        steerings = np.linspace(
            steer_min,
            steer_max,
            self.config.steering_sample_count,
        )

        print(f"[DWA] 动态窗口 - 速度范围: [{speed_min:.2f}, {speed_max:.2f}] m/s, "
              f"转向范围: [{math.degrees(steer_min):.1f}, {math.degrees(steer_max):.1f}]°")

        return speeds, steerings

    def trajectory_clearance(
        self,
        trajectory: list[Pose],
        obstacles: list[CircleObstacle],
    ) -> tuple[bool, float]:
        minimum_clearance = math.inf

        for pose in trajectory:
            if not pose_inside_bounds(pose, self.vehicle, self.bounds):
                return False, -1.0

            for obstacle in obstacles:
                clearance = circle_to_vehicle_clearance(
                    pose,
                    self.vehicle,
                    obstacle,
                )

                if clearance <= 0.0:
                    return False, clearance

                minimum_clearance = min(minimum_clearance, clearance)

        return True, minimum_clearance

    def path_distance(
        self,
        pose: Pose,
        start_index: int,
        end_index: int,
    ) -> tuple[float, int]:
        end_index = max(start_index + 1, end_index)

        dx = self.path_x[start_index:end_index] - pose.x
        dy = self.path_y[start_index:end_index] - pose.y
        distances = dx * dx + dy * dy

        local_index = int(np.argmin(distances))
        global_index = start_index + local_index

        return math.sqrt(float(distances[local_index])), global_index

    def average_trajectory_path_distance(
        self,
        trajectory: list[Pose],
        nearest_index: int,
        search_end: int,
    ) -> float:
        sample_step = max(1, len(trajectory) // 8)
        sampled_poses = trajectory[::sample_step]

        distances = [
            self.path_distance(pose, nearest_index, search_end)[0]
            for pose in sampled_poses
        ]

        return sum(distances) / len(distances)

    def evaluate_trajectory(
        self,
        state: DWAState,
        candidate: CandidateTrajectory,
        local_goal: Pose,
        nearest_index: int,
        local_goal_index: int,
    ) -> float:
        end_pose = candidate.trajectory[-1]

        goal_cost = math.hypot(
            end_pose.x - local_goal.x,
            end_pose.y - local_goal.y,
        )

        search_end = min(
            len(self.global_path),
            local_goal_index + 100,
        )

        path_cost = self.average_trajectory_path_distance(
            candidate.trajectory,
            nearest_index,
            search_end,
        )

        end_path_distance, end_path_index = self.path_distance(
            end_pose,
            nearest_index,
            search_end,
        )

        reference_yaw = self.global_path[end_path_index].yaw
        heading_cost = abs(
            angle_difference(end_pose.yaw, reference_yaw)
        )

        if math.isinf(candidate.clearance):
            obstacle_cost = 0.0
        else:
            obstacle_cost = 1.0 / max(
                candidate.clearance + 0.05,
                0.05,
            )

        speed_cost = self.config.max_speed - candidate.control.speed

        steering_cost = abs(candidate.control.steering) / max(
            self.config.max_steer,
            1e-8,
        )

        steering_change_cost = abs(
            candidate.control.steering - state.steering
        ) / max(self.config.max_steer, 1e-8)

        progress = max(0, end_path_index - nearest_index)

        total_cost = (
            self.config.goal_cost_weight * goal_cost
            + self.config.path_cost_weight * (path_cost + end_path_distance)
            + self.config.heading_cost_weight * heading_cost
            + self.config.obstacle_cost_weight * obstacle_cost
            + self.config.speed_cost_weight * speed_cost
            + self.config.steering_cost_weight * steering_cost
            + self.config.steering_change_cost_weight * steering_change_cost
            - self.config.progress_reward_weight * progress
        )

        return total_cost

    def plan(
        self,
        state: DWAState,
        obstacles: list[CircleObstacle],
    ) -> DWAPlanResult:
        """
        统一 DWA 接口。

        输入：
            当前车辆状态
            当前障碍物

        输出：
            所有采样轨迹
            最优控制轨迹
            局部目标
        """
        print(f"[DWA] 规划中 - 位置: ({state.x:.2f}, {state.y:.2f}), 速度: {state.speed:.2f} m/s, 航向: {math.degrees(state.yaw):.1f}°")

        nearest_index = self.find_nearest_path_index(state)

        local_goal, local_goal_index = self.find_local_goal(
            nearest_index
        )

        speed_samples, steering_samples = self.dynamic_window(state)

        candidates = []
        best_candidate = None
        best_cost = math.inf

        for speed in speed_samples:
            for steering in steering_samples:
                control = DWAControl(
                    float(speed),
                    float(steering),
                )

                trajectory = self.predict_trajectory(
                    state,
                    control,
                )

                valid, clearance = self.trajectory_clearance(
                    trajectory,
                    obstacles,
                )

                candidate = CandidateTrajectory(
                    control=control,
                    trajectory=trajectory,
                    valid=valid,
                    clearance=clearance,
                )

                if valid:
                    candidate.cost = self.evaluate_trajectory(
                        state,
                        candidate,
                        local_goal,
                        nearest_index,
                        local_goal_index,
                    )

                    if candidate.cost < best_cost:
                        best_cost = candidate.cost
                        best_candidate = candidate

                candidates.append(candidate)

        if best_candidate:
            print(f"[DWA] 最优轨迹 - 速度: {best_candidate.control.speed:.2f} m/s, "
                  f"转向: {math.degrees(best_candidate.control.steering):.1f}°, "
                  f"代价: {best_candidate.cost:.2f}")

            # 🆕 显示有效轨迹的速度分布
            valid_speeds = [c.control.speed for c in candidates if c.valid]
            valid_costs = [c.cost for c in candidates if c.valid]
            if valid_speeds:
                print(f"[DWA] 有效轨迹: {len(valid_speeds)} 条, "
                      f"速度范围: [{min(valid_speeds):.2f}, {max(valid_speeds):.2f}] m/s, "
                      f"代价范围: [{min(valid_costs):.2f}, {max(valid_costs):.2f}]")
        else:
            print(f"[DWA] ⚠️  未找到有效轨迹")

        return DWAPlanResult(
            best=best_candidate,
            candidates=candidates,
            local_goal=local_goal,
            nearest_path_index=nearest_index,
        )

    def goal_reached(self, state: DWAState) -> bool:
        goal = self.global_path[-1]

        distance = math.hypot(
            state.x - goal.x,
            state.y - goal.y,
        )

        return (
            distance <= self.config.goal_tolerance
            and self.progress_index >= len(self.global_path) - 20
        )


# ============================================================
# Matplotlib 交互测试
# ============================================================

class InteractiveDWATest:
    def __init__(self):
        self.bounds = (0.0, 20.0, 0.0, 12.0)

        self.vehicle = VehicleGeometry(
            front_length=1.4,
            rear_length=0.6,
            width=1.0,
            safety_margin=0.12,
        )

        self.config = DWAConfig()

        self.global_path = generate_global_path()

        self.dwa = AckermannDWA(
            config=self.config,
            vehicle=self.vehicle,
            global_path=self.global_path,
            bounds=self.bounds,
        )

        start_pose = self.global_path[0]

        self.state = DWAState(
            x=start_pose.x,
            y=start_pose.y,
            yaw=start_pose.yaw,
            speed=0.0,
            steering=0.0,
        )

        first_index = min(
            range(len(self.global_path)),
            key=lambda i: abs(self.global_path[i].x - 9.2),
        )

        second_index = min(
            range(len(self.global_path)),
            key=lambda i: abs(self.global_path[i].x - 13.5),
        )

        self.initial_obstacles = [
            CircleObstacle(
                self.global_path[first_index].x,
                self.global_path[first_index].y + 0.15,
                0.48,
            ),
            CircleObstacle(
                self.global_path[second_index].x,
                self.global_path[second_index].y - 1.35,
                0.50,
            ),
        ]

        self.obstacles = [
            CircleObstacle(obs.x, obs.y, obs.radius)
            for obs in self.initial_obstacles
        ]

        self.executed_x = [self.state.x]
        self.executed_y = [self.state.y]

        self.paused = False
        self.finished = False
        self.dragged_obstacle_index = None

        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        self.valid_collection = LineCollection(
            [],
            linewidths=0.7,
            alpha=0.18,
            colors="gray",
            label="可行采样轨迹",
        )

        self.invalid_collection = LineCollection(
            [],
            linewidths=0.7,
            alpha=0.12,
            colors="red",
            label="碰撞采样轨迹",
        )

        self.ax.add_collection(self.valid_collection)
        self.ax.add_collection(self.invalid_collection)

        self.best_line, = self.ax.plot(
            [],
            [],
            linewidth=3,
            color="limegreen",
            label="DWA最优轨迹",
        )

        self.executed_line, = self.ax.plot(
            self.executed_x,
            self.executed_y,
            linewidth=2.5,
            color="royalblue",
            label="车辆实际轨迹",
        )

        self.local_goal_point, = self.ax.plot(
            [],
            [],
            marker="x",
            markersize=10,
            linestyle="None",
            color="black",
            label="局部目标",
        )

        self.vehicle_patch = Polygon(
            get_vehicle_corners(self.state.pose, self.vehicle),
            closed=True,
            fill=False,
            linewidth=2,
            edgecolor="black",
            label="车辆",
        )

        self.ax.add_patch(self.vehicle_patch)

        self.obstacle_patches = []

        self.draw_static_scene()
        self.connect_events()

        self.timer = self.fig.canvas.new_timer(
            interval=int(self.config.dt * 1000)
        )

        self.timer.add_callback(self.update)
        self.timer.start()

    def draw_static_scene(self):
        path_x = [pose.x for pose in self.global_path]
        path_y = [pose.y for pose in self.global_path]

        self.ax.plot(
            path_x,
            path_y,
            linestyle="--",
            linewidth=2,
            color="darkorange",
            label="全局路径",
        )

        self.ax.scatter(
            path_x[0],
            path_y[0],
            s=70,
            color="green",
            label="起点",
        )

        self.ax.scatter(
            path_x[-1],
            path_y[-1],
            s=70,
            color="purple",
            label="终点",
        )

        self.refresh_obstacle_patches()

        x_min, x_max, y_min, y_max = self.bounds
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X / m")
        self.ax.set_ylabel("Y / m")

        self.ax.set_title(
            "左键拖动障碍物；右键添加障碍物；空格暂停；R重置车辆；C恢复障碍物"
        )

        self.ax.legend(loc="upper left")

    def refresh_obstacle_patches(self):
        for patch in self.obstacle_patches:
            patch.remove()

        self.obstacle_patches = []

        for index, obstacle in enumerate(self.obstacles):
            patch = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                facecolor="tomato",
                edgecolor="darkred",
                alpha=0.65,
                label="可拖动障碍物" if index == 0 else None,
            )

            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)

    def connect_events(self):
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_press,
        )

        self.fig.canvas.mpl_connect(
            "motion_notify_event",
            self.on_mouse_motion,
        )

        self.fig.canvas.mpl_connect(
            "button_release_event",
            self.on_mouse_release,
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )

    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            self.obstacles.append(
                CircleObstacle(
                    event.xdata,
                    event.ydata,
                    0.45,
                )
            )

            self.refresh_obstacle_patches()
            self.fig.canvas.draw_idle()
            return

        if event.button != 1:
            return

        for index in reversed(range(len(self.obstacles))):
            obstacle = self.obstacles[index]

            distance = math.hypot(
                event.xdata - obstacle.x,
                event.ydata - obstacle.y,
            )

            if distance <= obstacle.radius + 0.25:
                self.dragged_obstacle_index = index
                return

    def on_mouse_motion(self, event):
        if (
            self.dragged_obstacle_index is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        x_min, x_max, y_min, y_max = self.bounds
        obstacle = self.obstacles[self.dragged_obstacle_index]

        obstacle.x = min(
            max(event.xdata, x_min + obstacle.radius),
            x_max - obstacle.radius,
        )

        obstacle.y = min(
            max(event.ydata, y_min + obstacle.radius),
            y_max - obstacle.radius,
        )

        self.obstacle_patches[
            self.dragged_obstacle_index
        ].center = (obstacle.x, obstacle.y)

        self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        self.dragged_obstacle_index = None

    def on_key_press(self, event):
        if event.key == " ":
            self.paused = not self.paused

            state_text = "暂停" if self.paused else "运行"
            self.ax.set_title(
                f"DWA状态：{state_text}；左键可拖动障碍物"
            )

            self.fig.canvas.draw_idle()

        elif event.key in ("r", "R"):
            self.reset_vehicle()

        elif event.key in ("c", "C"):
            self.restore_obstacles()

        elif event.key == "escape":
            self.timer.stop()
            plt.close(self.fig)

    def reset_vehicle(self):
        start_pose = self.global_path[0]

        self.state = DWAState(
            start_pose.x,
            start_pose.y,
            start_pose.yaw,
            0.0,
            0.0,
        )

        self.dwa.reset()

        self.executed_x = [self.state.x]
        self.executed_y = [self.state.y]

        self.executed_line.set_data(
            self.executed_x,
            self.executed_y,
        )

        self.vehicle_patch.set_xy(
            get_vehicle_corners(
                self.state.pose,
                self.vehicle,
            )
        )

        self.best_line.set_data([], [])
        self.valid_collection.set_segments([])
        self.invalid_collection.set_segments([])
        self.local_goal_point.set_data([], [])

        self.finished = False
        self.paused = False

        self.ax.set_title("车辆已重置，DWA重新开始")
        self.fig.canvas.draw_idle()

    def restore_obstacles(self):
        self.obstacles = [
            CircleObstacle(obs.x, obs.y, obs.radius)
            for obs in self.initial_obstacles
        ]

        self.refresh_obstacle_patches()
        self.fig.canvas.draw_idle()

    @staticmethod
    def trajectory_to_segment(
        trajectory: list[Pose],
    ) -> list[tuple[float, float]]:
        return [
            (pose.x, pose.y)
            for pose in trajectory
        ]

    def update_sampled_trajectories(
        self,
        plan_result: DWAPlanResult,
    ):
        valid_segments = []
        invalid_segments = []

        for candidate in plan_result.candidates:
            segment = self.trajectory_to_segment(
                candidate.trajectory
            )

            if candidate.valid:
                valid_segments.append(segment)
            else:
                invalid_segments.append(segment)

        self.valid_collection.set_segments(valid_segments)
        self.invalid_collection.set_segments(invalid_segments)

        if plan_result.best is None:
            self.best_line.set_data([], [])
        else:
            self.best_line.set_data(
                [
                    pose.x
                    for pose in plan_result.best.trajectory
                ],
                [
                    pose.y
                    for pose in plan_result.best.trajectory
                ],
            )

        self.local_goal_point.set_data(
            [plan_result.local_goal.x],
            [plan_result.local_goal.y],
        )

    def update(self):
        if self.paused or self.finished:
            return

        plan_result = self.dwa.plan(
            state=self.state,
            obstacles=self.obstacles,
        )

        self.update_sampled_trajectories(plan_result)

        if plan_result.best is None:
            self.state.speed = 0.0
            self.state.steering = 0.0

            self.ax.set_title(
                "没有安全局部轨迹：车辆停车；请拖动障碍物恢复通道"
            )

            self.fig.canvas.draw_idle()
            return

        self.state = self.dwa.step(
            self.state,
            plan_result.best.control,
        )

        self.executed_x.append(self.state.x)
        self.executed_y.append(self.state.y)

        self.executed_line.set_data(
            self.executed_x,
            self.executed_y,
        )

        self.vehicle_patch.set_xy(
            get_vehicle_corners(
                self.state.pose,
                self.vehicle,
            )
        )

        valid_count = sum(
            candidate.valid
            for candidate in plan_result.candidates
        )

        clearance = plan_result.best.clearance

        self.ax.set_title(
            f"DWA实时局部规划 | "
            f"v={self.state.speed:.2f} m/s | "
            f"steer={math.degrees(self.state.steering):.1f}° | "
            f"可行轨迹={valid_count}/{len(plan_result.candidates)} | "
            f"最小间距={clearance:.2f} m"
        )

        if self.dwa.goal_reached(self.state):
            self.finished = True
            self.state.speed = 0.0

            self.ax.set_title(
                "车辆到达全局目标；按 R 可重新测试"
            )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Ackermann-DWA 交互式局部避障测试")
    print("=" * 70)
    print("灰色轨迹：无碰撞候选轨迹")
    print("红色轨迹：发生碰撞的候选轨迹")
    print("绿色粗线：DWA选择的最优轨迹")
    print("蓝色粗线：车辆已经执行的真实轨迹")
    print("左键拖动：移动障碍物")
    print("右键单击：添加障碍物")
    print("空格：暂停或继续")
    print("R：重置车辆")
    print("C：恢复初始障碍物")
    print("Esc：退出")

    app = InteractiveDWATest()
    app.show()


if __name__ == "__main__":
    main()