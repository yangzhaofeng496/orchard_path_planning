import math
import random
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon

from reeds_shepp_path_test import Pose, plan_reeds_shepp_path
from vehicle_collision_test import (
    VehicleGeometry,
    CircleObstacle,
    check_path_collision,
    check_pose_collision,
    get_vehicle_corners,
)


# ============================================================
# Matplotlib 中文字体
# ============================================================

font_candidates = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name]
        break

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EdgePath:
    x: list[float]
    y: list[float]
    yaw: list[float]
    directions: list[int]

    @property
    def length(self) -> float:
        return sum(
            math.hypot(self.x[i] - self.x[i - 1], self.y[i] - self.y[i - 1])
            for i in range(1, len(self.x))
        )


@dataclass
class Node:
    pose: Pose
    parent: Optional[int] = None
    cost: float = 0.0
    path_x: list[float] = field(default_factory=list)
    path_y: list[float] = field(default_factory=list)
    path_yaw: list[float] = field(default_factory=list)
    path_directions: list[int] = field(default_factory=list)


# ============================================================
# 基础函数
# ============================================================

def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(angle1: float, angle2: float) -> float:
    return normalize_angle(angle1 - angle2)


def interpolate_angle(angle1: float, angle2: float, ratio: float) -> float:
    return normalize_angle(angle1 + ratio * angle_difference(angle2, angle1))


def pose_inside_bounds(
    pose: Pose,
    vehicle: VehicleGeometry,
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    corners = get_vehicle_corners(pose, vehicle)

    return all(
        x_min <= x <= x_max and y_min <= y <= y_max
        for x, y in corners
    )


def truncate_path(path, maximum_length: float) -> EdgePath:
    """沿 Reeds-Shepp 曲线截取前 maximum_length 米。"""
    path_x = [float(path.x[0])]
    path_y = [float(path.y[0])]
    path_yaw = [float(path.yaw[0])]
    directions = [int(path.directions[0])]
    traveled = 0.0

    for i in range(1, len(path.x)):
        x0, y0 = float(path.x[i - 1]), float(path.y[i - 1])
        x1, y1 = float(path.x[i]), float(path.y[i])
        segment_length = math.hypot(x1 - x0, y1 - y0)

        if segment_length < 1e-10:
            continue

        remaining = maximum_length - traveled
        if remaining <= 1e-10:
            break

        if segment_length <= remaining + 1e-10:
            path_x.append(x1)
            path_y.append(y1)
            path_yaw.append(float(path.yaw[i]))
            directions.append(int(path.directions[i]))
            traveled += segment_length
            continue

        ratio = remaining / segment_length
        path_x.append(x0 + ratio * (x1 - x0))
        path_y.append(y0 + ratio * (y1 - y0))
        path_yaw.append(
            interpolate_angle(float(path.yaw[i - 1]), float(path.yaw[i]), ratio)
        )
        directions.append(int(path.directions[i]))
        break

    return EdgePath(path_x, path_y, path_yaw, directions)


def split_path_by_direction(
    path_x: list[float],
    path_y: list[float],
    directions: list[int],
):
    if not path_x:
        return []

    segments = []
    start_index = 0

    for i in range(1, len(path_x)):
        if directions[i] == directions[i - 1]:
            continue

        segments.append(
            (
                directions[i - 1],
                path_x[start_index:i + 1],
                path_y[start_index:i + 1],
            )
        )
        start_index = i - 1

    segments.append(
        (
            directions[-1],
            path_x[start_index:],
            path_y[start_index:],
        )
    )

    return segments


def calculate_path_statistics(result):
    path_x, path_y, _, directions = result

    if len(path_x) < 2:
        return 0.0, 0.0, 0.0, 0

    total_length = 0.0
    forward_length = 0.0
    reverse_length = 0.0
    switch_count = 0
    previous_direction = directions[0]

    for i in range(1, len(path_x)):
        distance = math.hypot(
            path_x[i] - path_x[i - 1],
            path_y[i] - path_y[i - 1],
        )

        total_length += distance

        if directions[i] > 0:
            forward_length += distance
        else:
            reverse_length += distance

        if directions[i] != previous_direction:
            switch_count += 1
            previous_direction = directions[i]

    return total_length, forward_length, reverse_length, switch_count


# ============================================================
# Ackermann-RRT
# ============================================================

class AckermannRRT:
    def __init__(
        self,
        start: Pose,
        goal: Pose,
        bounds: tuple[float, float, float, float],
        vehicle: VehicleGeometry,
        obstacles: list[CircleObstacle],
        curvature: float,
        step_size: float = 0.02,
        expand_length: float = 3.0,
        goal_connect_distance: float = 7.0,
        goal_sample_rate: float = 0.15,
        max_iterations: int = 5000,
        yaw_weight: float = 2.0,
        min_node_spacing: float = 0.15,
        random_seed: int = 7,
        animate: bool = True,
        fig=None,
        ax=None,
    ):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

        self.vehicle = vehicle
        self.obstacles = obstacles
        self.curvature = curvature
        self.step_size = step_size
        self.expand_length = expand_length
        self.goal_connect_distance = goal_connect_distance
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.yaw_weight = yaw_weight
        self.min_node_spacing = min_node_spacing
        self.animate = animate

        self.nodes = [Node(start)]
        self.accepted_node_count = 0
        self.fig = fig
        self.ax = ax

        random.seed(random_seed)

    def sample_pose(self) -> Pose:
        if random.random() < self.goal_sample_rate:
            return Pose(self.goal.x, self.goal.y, self.goal.yaw)

        return Pose(
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            random.uniform(-math.pi, math.pi),
        )

    def state_distance(self, pose1: Pose, pose2: Pose) -> float:
        position_distance = math.hypot(
            pose1.x - pose2.x,
            pose1.y - pose2.y,
        )
        yaw_distance = abs(angle_difference(pose1.yaw, pose2.yaw))
        return position_distance + self.yaw_weight * yaw_distance

    def nearest_node_index(self, pose: Pose) -> int:
        distances = [
            self.state_distance(node.pose, pose)
            for node in self.nodes
        ]
        return min(range(len(distances)), key=distances.__getitem__)

    def steer(
        self,
        from_pose: Pose,
        to_pose: Pose,
        maximum_length: Optional[float] = None,
    ) -> Optional[EdgePath]:
        position_distance = math.hypot(
            to_pose.x - from_pose.x,
            to_pose.y - from_pose.y,
        )
        yaw_distance = abs(angle_difference(to_pose.yaw, from_pose.yaw))

        if position_distance < 1e-4 and yaw_distance < 1e-4:
            return None

        connection_scale = position_distance + yaw_distance / self.curvature

        step_candidates = [
            min(self.step_size, max(0.002, connection_scale / 250.0)),
            0.01,
            0.005,
            0.002,
        ]

        path = None

        for current_step in step_candidates:
            path = plan_reeds_shepp_path(
                from_pose,
                to_pose,
                curvature=self.curvature,
                step_size=current_step,
            )

            if path is not None and len(path.x) >= 2:
                break

        if path is None or len(path.x) < 2:
            return None

        if maximum_length is not None:
            return truncate_path(path, maximum_length)

        return EdgePath(
            x=[float(value) for value in path.x],
            y=[float(value) for value in path.y],
            yaw=[float(value) for value in path.yaw],
            directions=[int(value) for value in path.directions],
        )

    def pose_inside_bounds(self, pose: Pose) -> bool:
        return pose_inside_bounds(pose, self.vehicle, self.bounds)

    def pose_is_collision_free(self, pose: Pose) -> bool:
        return not check_pose_collision(
            pose,
            self.vehicle,
            self.obstacles,
        )

    def path_inside_bounds(self, path: EdgePath) -> bool:
        for x, y, yaw in zip(path.x, path.y, path.yaw):
            if not self.pose_inside_bounds(Pose(x, y, yaw)):
                return False
        return True

    def path_is_valid(
        self,
        path: EdgePath,
        check_spacing: bool = True,
    ) -> bool:
        if check_spacing and path.length < self.min_node_spacing:
            return False

        if not self.path_inside_bounds(path):
            return False

        has_collision, _ = check_path_collision(
            path,
            self.vehicle,
            self.obstacles,
        )
        return not has_collision

    def node_is_new(self, pose: Pose) -> bool:
        return all(
            self.state_distance(node.pose, pose) >= self.min_node_spacing
            for node in self.nodes
        )

    def add_node(self, parent_index: int, path: EdgePath) -> int:
        parent = self.nodes[parent_index]
        pose = Pose(path.x[-1], path.y[-1], path.yaw[-1])

        node = Node(
            pose=pose,
            parent=parent_index,
            cost=parent.cost + path.length,
            path_x=path.x,
            path_y=path.y,
            path_yaw=path.yaw,
            path_directions=path.directions,
        )

        self.nodes.append(node)
        return len(self.nodes) - 1

    def try_connect_goal(self, node_index: int) -> Optional[int]:
        node = self.nodes[node_index]

        distance_to_goal = math.hypot(
            node.pose.x - self.goal.x,
            node.pose.y - self.goal.y,
        )

        if distance_to_goal > self.goal_connect_distance:
            return None

        path = self.steer(node.pose, self.goal)

        if path is None or not self.path_is_valid(path, check_spacing=False):
            return None

        goal_index = self.add_node(node_index, path)

        if self.animate:
            self.draw_edge(path, linewidth=1.2)

        return goal_index

    def validate_start_and_goal(self) -> bool:
        if not self.pose_inside_bounds(self.start):
            print("错误：起点车辆轮廓超出地图边界")
            return False

        if not self.pose_inside_bounds(self.goal):
            print("错误：目标车辆轮廓超出地图边界")
            return False

        if not self.pose_is_collision_free(self.start):
            print("错误：起点车辆与障碍物碰撞")
            return False

        if not self.pose_is_collision_free(self.goal):
            print("错误：目标车辆与障碍物碰撞")
            return False

        return True

    def planning(self):
        if not self.validate_start_and_goal():
            return None

        if self.animate:
            self.initialize_plot()

        for iteration in range(self.max_iterations):
            random_pose = self.sample_pose()
            nearest_index = self.nearest_node_index(random_pose)
            nearest_pose = self.nodes[nearest_index].pose

            path = self.steer(
                nearest_pose,
                random_pose,
                maximum_length=self.expand_length,
            )

            if path is None or not self.path_is_valid(path):
                continue

            new_pose = Pose(path.x[-1], path.y[-1], path.yaw[-1])

            if not self.node_is_new(new_pose):
                continue

            new_index = self.add_node(nearest_index, path)
            self.accepted_node_count += 1

            if self.animate:
                self.draw_edge(path)

            goal_index = self.try_connect_goal(new_index)

            if goal_index is not None:
                result = self.extract_path(goal_index)

                print(f"第 {iteration + 1} 次迭代找到路径")
                print(f"树节点数量：{len(self.nodes)}")
                print(f"搜索树路径代价：{self.nodes[goal_index].cost:.3f} m")

                if self.animate:
                    self.draw_final_path(result)

                return result

            if (iteration + 1) % 500 == 0:
                print(f"迭代：{iteration + 1}，树节点：{len(self.nodes)}")

        print("达到最大迭代次数，未找到路径")
        print(f"树节点数量：{len(self.nodes)}")
        return None

    def extract_path(self, goal_index: int):
        segments = []
        current_index = goal_index

        while self.nodes[current_index].parent is not None:
            node = self.nodes[current_index]

            segments.append(
                (
                    node.path_x,
                    node.path_y,
                    node.path_yaw,
                    node.path_directions,
                )
            )

            current_index = node.parent

        segments.reverse()

        path_x = []
        path_y = []
        path_yaw = []
        directions = []

        for segment_x, segment_y, segment_yaw, segment_directions in segments:
            start_index = 0 if not path_x else 1
            path_x.extend(segment_x[start_index:])
            path_y.extend(segment_y[start_index:])
            path_yaw.extend(segment_yaw[start_index:])
            directions.extend(segment_directions[start_index:])

        return path_x, path_y, path_yaw, directions

    def initialize_plot(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(11, 9))
        else:
            self.ax.clear()

        for i, obstacle in enumerate(self.obstacles):
            label = "障碍物" if i == 0 else None

            self.ax.add_patch(
                Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    fill=False,
                    linewidth=2,
                    label=label,
                )
            )

        self.draw_pose(self.start, "起点")
        self.draw_pose(self.goal, "目标")

        self.ax.add_patch(
            Polygon(
                get_vehicle_corners(self.start, self.vehicle),
                closed=True,
                fill=False,
                linewidth=1.5,
            )
        )

        self.ax.add_patch(
            Polygon(
                get_vehicle_corners(self.goal, self.vehicle),
                closed=True,
                fill=False,
                linewidth=1.5,
            )
        )

        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X / m")
        self.ax.set_ylabel("Y / m")
        self.ax.set_title("Ackermann-RRT 正在规划")
        self.ax.legend()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def draw_pose(self, pose: Pose, label: str):
        self.ax.scatter(pose.x, pose.y, s=70, label=label)

        self.ax.arrow(
            pose.x,
            pose.y,
            0.8 * math.cos(pose.yaw),
            0.8 * math.sin(pose.yaw),
            width=0.025,
            length_includes_head=True,
        )

    def draw_edge(self, path: EdgePath, linewidth: float = 0.6):
        self.ax.plot(
            path.x,
            path.y,
            linewidth=linewidth,
            alpha=0.45,
        )

        if self.accepted_node_count % 5 == 0:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def draw_final_path(self, result):
        path_x, path_y, path_yaw, directions = result
        segments = split_path_by_direction(path_x, path_y, directions)

        forward_label_added = False
        reverse_label_added = False

        for direction, segment_x, segment_y in segments:
            if direction > 0:
                label = None if forward_label_added else "最终前进路径"
                self.ax.plot(
                    segment_x,
                    segment_y,
                    "-",
                    linewidth=3,
                    label=label,
                )
                forward_label_added = True

            else:
                label = None if reverse_label_added else "最终倒车路径"
                self.ax.plot(
                    segment_x,
                    segment_y,
                    "--",
                    linewidth=3,
                    label=label,
                )
                reverse_label_added = True

        interval = max(1, len(path_x) // 15)

        for i in range(0, len(path_x), interval):
            pose = Pose(path_x[i], path_y[i], path_yaw[i])

            self.ax.add_patch(
                Polygon(
                    get_vehicle_corners(pose, self.vehicle),
                    closed=True,
                    fill=False,
                    linewidth=1,
                    alpha=0.6,
                )
            )

        self.ax.set_title("规划完成：再次左键拖动可重新规划")
        self.ax.legend()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)


# ============================================================
# 交互式重复规划
# ============================================================

class InteractiveAckermannRRTApp:
    def __init__(
        self,
        start: Pose,
        bounds: tuple[float, float, float, float],
        vehicle: VehicleGeometry,
        obstacles: list[CircleObstacle],
        curvature: float,
    ):
        self.start = start
        self.bounds = bounds
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.curvature = curvature

        self.is_planning = False
        self.press_position = None
        self.preview_point = None
        self.preview_heading = None
        self.plan_count = 0

        self.fig, self.ax = plt.subplots(figsize=(11, 9))

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw_initial_scene()

    def draw_initial_scene(self):
        self.ax.clear()

        for i, obstacle in enumerate(self.obstacles):
            label = "障碍物" if i == 0 else None

            self.ax.add_patch(
                Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    fill=False,
                    linewidth=2,
                    label=label,
                )
            )

        self.ax.scatter(
            self.start.x,
            self.start.y,
            s=80,
            label="起点",
        )

        self.ax.arrow(
            self.start.x,
            self.start.y,
            0.9 * math.cos(self.start.yaw),
            0.9 * math.sin(self.start.yaw),
            width=0.025,
            length_includes_head=True,
        )

        self.ax.add_patch(
            Polygon(
                get_vehicle_corners(self.start, self.vehicle),
                closed=True,
                fill=False,
                linewidth=1.5,
                label="车辆轮廓",
            )
        )

        x_min, x_max, y_min, y_max = self.bounds
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X / m")
        self.ax.set_ylabel("Y / m")
        self.ax.set_title("左键按下设置目标，拖动设置航向，松开后规划")
        self.ax.legend()

        self.fig.canvas.draw_idle()

    def remove_preview(self):
        for artist in (self.preview_point, self.preview_heading):
            if artist is None:
                continue

            try:
                artist.remove()
            except ValueError:
                pass

        self.preview_point = None
        self.preview_heading = None

    def validate_goal(self, goal: Pose) -> tuple[bool, str]:
        if not pose_inside_bounds(goal, self.vehicle, self.bounds):
            return False, "目标车辆轮廓超出地图边界"

        if check_pose_collision(goal, self.vehicle, self.obstacles):
            return False, "目标车辆与障碍物碰撞"

        return True, ""

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        if self.is_planning:
            self.ax.set_title("当前正在规划，请等待本次搜索结束")
            self.fig.canvas.draw_idle()
            return

        self.press_position = (event.xdata, event.ydata)
        self.remove_preview()

        self.preview_point, = self.ax.plot(
            [event.xdata],
            [event.ydata],
            "o",
            markersize=8,
        )

        self.preview_heading, = self.ax.plot([], [], linewidth=2)

        self.ax.set_title("按住左键拖动，设置目标车辆航向")
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if self.press_position is None:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x0, y0 = self.press_position
        self.preview_heading.set_data(
            [x0, event.xdata],
            [y0, event.ydata],
        )
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press_position is None or event.button != 1:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.press_position = None
            self.remove_preview()
            return

        x0, y0 = self.press_position
        dx = event.xdata - x0
        dy = event.ydata - y0

        yaw = 0.0 if math.hypot(dx, dy) < 0.2 else math.atan2(dy, dx)
        goal = Pose(x0, y0, yaw)

        self.press_position = None
        self.remove_preview()

        valid, message = self.validate_goal(goal)

        if not valid:
            print(message)
            self.ax.set_title(f"{message}，请重新选择目标")
            self.fig.canvas.draw_idle()
            return

        self.run_planning(goal)

    def run_planning(self, goal: Pose):
        self.is_planning = True
        self.plan_count += 1

        print("=" * 70)
        print(f"第 {self.plan_count} 次规划")
        print("=" * 70)
        print(f"目标 X：{goal.x:.3f} m")
        print(f"目标 Y：{goal.y:.3f} m")
        print(f"目标航向：{math.degrees(goal.yaw):.3f} deg")

        planner = AckermannRRT(
            start=self.start,
            goal=goal,
            bounds=self.bounds,
            vehicle=self.vehicle,
            obstacles=self.obstacles,
            curvature=self.curvature,
            step_size=0.02,
            expand_length=3.0,
            goal_connect_distance=7.0,
            goal_sample_rate=0.15,
            max_iterations=5000,
            yaw_weight=2.0,
            min_node_spacing=0.15,
            random_seed=7 + self.plan_count,
            animate=True,
            fig=self.fig,
            ax=self.ax,
        )

        try:
            result = planner.planning()
        finally:
            self.is_planning = False

        if result is None:
            self.ax.set_title("未找到路径：再次左键拖动可重新规划")
            self.fig.canvas.draw_idle()
            return

        total, forward, reverse, switches = calculate_path_statistics(result)

        print("=" * 70)
        print("最终路径统计")
        print("=" * 70)
        print(f"总路径长度：{total:.3f} m")
        print(f"前进距离：{forward:.3f} m")
        print(f"倒车距离：{reverse:.3f} m")
        print(f"换向次数：{switches}")

        self.ax.set_title(
            f"规划完成：长度 {total:.2f} m；再次左键拖动可重新规划"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_key(self, event):
        if event.key == "escape":
            plt.close(self.fig)

        elif event.key == "r" and not self.is_planning:
            self.draw_initial_scene()

    def show(self):
        plt.show()


# ============================================================
# 主程序
# ============================================================

def main():
    wheel_base = 2.5
    max_steering_angle = math.radians(30.0)

    minimum_turning_radius = wheel_base / math.tan(max_steering_angle)
    maximum_curvature = 1.0 / minimum_turning_radius

    bounds = (0.0, 25.0, 0.0, 23.0)
    start = Pose(2.0, 2.0, math.radians(0.0))

    vehicle = VehicleGeometry(
        front_length=3.0,
        rear_length=1.0,
        width=1.6,
        safety_margin=0.15,
    )

    obstacles = [
        CircleObstacle(7.0, 7.0, 0.9),
        CircleObstacle(8.0, 13.0, 1.0),
        CircleObstacle(13.0, 10.0, 1.1),
        CircleObstacle(17.0, 6.0, 0.9),
        CircleObstacle(18.0, 14.0, 1.0),
    ]

    print("=" * 70)
    print("交互式 Ackermann-RRT")
    print("=" * 70)
    print(f"最小转弯半径：{minimum_turning_radius:.3f} m")
    print(f"最大曲率：{maximum_curvature:.6f} 1/m")
    print("左键按下：设置目标位置")
    print("拖动鼠标：设置目标航向")
    print("松开左键：开始规划")
    print("再次左键拖动：清除旧结果并重新规划")
    print("R：清空当前结果")
    print("Esc：退出程序")

    app = InteractiveAckermannRRTApp(
        start=start,
        bounds=bounds,
        vehicle=vehicle,
        obstacles=obstacles,
        curvature=maximum_curvature,
    )

    app.show()


if __name__ == "__main__":
    main()