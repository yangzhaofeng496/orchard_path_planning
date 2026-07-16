import math
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt


@dataclass
class Node:
    """RRT树节点。"""
    x: float
    y: float
    parent: Optional[int] = None


class InteractiveRRT:
    def __init__(
        self,
        map_width: float = 100.0,
        map_height: float = 100.0,
        start: tuple[float, float] = (10.0, 10.0),
        step_size: float = 4.0,
        goal_sample_rate: float = 0.10,
        goal_tolerance: float = 5.0,
        max_iterations: int = 3000,
        collision_resolution: float = 0.5,
        animation_interval: int = 5,
    ):
        """
        参数说明
        ----------
        map_width, map_height:
            地图宽度和高度。

        start:
            起点坐标。

        step_size:
            每次树扩展的最大距离。

        goal_sample_rate:
            直接采样目标点的概率，用于提高搜索速度。

        goal_tolerance:
            新节点距离目标点小于该值时，尝试直接连接目标点。

        max_iterations:
            最大搜索次数。

        collision_resolution:
            线段碰撞检测的采样间隔。

        animation_interval:
            每扩展多少次刷新一次图像。
        """
        self.map_width = map_width
        self.map_height = map_height
        self.start = start

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations
        self.collision_resolution = collision_resolution
        self.animation_interval = animation_interval

        self.goal: Optional[tuple[float, float]] = None
        self.nodes: list[Node] = []
        self.is_planning = False

        # 矩形障碍物：(左下角x, 左下角y, 宽度, 高度)
        self.rect_obstacles = [
            (20.0, 20.0, 18.0, 12.0),
            (48.0, 10.0, 12.0, 35.0),
            (15.0, 55.0, 25.0, 12.0),
            (62.0, 60.0, 25.0, 10.0),
            (45.0, 78.0, 12.0, 16.0),
        ]

        # 圆形障碍物：(圆心x, 圆心y, 半径)
        self.circle_obstacles = [
            (75.0, 30.0, 8.0),
            (45.0, 58.0, 6.0),
        ]

        self.fig, self.ax = plt.subplots(figsize=(9, 8))

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_click,
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )

        self.draw_environment()
        self.print_instructions()

    def print_instructions(self) -> None:
        print("=" * 60)
        print("交互式 RRT 路径规划")
        print("左键：设置目标点并开始规划")
        print("右键：清空当前规划结果")
        print("R 键：清空当前规划结果")
        print("Esc 键：关闭窗口")
        print("=" * 60)

    def draw_environment(self) -> None:
        """绘制地图、起点和障碍物。"""
        self.ax.clear()

        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(
            "Interactive RRT Planner\n"
            "Left click: set goal | Right click / R: reset"
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)

        # 绘制矩形障碍物
        for x, y, width, height in self.rect_obstacles:
            rectangle = plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor="gray",
                edgecolor="black",
                alpha=0.8,
            )
            self.ax.add_patch(rectangle)

        # 绘制圆形障碍物
        for cx, cy, radius in self.circle_obstacles:
            circle = plt.Circle(
                (cx, cy),
                radius,
                facecolor="gray",
                edgecolor="black",
                alpha=0.8,
            )
            self.ax.add_patch(circle)

        # 绘制起点
        self.ax.scatter(
            self.start[0],
            self.start[1],
            c="red",
            s=100,
            marker="o",
            edgecolors="black",
            zorder=10,
            label="Start",
        )

        # 如果目标点已经存在，则绘制目标点
        if self.goal is not None:
            self.ax.scatter(
                self.goal[0],
                self.goal[1],
                c="limegreen",
                s=130,
                marker="*",
                edgecolors="black",
                zorder=10,
                label="Goal",
            )

        self.ax.legend(loc="upper left")
        self.fig.canvas.draw_idle()

    def on_mouse_click(self, event) -> None:
        """处理鼠标点击事件。"""
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # 鼠标右键：重置
        if event.button == 3:
            self.reset()
            return

        # 只响应鼠标左键
        if event.button != 1:
            return

        if self.is_planning:
            print("当前正在规划，请等待本次搜索结束。")
            return

        clicked_goal = (float(event.xdata), float(event.ydata))

        if not self.is_point_valid(clicked_goal[0], clicked_goal[1]):
            print(
                f"目标点 {clicked_goal} 位于障碍物内或地图外，"
                "请重新点击。"
            )
            return

        if not self.is_point_valid(self.start[0], self.start[1]):
            print("起点位于障碍物内，无法规划。")
            return

        self.goal = clicked_goal

        print()
        print(
            f"目标点设置为："
            f"({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        )

        self.plan()

    def on_key_press(self, event) -> None:
        """处理键盘事件。"""
        if event.key in ("r", "R"):
            self.reset()
        elif event.key == "escape":
            plt.close(self.fig)

    def reset(self) -> None:
        """清空当前搜索树和目标点。"""
        if self.is_planning:
            print("当前正在规划，暂时无法重置。")
            return

        self.goal = None
        self.nodes = []
        self.draw_environment()
        print("规划结果已清空。")

    def plan(self) -> None:
        """执行RRT路径规划。"""
        if self.goal is None:
            return

        self.is_planning = True
        self.nodes = [
            Node(
                x=self.start[0],
                y=self.start[1],
                parent=None,
            )
        ]

        self.draw_environment()

        goal_node_index: Optional[int] = None

        print("开始执行 RRT 搜索……")

        for iteration in range(self.max_iterations):
            sampled_x, sampled_y = self.sample_point()

            nearest_index = self.find_nearest_node_index(
                sampled_x,
                sampled_y,
            )
            nearest_node = self.nodes[nearest_index]

            new_node = self.steer(
                nearest_node,
                sampled_x,
                sampled_y,
                nearest_index,
            )

            if new_node is None:
                continue

            if not self.is_point_valid(new_node.x, new_node.y):
                continue

            if not self.is_segment_collision_free(
                nearest_node.x,
                nearest_node.y,
                new_node.x,
                new_node.y,
            ):
                continue

            self.nodes.append(new_node)
            new_node_index = len(self.nodes) - 1

            # 实时绘制新扩展的边
            self.ax.plot(
                [nearest_node.x, new_node.x],
                [nearest_node.y, new_node.y],
                color="lightskyblue",
                linewidth=0.8,
                alpha=0.8,
                zorder=2,
            )

            # 判断是否接近目标点
            distance_to_goal = self.distance(
                new_node.x,
                new_node.y,
                self.goal[0],
                self.goal[1],
            )

            if distance_to_goal <= self.goal_tolerance:
                # 接近目标后还必须检查能否直接连接
                if self.is_segment_collision_free(
                    new_node.x,
                    new_node.y,
                    self.goal[0],
                    self.goal[1],
                ):
                    goal_node = Node(
                        x=self.goal[0],
                        y=self.goal[1],
                        parent=new_node_index,
                    )
                    self.nodes.append(goal_node)
                    goal_node_index = len(self.nodes) - 1
                    break

            if iteration % self.animation_interval == 0:
                self.ax.set_title(
                    "Interactive RRT Planner\n"
                    f"Searching: iteration {iteration + 1}, "
                    f"nodes {len(self.nodes)}"
                )
                self.fig.canvas.draw_idle()
                plt.pause(0.001)

        if goal_node_index is not None:
            path = self.extract_path(goal_node_index)
            self.draw_final_path(path)

            path_length = self.calculate_path_length(path)

            print("规划成功。")
            print(f"搜索节点数：{len(self.nodes)}")
            print(f"路径节点数：{len(path)}")
            print(f"路径长度：{path_length:.2f}")
        else:
            self.ax.set_title(
                "RRT planning failed\n"
                "Click another goal or adjust parameters"
            )
            self.fig.canvas.draw_idle()

            print(
                f"规划失败：达到最大迭代次数 "
                f"{self.max_iterations}。"
            )
            print("可以重新点击目标点，或增大 max_iterations。")

        self.is_planning = False
        plt.pause(0.001)

    def sample_point(self) -> tuple[float, float]:
        """
        对状态空间进行随机采样。

        以一定概率直接采样目标点，即目标偏置。
        """
        if self.goal is not None and random.random() < self.goal_sample_rate:
            return self.goal

        return (
            random.uniform(0.0, self.map_width),
            random.uniform(0.0, self.map_height),
        )

    def find_nearest_node_index(
        self,
        sample_x: float,
        sample_y: float,
    ) -> int:
        """查找距离采样点最近的树节点。"""
        minimum_distance_squared = float("inf")
        nearest_index = 0

        for index, node in enumerate(self.nodes):
            distance_squared = (
                (node.x - sample_x) ** 2
                + (node.y - sample_y) ** 2
            )

            if distance_squared < minimum_distance_squared:
                minimum_distance_squared = distance_squared
                nearest_index = index

        return nearest_index

    def steer(
        self,
        from_node: Node,
        target_x: float,
        target_y: float,
        parent_index: int,
    ) -> Optional[Node]:
        """
        从最近节点朝采样点方向扩展固定步长。
        """
        dx = target_x - from_node.x
        dy = target_y - from_node.y

        distance_to_target = math.hypot(dx, dy)

        if distance_to_target < 1e-9:
            return None

        move_distance = min(
            self.step_size,
            distance_to_target,
        )

        angle = math.atan2(dy, dx)

        new_x = from_node.x + move_distance * math.cos(angle)
        new_y = from_node.y + move_distance * math.sin(angle)

        return Node(
            x=new_x,
            y=new_y,
            parent=parent_index,
        )

    def is_point_valid(self, x: float, y: float) -> bool:
        """判断一个点是否位于地图内且不在障碍物中。"""
        if not (0.0 <= x <= self.map_width):
            return False

        if not (0.0 <= y <= self.map_height):
            return False

        # 矩形障碍物检测
        for rect_x, rect_y, width, height in self.rect_obstacles:
            if (
                rect_x <= x <= rect_x + width
                and rect_y <= y <= rect_y + height
            ):
                return False

        # 圆形障碍物检测
        for center_x, center_y, radius in self.circle_obstacles:
            if self.distance(x, y, center_x, center_y) <= radius:
                return False

        return True

    def is_segment_collision_free(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> bool:
        """
        对两个节点之间的线段进行离散碰撞检测。
        """
        segment_length = self.distance(x1, y1, x2, y2)

        sample_count = max(
            2,
            int(segment_length / self.collision_resolution) + 1,
        )

        for index in range(sample_count + 1):
            ratio = index / sample_count

            sample_x = x1 + ratio * (x2 - x1)
            sample_y = y1 + ratio * (y2 - y1)

            if not self.is_point_valid(sample_x, sample_y):
                return False

        return True

    def extract_path(
        self,
        goal_node_index: int,
    ) -> list[tuple[float, float]]:
        """通过父节点索引从目标点回溯到起点。"""
        path = []
        current_index: Optional[int] = goal_node_index

        while current_index is not None:
            node = self.nodes[current_index]
            path.append((node.x, node.y))
            current_index = node.parent

        path.reverse()
        return path

    def draw_final_path(
        self,
        path: list[tuple[float, float]],
    ) -> None:
        """绘制最终路径。"""
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]

        self.ax.plot(
            path_x,
            path_y,
            color="blue",
            linewidth=3.0,
            zorder=8,
            label="RRT path",
        )

        self.ax.scatter(
            path_x,
            path_y,
            color="blue",
            s=15,
            zorder=9,
        )

        self.ax.scatter(
            self.goal[0],
            self.goal[1],
            c="limegreen",
            s=150,
            marker="*",
            edgecolors="black",
            zorder=10,
        )

        self.ax.set_title(
            f"RRT planning succeeded\n"
            f"Tree nodes: {len(self.nodes)}"
        )

        self.ax.legend(loc="upper left")
        self.fig.canvas.draw_idle()

    @staticmethod
    def calculate_path_length(
        path: list[tuple[float, float]],
    ) -> float:
        """计算路径总长度。"""
        total_length = 0.0

        for index in range(len(path) - 1):
            x1, y1 = path[index]
            x2, y2 = path[index + 1]
            total_length += math.hypot(x2 - x1, y2 - y1)

        return total_length

    @staticmethod
    def distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def run(self) -> None:
        """启动交互窗口。"""
        plt.show()


if __name__ == "__main__":
    random.seed()

    planner = InteractiveRRT(
        map_width=100.0,
        map_height=100.0,
        start=(10.0, 10.0),

        # 每次扩展的距离
        step_size=10.0,

        # 10%的概率直接采样目标点
        goal_sample_rate=0.10,

        # 距离目标点5以内时尝试连接目标
        goal_tolerance=10.0,

        # 最大迭代次数
        max_iterations=3000,

        # 碰撞检测精度
        collision_resolution=0.5,

        # 每5次扩展刷新一次画面
        animation_interval=5,
    )

    planner.run()