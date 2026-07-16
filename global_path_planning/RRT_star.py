import math
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


@dataclass
class Node:
    """RRT* 搜索树节点。"""

    x: float
    y: float

    # 父节点在 nodes 列表中的索引
    parent: Optional[int] = None

    # 从起点到当前节点的累计路径代价
    cost: float = 0.0


class InteractiveRRTStar:
    def __init__(
        self,
        map_width: float = 100.0,
        map_height: float = 100.0,
        start: tuple[float, float] = (10.0, 10.0),
        step_size: float = 4.0,
        goal_sample_rate: float = 0.08,
        goal_connection_radius: float = 6.0,
        rewire_radius: float = 15.0,
        max_iterations: int = 2500,
        collision_resolution: float = 0.5,
        animation_interval: int = 15,
    ):
        """
        参数
        ----------
        map_width, map_height:
            地图尺寸。

        start:
            起点坐标。

        step_size:
            每次从最近节点向采样点扩展的最大距离。

        goal_sample_rate:
            直接采样目标点的概率。

        goal_connection_radius:
            节点距离目标点小于该值时，尝试连接目标点。

        rewire_radius:
            RRT* 选择父节点和重新连接时的最大邻域半径。

        max_iterations:
            最大迭代次数。

        collision_resolution:
            线段碰撞检测时，相邻检测点的距离。

        animation_interval:
            每成功加入多少个节点刷新一次图像。
        """

        self.map_width = map_width
        self.map_height = map_height
        self.start = start

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_connection_radius = goal_connection_radius
        self.rewire_radius = rewire_radius
        self.max_iterations = max_iterations
        self.collision_resolution = collision_resolution
        self.animation_interval = animation_interval

        self.goal: Optional[tuple[float, float]] = None
        self.nodes: list[Node] = []

        self.is_planning = False

        # 当前找到的最优目标连接节点
        self.best_goal_parent: Optional[int] = None

        # 当前最优路径总代价
        self.best_goal_cost = float("inf")

        # 矩形障碍物：
        # (左下角 x, 左下角 y, 宽度, 高度)
        self.rect_obstacles = [
            (20.0, 20.0, 18.0, 12.0),
            (48.0, 10.0, 12.0, 35.0),
            (15.0, 55.0, 25.0, 12.0),
            (62.0, 60.0, 25.0, 10.0),
            (45.0, 78.0, 12.0, 16.0),
        ]

        # 圆形障碍物：
        # (圆心 x, 圆心 y, 半径)
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

        self.draw_scene()
        self.print_instructions()

    def print_instructions(self) -> None:
        print("=" * 60)
        print("交互式 RRT* 路径规划")
        print("鼠标左键：设置目标点并开始规划")
        print("鼠标右键：清空规划结果")
        print("R 键：清空规划结果")
        print("Esc 键：关闭窗口")
        print("=" * 60)

    def on_mouse_click(self, event) -> None:
        """响应鼠标点击事件。"""

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # 鼠标右键
        if event.button == 3:
            self.reset()
            return

        # 仅处理鼠标左键
        if event.button != 1:
            return

        if self.is_planning:
            print("当前正在规划，请等待本次规划结束。")
            return

        clicked_goal = (
            float(event.xdata),
            float(event.ydata),
        )

        if not self.is_point_valid(
            clicked_goal[0],
            clicked_goal[1],
        ):
            print(
                f"目标点 ({clicked_goal[0]:.2f}, "
                f"{clicked_goal[1]:.2f}) 位于障碍物内或地图外。"
            )
            return

        if not self.is_point_valid(
            self.start[0],
            self.start[1],
        ):
            print("起点位于障碍物内，无法规划。")
            return

        self.goal = clicked_goal

        print()
        print(
            f"目标点：({self.goal[0]:.2f}, "
            f"{self.goal[1]:.2f})"
        )

        self.plan()

    def on_key_press(self, event) -> None:
        """响应键盘事件。"""

        if event.key in ("r", "R"):
            self.reset()

        elif event.key == "escape":
            plt.close(self.fig)

    def reset(self) -> None:
        """清空搜索结果。"""

        if self.is_planning:
            print("当前正在规划，暂时无法重置。")
            return

        self.goal = None
        self.nodes = []

        self.best_goal_parent = None
        self.best_goal_cost = float("inf")

        self.draw_scene()

        print("规划结果已清空。")

    def plan(self) -> None:
        """运行 RRT* 路径规划。"""

        if self.goal is None:
            return

        self.is_planning = True

        self.nodes = [
            Node(
                x=self.start[0],
                y=self.start[1],
                parent=None,
                cost=0.0,
            )
        ]

        self.best_goal_parent = None
        self.best_goal_cost = float("inf")

        self.draw_scene(
            iteration=0,
            status="开始搜索",
        )

        print("开始执行 RRT* 搜索……")

        successful_expansions = 0

        for iteration in range(1, self.max_iterations + 1):
            # 如果窗口已经关闭，停止搜索
            if not plt.fignum_exists(self.fig.number):
                break

            # 1. 随机采样
            sample_x, sample_y = self.sample_point()

            # 2. 寻找距离采样点最近的节点
            nearest_index = self.find_nearest_node_index(
                sample_x,
                sample_y,
            )

            nearest_node = self.nodes[nearest_index]

            # 3. 从最近节点朝采样点扩展
            new_node = self.steer(
                from_node=nearest_node,
                target_x=sample_x,
                target_y=sample_y,
                parent_index=nearest_index,
            )

            if new_node is None:
                continue

            # 新节点本身不能位于障碍物内
            if not self.is_point_valid(
                new_node.x,
                new_node.y,
            ):
                continue

            # 最近节点到新节点的连线不能碰撞
            if not self.is_segment_collision_free(
                nearest_node.x,
                nearest_node.y,
                new_node.x,
                new_node.y,
            ):
                continue

            # 4. 找出新节点附近的已有节点
            near_indices = self.find_near_node_indices(
                new_node,
            )

            # 5. 从附近节点中选择代价最低的父节点
            best_parent, best_cost = self.choose_best_parent(
                new_node=new_node,
                nearest_index=nearest_index,
                near_indices=near_indices,
            )

            new_node.parent = best_parent
            new_node.cost = best_cost

            # 将新节点加入搜索树
            self.nodes.append(new_node)
            new_index = len(self.nodes) - 1

            successful_expansions += 1

            # 6. 使用新节点重新连接附近节点
            self.rewire(
                new_index=new_index,
                near_indices=near_indices,
            )

            # 7. 判断是否可以连接目标点
            self.update_goal_connection(new_index)

            # 8. 实时刷新画面
            if successful_expansions % self.animation_interval == 0:
                # 重新检查一次所有目标连接，
                # 因为重连可能降低已有节点的路径代价
                self.find_best_goal_connection()

                current_path = None

                if self.best_goal_parent is not None:
                    current_path = self.extract_goal_path(
                        self.best_goal_parent
                    )

                self.draw_scene(
                    path=current_path,
                    iteration=iteration,
                    status="正在优化",
                )

                plt.pause(0.001)

        # 搜索结束后重新计算最终最优目标连接
        self.find_best_goal_connection()

        if self.best_goal_parent is not None:
            final_path = self.extract_goal_path(
                self.best_goal_parent
            )

            self.draw_scene(
                path=final_path,
                iteration=self.max_iterations,
                status="规划成功",
            )

            print("RRT* 规划成功。")
            print(f"搜索树节点数：{len(self.nodes)}")
            print(f"路径节点数：{len(final_path)}")
            print(f"最终路径长度：{self.best_goal_cost:.2f}")

        else:
            self.draw_scene(
                iteration=self.max_iterations,
                status="规划失败",
            )

            print(
                f"规划失败：在 {self.max_iterations} "
                "次迭代内未找到可行路径。"
            )

        self.is_planning = False
        plt.pause(0.001)

    def sample_point(self) -> tuple[float, float]:
        """
        在地图中随机采样。

        以 goal_sample_rate 的概率直接返回目标点，
        这种方法称为目标偏置。
        """

        if (
            self.goal is not None
            and random.random() < self.goal_sample_rate
        ):
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
        """寻找距离采样点最近的节点。"""

        nearest_index = 0
        minimum_distance_squared = float("inf")

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
        从 from_node 朝目标方向扩展。

        每次扩展长度不超过 step_size。
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

        new_x = (
            from_node.x
            + move_distance * math.cos(angle)
        )

        new_y = (
            from_node.y
            + move_distance * math.sin(angle)
        )

        return Node(
            x=new_x,
            y=new_y,
            parent=parent_index,
            cost=from_node.cost + move_distance,
        )

    def find_near_node_indices(
        self,
        new_node: Node,
    ) -> list[int]:
        """
        寻找新节点附近的已有节点。

        邻域半径会随节点数增加而逐渐缩小，但不会低于
        2 倍 step_size。
        """

        node_count = len(self.nodes) + 1

        if node_count <= 1:
            return []

        dynamic_radius = (
            self.rewire_radius
            * math.sqrt(
                math.log(node_count) / node_count
            )
        )

        radius = max(
            self.step_size * 2.0,
            dynamic_radius,
        )

        radius = min(
            radius,
            self.rewire_radius,
        )

        radius_squared = radius**2

        near_indices = []

        for index, node in enumerate(self.nodes):
            distance_squared = (
                (node.x - new_node.x) ** 2
                + (node.y - new_node.y) ** 2
            )

            if distance_squared <= radius_squared:
                near_indices.append(index)

        return near_indices

    def choose_best_parent(
        self,
        new_node: Node,
        nearest_index: int,
        near_indices: list[int],
    ) -> tuple[int, float]:
        """
        在附近节点中选择路径总代价最小的父节点。

        这是 RRT* 相比普通 RRT 的第一个核心步骤。
        """

        nearest_node = self.nodes[nearest_index]

        best_parent = nearest_index

        best_cost = (
            nearest_node.cost
            + self.distance(
                nearest_node.x,
                nearest_node.y,
                new_node.x,
                new_node.y,
            )
        )

        for near_index in near_indices:
            near_node = self.nodes[near_index]

            connection_distance = self.distance(
                near_node.x,
                near_node.y,
                new_node.x,
                new_node.y,
            )

            candidate_cost = (
                near_node.cost
                + connection_distance
            )

            if candidate_cost >= best_cost:
                continue

            if not self.is_segment_collision_free(
                near_node.x,
                near_node.y,
                new_node.x,
                new_node.y,
            ):
                continue

            best_parent = near_index
            best_cost = candidate_cost

        return best_parent, best_cost

    def rewire(
        self,
        new_index: int,
        near_indices: list[int],
    ) -> None:
        """
        尝试通过新节点降低附近已有节点的路径代价。

        这是 RRT* 相比普通 RRT 的第二个核心步骤。
        """

        new_node = self.nodes[new_index]

        for near_index in near_indices:
            # 新节点自己的父节点不需要重连
            if near_index == new_node.parent:
                continue

            near_node = self.nodes[near_index]

            connection_distance = self.distance(
                new_node.x,
                new_node.y,
                near_node.x,
                near_node.y,
            )

            rewired_cost = (
                new_node.cost
                + connection_distance
            )

            # 新路径没有更优
            if rewired_cost >= near_node.cost - 1e-9:
                continue

            # 新连接发生碰撞
            if not self.is_segment_collision_free(
                new_node.x,
                new_node.y,
                near_node.x,
                near_node.y,
            ):
                continue

            # 更新父节点和累计代价
            near_node.parent = new_index
            near_node.cost = rewired_cost

            # 该节点代价变化后，需要同步更新它的所有后代
            self.update_descendant_costs(near_index)

    def update_descendant_costs(
        self,
        root_index: int,
    ) -> None:
        """更新某个节点所有后代的累计路径代价。"""

        stack = [root_index]

        while stack:
            parent_index = stack.pop()
            parent_node = self.nodes[parent_index]

            for child_index, child_node in enumerate(self.nodes):
                if child_node.parent != parent_index:
                    continue

                child_node.cost = (
                    parent_node.cost
                    + self.distance(
                        parent_node.x,
                        parent_node.y,
                        child_node.x,
                        child_node.y,
                    )
                )

                stack.append(child_index)

    def update_goal_connection(
        self,
        node_index: int,
    ) -> None:
        """检查指定节点是否能够连接目标点。"""

        if self.goal is None:
            return

        node = self.nodes[node_index]

        distance_to_goal = self.distance(
            node.x,
            node.y,
            self.goal[0],
            self.goal[1],
        )

        if distance_to_goal > self.goal_connection_radius:
            return

        if not self.is_segment_collision_free(
            node.x,
            node.y,
            self.goal[0],
            self.goal[1],
        ):
            return

        total_cost = (
            node.cost
            + distance_to_goal
        )

        if total_cost < self.best_goal_cost:
            self.best_goal_parent = node_index
            self.best_goal_cost = total_cost

    def find_best_goal_connection(self) -> None:
        """
        遍历搜索树，寻找连接目标点总代价最低的节点。

        重连后旧节点的代价可能下降，因此需要重新检查。
        """

        if self.goal is None:
            return

        best_parent = None
        best_cost = float("inf")

        for index, node in enumerate(self.nodes):
            distance_to_goal = self.distance(
                node.x,
                node.y,
                self.goal[0],
                self.goal[1],
            )

            if distance_to_goal > self.goal_connection_radius:
                continue

            if not self.is_segment_collision_free(
                node.x,
                node.y,
                self.goal[0],
                self.goal[1],
            ):
                continue

            total_cost = (
                node.cost
                + distance_to_goal
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_parent = index

        self.best_goal_parent = best_parent
        self.best_goal_cost = best_cost

    def extract_goal_path(
        self,
        goal_parent_index: int,
    ) -> list[tuple[float, float]]:
        """从目标连接节点回溯到起点。"""

        if self.goal is None:
            return []

        path = []

        current_index: Optional[int] = goal_parent_index

        while current_index is not None:
            node = self.nodes[current_index]

            path.append(
                (node.x, node.y)
            )

            current_index = node.parent

        # 当前顺序为目标附近节点到起点
        path.reverse()

        # 添加真实目标点
        path.append(self.goal)

        return path

    def is_point_valid(
        self,
        x: float,
        y: float,
    ) -> bool:
        """判断点是否在地图内且不位于障碍物内。"""

        if not (0.0 <= x <= self.map_width):
            return False

        if not (0.0 <= y <= self.map_height):
            return False

        # 矩形障碍物
        for rect_x, rect_y, width, height in self.rect_obstacles:
            if (
                rect_x <= x <= rect_x + width
                and rect_y <= y <= rect_y + height
            ):
                return False

        # 圆形障碍物
        for center_x, center_y, radius in self.circle_obstacles:
            if (
                self.distance(
                    x,
                    y,
                    center_x,
                    center_y,
                )
                <= radius
            ):
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

        segment_length = self.distance(
            x1,
            y1,
            x2,
            y2,
        )

        sample_count = max(
            2,
            math.ceil(
                segment_length
                / self.collision_resolution
            ),
        )

        for index in range(sample_count + 1):
            ratio = index / sample_count

            sample_x = x1 + ratio * (x2 - x1)
            sample_y = y1 + ratio * (y2 - y1)

            if not self.is_point_valid(
                sample_x,
                sample_y,
            ):
                return False

        return True

    def draw_scene(
        self,
        path: Optional[list[tuple[float, float]]] = None,
        iteration: Optional[int] = None,
        status: str = "点击设置目标点",
    ) -> None:
        """绘制地图、搜索树和最优路径。"""

        self.ax.clear()

        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.ax.grid(
            True,
            alpha=0.3,
        )

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
        for center_x, center_y, radius in self.circle_obstacles:
            circle = plt.Circle(
                (center_x, center_y),
                radius,
                facecolor="gray",
                edgecolor="black",
                alpha=0.8,
            )

            self.ax.add_patch(circle)

        # 绘制搜索树
        if len(self.nodes) > 1:
            tree_segments = []

            for node in self.nodes:
                if node.parent is None:
                    continue

                parent_node = self.nodes[node.parent]

                tree_segments.append(
                    [
                        (parent_node.x, parent_node.y),
                        (node.x, node.y),
                    ]
                )

            tree_collection = LineCollection(
                tree_segments,
                colors="lightskyblue",
                linewidths=0.7,
                alpha=0.8,
                zorder=2,
            )

            self.ax.add_collection(tree_collection)

            node_x = [node.x for node in self.nodes]
            node_y = [node.y for node in self.nodes]

            self.ax.scatter(
                node_x,
                node_y,
                s=4,
                color="deepskyblue",
                alpha=0.7,
                zorder=3,
            )

        # 绘制当前最优路径
        if path:
            path_x = [
                point[0]
                for point in path
            ]

            path_y = [
                point[1]
                for point in path
            ]

            self.ax.plot(
                path_x,
                path_y,
                color="blue",
                linewidth=3.0,
                zorder=8,
                label="Best path",
            )

            self.ax.scatter(
                path_x,
                path_y,
                color="blue",
                s=18,
                zorder=9,
            )

        # 起点
        self.ax.scatter(
            self.start[0],
            self.start[1],
            color="red",
            s=100,
            marker="o",
            edgecolors="black",
            zorder=10,
            label="Start",
        )

        # 目标点
        if self.goal is not None:
            self.ax.scatter(
                self.goal[0],
                self.goal[1],
                color="limegreen",
                s=160,
                marker="*",
                edgecolors="black",
                zorder=10,
                label="Goal",
            )

        title_lines = [
            "Interactive RRT* Planner",
            status,
        ]

        if iteration is not None:
            title_lines.append(
                f"Iteration: {iteration} | "
                f"Nodes: {len(self.nodes)}"
            )

        if self.best_goal_parent is not None:
            title_lines.append(
                f"Best cost: {self.best_goal_cost:.2f}"
            )

        self.ax.set_title(
            "\n".join(title_lines)
        )

        self.ax.legend(
            loc="upper left"
        )

        self.fig.canvas.draw_idle()

    @staticmethod
    def distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """计算二维欧氏距离。"""

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )

    def run(self) -> None:
        """显示交互窗口。"""

        plt.show()


if __name__ == "__main__":
    # 设置固定种子可获得可重复结果。
    # 删除这一行或改成 random.seed()，每次结果会不同。
    random.seed(10)

    planner = InteractiveRRTStar(
        map_width=100.0,
        map_height=100.0,
        start=(10.0, 10.0),

        # 每次最大扩展距离
        step_size=4.0,

        # 8% 概率直接采样目标点
        goal_sample_rate=0.08,

        # 节点进入该范围后尝试连接目标
        goal_connection_radius=6.0,

        # 选择父节点和重连的最大邻域范围
        rewire_radius=15.0,

        # RRT* 找到路径后仍会继续优化
        max_iterations=2500,

        # 碰撞检测采样间隔
        collision_resolution=0.5,

        # 每成功增加 15 个节点刷新一次
        animation_interval=15,
    )

    planner.run()