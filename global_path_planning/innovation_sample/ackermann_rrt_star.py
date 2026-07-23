import sys
import os
import math
import time
from dataclasses import dataclass, field
from typing import Optional

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vehicle.reeds_shepp_path import Pose, plan_reeds_shepp_path
from vehicle.dubins_path_test import plan_dubins_path
from vehicle.vehicle_collision import (
    VehicleGeometry,
    CircleObstacle,
    check_path_collision,
    check_pose_collision,
    get_vehicle_corners,
)

from global_path_planning.innovation_sample.hybrid_sampler import HybridSampler


@dataclass
class EdgePath:
    x: list
    y: list
    yaw: list
    directions: list

    @property
    def length(self):
        length = 0.0
        for i in range(1, len(self.x)):
            length += math.hypot(self.x[i] - self.x[i-1], self.y[i] - self.y[i-1])
        return length


@dataclass
class Node:
    pose: Pose
    parent: Optional[int] = None
    cost: float = 0.0
    path_x: list = field(default_factory=list)
    path_y: list = field(default_factory=list)
    path_yaw: list = field(default_factory=list)
    path_directions: list = field(default_factory=list)


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_difference(a, b):
    return normalize_angle(a - b)


def interpolate_angle(a, b, ratio):
    return normalize_angle(a + ratio * angle_difference(b, a))


def truncate_path(path, max_length):
    x = [path.x[0]]
    y = [path.y[0]]
    yaw = [path.yaw[0]]
    direction = [path.directions[0]]

    length = 0.0

    for i in range(1, len(path.x)):
        dx = path.x[i] - path.x[i-1]
        dy = path.y[i] - path.y[i-1]
        ds = math.hypot(dx, dy)

        if length + ds <= max_length:
            x.append(path.x[i])
            y.append(path.y[i])
            yaw.append(path.yaw[i])
            direction.append(path.directions[i])
            length += ds
        else:
            ratio = (max_length - length) / ds
            x.append(path.x[i-1] + ratio * dx)
            y.append(path.y[i-1] + ratio * dy)
            yaw.append(interpolate_angle(path.yaw[i-1], path.yaw[i], ratio))
            direction.append(path.directions[i])
            break

    return EdgePath(x, y, yaw, direction)


class AckermannRRTStar:
    def __init__(
        self,
        start,
        goal,
        bounds,
        vehicle,
        obstacles,
        curvature,
        use_ackermann_constraints=True,
        expand_length=3.0,
        step_size=0.02,
        goal_connect_distance=7.0,
        near_radius=5.0,
        max_iterations=5000,
        yaw_weight=2.0,
        use_hybrid_sampling=True,
        corridors=None,
        goal_rectangle=None,
        rectangle_anchor_mode="latest",
        goal_probability=0.10,
        tangent_probability=0.05,
        corridor_probability=0.15,
        rectangle_probability=0.30,
        allow_reverse=False,
        use_tangent_guidance=True,
        tangent_clearance=0.5,
        tangent_extension=0.3,
        shrink_probability=0.60,
        shrink_length_factor=0.70,
        shrink_width_factor=0.65,
        shrink_activation_distance=20.0,
        near_anchor_probability=0.60,
        near_anchor_length_ratio=0.40,
        adaptive_sampling_probabilities=True,
        cluster_shape="ellipse",
        use_goal_connector=False,
        relax_goal_yaw=False,
        random_seed=0,
        tangent_distance_threshold=12.0,
        single_cluster_tangent_scale=0.30,
        multi_cluster_tangent_scale=1.00,
        tangent_target_tolerance=1.0,
        tangent_along_std=0.30,
        tangent_lateral_std=0.20,
        max_guidance_updates=20,
        tangent_detour_weight=20.0,
        max_tangent_detour_ratio=1.15,
        remaining_blocker_weight=2.0,
        side_switch_penalty=3.0,
        post_solution_iterations=300,
    ):

        self.start = start
        self.goal = goal

        self.bounds = bounds
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

        self.vehicle = vehicle
        self.obstacles = obstacles
        self.curvature = curvature
        self.use_ackermann_constraints = bool(use_ackermann_constraints)

        self.expand_length = expand_length
        self.step_size = step_size
        self.goal_connect_distance = goal_connect_distance

        self.near_radius = near_radius
        self.max_iterations = max_iterations

        self.yaw_weight = yaw_weight
        self.rectangle_anchor_mode = rectangle_anchor_mode
        self.allow_reverse = allow_reverse
        self.use_goal_connector = use_goal_connector
        self.relax_goal_yaw = relax_goal_yaw
        self.post_solution_iterations = max(0, int(post_solution_iterations))

        vehicle_radius = math.hypot(
            max(vehicle.front_length, vehicle.rear_length)
            + vehicle.safety_margin,
            vehicle.width / 2.0 + vehicle.safety_margin,
        )

        self.nodes = [Node(start)]
        self.latest_expanded_index = 0

        self.goal_index = None

        self.first_solution_iteration = None
        self.best_cost = float("inf")

        self.cost_history = []

        self.start_time = None
        self.planning_time = None

        self.sampler = HybridSampler(
            bounds=bounds,
            goal=goal,
            corridors=corridors,
            goal_rectangle=goal_rectangle,
            use_hybrid_sampling=use_hybrid_sampling,
            goal_probability=goal_probability,
            tangent_probability=tangent_probability,
            corridor_probability=corridor_probability,
            rectangle_probability=rectangle_probability,
            obstacles=obstacles,
            use_tangent_guidance=use_tangent_guidance,
            obstacle_inflation=vehicle_radius + tangent_clearance,
            tangent_extension=tangent_extension,
            shrink_probability=shrink_probability,
            shrink_length_factor=shrink_length_factor,
            shrink_width_factor=shrink_width_factor,
            shrink_activation_distance=shrink_activation_distance,
            near_anchor_probability=near_anchor_probability,
            near_anchor_length_ratio=near_anchor_length_ratio,
            adaptive_probabilities=adaptive_sampling_probabilities,
            cluster_shape=cluster_shape,
            random_seed=random_seed,
            tangent_distance_threshold=tangent_distance_threshold,
            single_cluster_tangent_scale=single_cluster_tangent_scale,
            multi_cluster_tangent_scale=multi_cluster_tangent_scale,
            tangent_target_tolerance=tangent_target_tolerance,
            tangent_along_std=tangent_along_std,
            tangent_lateral_std=tangent_lateral_std,
            max_guidance_updates=max_guidance_updates,
            tangent_detour_weight=tangent_detour_weight,
            max_tangent_detour_ratio=max_tangent_detour_ratio,
            remaining_blocker_weight=remaining_blocker_weight,
            side_switch_penalty=side_switch_penalty,
        )

    def rectangle_anchor_node(self):
        """返回用于构造动态目标矩形的树末端节点。"""
        if self.rectangle_anchor_mode == "closest_to_goal":
            return min(
                self.nodes,
                key=lambda node: math.hypot(
                    node.pose.x - self.goal.x,
                    node.pose.y - self.goal.y,
                ),
            )

        # 默认使用最近一次正常扩展得到的末端节点；check_goal() 为验证连接
        # 而插入的目标节点不会改变这个锚点。
        return self.nodes[self.latest_expanded_index]


    def state_distance(self, p1, p2):
        pos = math.hypot(p1.x - p2.x, p1.y - p2.y)
        yaw = abs(angle_difference(p1.yaw, p2.yaw))
        return pos + self.yaw_weight * yaw


    def nearest_node(self, pose):
        distances = [self.state_distance(node.pose, pose) for node in self.nodes]
        return min(range(len(distances)), key=distances.__getitem__)


    def near_nodes(self, pose):
        indexes = []
        for i, node in enumerate(self.nodes):
            if self.state_distance(node.pose, pose) < self.near_radius:
                indexes.append(i)
        return indexes


    def steer(self, from_pose, to_pose, max_length=None):
        distance = math.hypot(to_pose.x - from_pose.x, to_pose.y - from_pose.y)
        if distance < 1e-4:
            return None

        if not self.use_ackermann_constraints:
            # Classic geometric RRT*: connect states with a collision-checked
            # straight segment and do not impose a turning-radius constraint.
            heading = math.atan2(to_pose.y - from_pose.y, to_pose.x - from_pose.x)
            sample_count = max(2, int(math.ceil(distance / self.step_size)) + 1)
            path = EdgePath(
                [from_pose.x + (to_pose.x - from_pose.x) * i / (sample_count - 1)
                 for i in range(sample_count)],
                [from_pose.y + (to_pose.y - from_pose.y) * i / (sample_count - 1)
                 for i in range(sample_count)],
                [heading] * sample_count,
                [1] * sample_count,
            )
            return truncate_path(path, max_length) if max_length else path

        if self.allow_reverse:
            path = plan_reeds_shepp_path(
                from_pose,
                to_pose,
                curvature=self.curvature,
                step_size=self.step_size,
            )
        else:
            path = plan_dubins_path(
                from_pose,
                to_pose,
                curvature=self.curvature,
                step_size=self.step_size,
            )

        if path is None:
            return None

        directions = (
            list(path.directions)
            if self.allow_reverse
            else [1] * len(path.x)
        )
        result = EdgePath(
            list(path.x),
            list(path.y),
            list(path.yaw),
            directions,
        )
        if max_length:
            result = truncate_path(result, max_length)
        return result


    def pose_collision_free(self, pose):
        return not check_pose_collision(pose, self.vehicle, self.obstacles)


    def path_collision_free(self, path):
        collision, _ = check_path_collision(path, self.vehicle, self.obstacles)
        return not collision

    def choose_parent(self, near_indexes, nearest_index, nearest_path):
        best_parent = nearest_index
        best_path = nearest_path
        best_cost = self.nodes[nearest_index].cost + nearest_path.length

        for index in near_indexes:
            node = self.nodes[index]
            path = self.steer(node.pose, Pose(nearest_path.x[-1], nearest_path.y[-1], nearest_path.yaw[-1]), self.expand_length)
            if path is None or not self.path_collision_free(path):
                continue
            cost = node.cost + path.length
            if cost < best_cost:
                best_cost = cost
                best_parent = index
                best_path = path

        return best_parent, best_path


    def add_node(self, parent_index, path):
        parent = self.nodes[parent_index]
        node = Node(
            pose=Pose(path.x[-1], path.y[-1], path.yaw[-1]),
            parent=parent_index,
            cost=parent.cost + path.length,
            path_x=path.x,
            path_y=path.y,
            path_yaw=path.yaw,
            path_directions=path.directions,
        )
        self.nodes.append(node)
        return len(self.nodes) - 1


    def rewire(self, new_index, near_indexes):
        new_node = self.nodes[new_index]
        for index in near_indexes:
            if index == new_index:
                continue
            node = self.nodes[index]
            path = self.steer(new_node.pose, node.pose, self.expand_length)
            if path is None or not self.path_collision_free(path):
                continue
            new_cost = new_node.cost + path.length
            if new_cost < node.cost:
                node.parent = new_index
                node.cost = new_cost
                node.path_x = path.x
                node.path_y = path.y
                node.path_yaw = path.yaw
                node.path_directions = path.directions


    def check_goal(self, index):
        node = self.nodes[index]
        distance = math.hypot(node.pose.x - self.goal.x, node.pose.y - self.goal.y)
        if distance > self.goal_connect_distance:
            return False

        if not self.use_goal_connector:
            self.goal_index = index
            return True

        goal_candidates = [self.goal]
        if self.relax_goal_yaw:
            approach_yaw = math.atan2(
                self.goal.y - node.pose.y,
                self.goal.x - node.pose.x,
            )
            for offset in (0.0, math.radians(15.0), -math.radians(15.0)):
                goal_candidates.append(
                    Pose(
                        self.goal.x,
                        self.goal.y,
                        normalize_angle(approach_yaw + offset),
                    )
                )

        best_path = None
        for goal_pose in goal_candidates:
            path = self.steer(node.pose, goal_pose)
            if path is None or not self.path_collision_free(path):
                continue
            if best_path is None or path.length < best_path.length:
                best_path = path

        if best_path is None:
            return False

        goal_index = self.add_node(index, best_path)
        self.goal_index = goal_index
        return True


    def planning(self, callback=None, callback_interval=10):
        """
        规划路径
        callback: 可选的回调函数，每 callback_interval 次迭代调用一次
        callback_interval: 回调间隔（迭代次数）
        找到第一条路径后继续优化 post_solution_iterations 次
        """
        self.start_time = time.time()

        # 每次规划开始前重置采样器状态
        if hasattr(self.sampler, 'set_feasible_path_found'):
            self.sampler.set_feasible_path_found(False)

        first_solution_iteration = None

        for iteration in range(self.max_iterations):
            if self.sampler.goal_rectangle is not None:
                anchor_node = self.rectangle_anchor_node()
                self.sampler.update_rectangle_anchor(anchor_node.pose)

            self.sampler.update_probability(iteration)
            sx, sy, syaw = self.sampler.sample()
            random_pose = Pose(sx, sy, syaw)
            nearest_index = self.nearest_node(random_pose)
            nearest_pose = self.nodes[nearest_index].pose
            path = self.steer(nearest_pose, random_pose, self.expand_length)
            if path is None or not self.path_collision_free(path):
                continue
            new_pose = Pose(path.x[-1], path.y[-1], path.yaw[-1])
            near_indexes = self.near_nodes(new_pose)
            parent_index, best_path = self.choose_parent(near_indexes, nearest_index, path)
            new_index = self.add_node(parent_index, best_path)
            self.latest_expanded_index = new_index
            if self.sampler.goal_rectangle is not None:
                self.sampler.update_rectangle_anchor(
                    self.nodes[new_index].pose
                )

            self.rewire(new_index, near_indexes)
            cost = self.nodes[new_index].cost
            if cost < self.best_cost:
                self.best_cost = cost
                self.cost_history.append((iteration, cost))

            # 每 callback_interval 次迭代调用一次回调函数
            if callback is not None and iteration % callback_interval == 0:
                callback(iteration)

            # 检查是否到达目标
            if self.check_goal(new_index):
                # 第一次找到解时记录并通知采样器
                if first_solution_iteration is None:
                    first_solution_iteration = iteration
                    self.first_solution_iteration = iteration + 1

                    # 通知采样器首次解已找到，立即关闭切向引导
                    if hasattr(self.sampler, 'set_feasible_path_found'):
                        self.sampler.set_feasible_path_found(True)

            # 首次解后继续优化固定次数
            if first_solution_iteration is not None:
                if iteration - first_solution_iteration >= self.post_solution_iterations:
                    break

        self.planning_time = time.time() - self.start_time
        if self.goal_index is None:
            return None
        return self.extract_path()



    def extract_path(self):
        path_x = []
        path_y = []
        path_yaw = []
        directions = []
        index = self.goal_index
        segments = []

        while self.nodes[index].parent is not None:
            node = self.nodes[index]
            segments.append((node.path_x, node.path_y, node.path_yaw, node.path_directions))
            index = node.parent

        segments.reverse()
        for sx, sy, syaw, sd in segments:
            start = 1 if path_x else 0
            path_x.extend(sx[start:])
            path_y.extend(sy[start:])
            path_yaw.extend(syaw[start:])
            directions.extend(sd[start:])

        return (path_x, path_y, path_yaw, directions)


    def get_metrics(self, result):
        path_x, path_y, _, directions = result
        length = 0.0
        reverse = 0.0
        switch = 0

        for i in range(1, len(path_x)):
            ds = math.hypot(path_x[i] - path_x[i-1], path_y[i] - path_y[i-1])
            length += ds
            if directions[i] < 0:
                reverse += ds
            if directions[i] != directions[i-1]:
                switch += 1

        return {
            "planning_time": self.planning_time,
            "first_solution_iteration": self.first_solution_iteration,
            "node_count": len(self.nodes),
            "path_length": length,
            "reverse_length": reverse,
            "switch_count": switch,
            "best_cost": self.best_cost,
        }
