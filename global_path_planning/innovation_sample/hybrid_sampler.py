import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingCorridor:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float


@dataclass
class GoalRectangle:
    """
    朝向目标的矩形采样区域。

    anchor_x, anchor_y:
        矩形方向的参考起点，一般设置为车辆起点或RRT*根节点。

    length:
        矩形沿目标方向的总长度。

    width:
        矩形横向总宽度。

    forward_offset:
        矩形起点相对于anchor沿目标方向的偏移。
        取0时，矩形从anchor开始向目标方向延伸。
        取负值时，矩形会覆盖anchor后方的一小部分。
    """

    anchor_x: float
    anchor_y: float
    length: float
    width: float
    forward_offset: float = 0.0


@dataclass
class TangentGuidance:
    obstacle_index: Optional[int] = None
    obstacle_indexes: tuple = ()
    side: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    cluster_x: Optional[float] = None
    cluster_y: Optional[float] = None
    ellipse_a: Optional[float] = None
    ellipse_b: Optional[float] = None
    ellipse_yaw: Optional[float] = None

    @property
    def active(self):
        return self.obstacle_index is not None


class HybridSampler:
    def __init__(
        self,
        bounds,
        goal,
        corridors=None,
        goal_rectangle: Optional[GoalRectangle] = None,
        use_hybrid_sampling=True,
        goal_probability=0.10,
        corridor_probability=0.15,
        rectangle_probability=0.30,
        random_seed=0,
        heading_std_degrees=12.0,
        max_sample_attempts=30,
        obstacles=None,
        use_tangent_guidance=True,
        obstacle_inflation=2.5,
        tangent_extension=4.0,
        turn_cost_weight=2.0,
        shrink_probability=0.60,
        shrink_length_factor=0.70,
        shrink_width_factor=0.65,
        min_rectangle_length=10.0,
        min_rectangle_width=8.0,
        shrink_activation_distance=20.0,
        near_anchor_probability=0.60,
        near_anchor_length_ratio=0.40,
        adaptive_probabilities=True,
        cluster_shape="ellipse",
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

        self.goal = goal
        self.corridors = corridors if corridors else []
        self.goal_rectangle = goal_rectangle
        self.obstacles = obstacles if obstacles else []
        self.use_tangent_guidance = use_tangent_guidance
        self.obstacle_inflation = float(obstacle_inflation)
        self.tangent_extension = float(tangent_extension)
        self.turn_cost_weight = float(turn_cost_weight)
        self.shrink_probability = max(0.0, min(1.0, float(shrink_probability)))
        self.shrink_length_factor = max(0.05, min(1.0, float(shrink_length_factor)))
        self.shrink_width_factor = max(0.05, min(1.0, float(shrink_width_factor)))
        self.min_rectangle_length = float(min_rectangle_length)
        self.min_rectangle_width = float(min_rectangle_width)
        self.shrink_activation_distance = max(
            0.0, float(shrink_activation_distance)
        )
        self.near_anchor_probability = max(
            0.0, min(1.0, float(near_anchor_probability))
        )
        self.near_anchor_length_ratio = max(
            0.05, min(1.0, float(near_anchor_length_ratio))
        )
        self.adaptive_probabilities = adaptive_probabilities
        self.cluster_shape = cluster_shape
        self.base_rectangle_length = (
            float(goal_rectangle.length) if goal_rectangle is not None else 0.0
        )
        self.base_rectangle_width = (
            float(goal_rectangle.width) if goal_rectangle is not None else 0.0
        )
        self.rectangle_shrunk = False
        self.blocking_obstacle_count = 0
        self.nearest_blocking_distance = math.inf
        self.shrink_decision_made = False
        self.sampling_target = (float(goal.x), float(goal.y))
        self.tangent_guidance = TangentGuidance()

        self.use_hybrid_sampling = use_hybrid_sampling

        self.initial_goal_probability = goal_probability
        self.initial_corridor_probability = corridor_probability
        self.initial_rectangle_probability = rectangle_probability

        self.goal_probability = goal_probability
        self.corridor_probability = corridor_probability
        self.rectangle_probability = rectangle_probability

        self.uniform_probability = 0.0
        self._normalize_probabilities()

        self.random = random.Random(random_seed)

        self.heading_std = math.radians(heading_std_degrees)
        self.max_sample_attempts = max_sample_attempts

        self.iteration = 0
        self.no_improvement_count = 0
        self.best_cost = math.inf

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def inside_bounds(self, x, y):
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
        )

    def update_rectangle_anchor(self, pose):
        """移动矩形锚点，并更新目标/切向引导方向。"""
        if self.goal_rectangle is None:
            return

        self.goal_rectangle.anchor_x = float(pose.x)
        self.goal_rectangle.anchor_y = float(pose.y)
        self.update_tangent_guidance(pose)

    def inflated_radius(self, obstacle):
        return float(obstacle.radius) + self.obstacle_inflation

    def restore_rectangle_size(self):
        if self.goal_rectangle is None:
            return
        self.goal_rectangle.length = self.base_rectangle_length
        self.goal_rectangle.width = self.base_rectangle_width
        self.rectangle_shrunk = False

    def obstacle_edge_distance(self, pose, obstacle):
        """当前节点到膨胀圆边缘的非负距离。"""
        center_distance = math.hypot(
            obstacle.x - pose.x,
            obstacle.y - pose.y,
        )
        return max(0.0, center_distance - self.inflated_radius(obstacle))

    def maybe_shrink_rectangle(self, pose, obstacle_indexes):
        """阻挡圆超过2个且进入触发距离后，按概率缩小矩形。"""
        obstacle_count = len(obstacle_indexes)
        self.blocking_obstacle_count = obstacle_count
        self.nearest_blocking_distance = min(
            (
                self.obstacle_edge_distance(pose, self.obstacles[index])
                for index in obstacle_indexes
            ),
            default=math.inf,
        )
        if self.goal_rectangle is None or obstacle_count <= 2:
            return False
        if self.nearest_blocking_distance > self.shrink_activation_distance:
            return False
        if self.shrink_decision_made:
            return self.rectangle_shrunk
        self.shrink_decision_made = True
        if self.random.random() >= self.shrink_probability:
            return False

        self.goal_rectangle.length = max(
            self.min_rectangle_length,
            self.base_rectangle_length * self.shrink_length_factor,
        )
        self.goal_rectangle.width = max(
            self.min_rectangle_width,
            self.base_rectangle_width * self.shrink_width_factor,
        )
        self.rectangle_shrunk = True
        return True

    @staticmethod
    def point_segment_distance(point, start, end):
        px, py = point
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 <= 1e-12:
            return math.hypot(px - ax, py - ay)
        ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
        closest_x = ax + ratio * dx
        closest_y = ay + ratio * dy
        return math.hypot(px - closest_x, py - closest_y)

    def segment_is_clear(
        self, start, end, ignored_index=None, ignored_indexes=()
    ):
        ignored = set(ignored_indexes)
        if ignored_index is not None:
            ignored.add(ignored_index)
        for index, obstacle in enumerate(self.obstacles):
            if index in ignored:
                continue
            distance = self.point_segment_distance(
                (obstacle.x, obstacle.y), start, end
            )
            if distance <= self.inflated_radius(obstacle):
                return False
        return True

    def overlapping_cluster(self, seed_index):
        """把相互重叠的膨胀圆合并为一个局部障碍簇。"""
        cluster = {seed_index}
        pending = [seed_index]
        while pending:
            current = pending.pop()
            a = self.obstacles[current]
            for index, b in enumerate(self.obstacles):
                if index in cluster:
                    continue
                center_distance = math.hypot(a.x - b.x, a.y - b.y)
                if center_distance <= self.inflated_radius(a) + self.inflated_radius(b):
                    cluster.add(index)
                    pending.append(index)
        return tuple(sorted(cluster))

    def cluster_enclosing_ellipse(self, obstacle_indexes):
        """沿圆心主方向构造包含所有膨胀圆的定向椭圆。"""
        weights = [
            self.inflated_radius(self.obstacles[index])
            for index in obstacle_indexes
        ]
        total = sum(weights)
        cx = sum(
            self.obstacles[index].x * weight
            for index, weight in zip(obstacle_indexes, weights)
        ) / total
        cy = sum(
            self.obstacles[index].y * weight
            for index, weight in zip(obstacle_indexes, weights)
        ) / total
        cov_xx = cov_xy = cov_yy = 0.0
        for index, weight in zip(obstacle_indexes, weights):
            dx = self.obstacles[index].x - cx
            dy = self.obstacles[index].y - cy
            cov_xx += weight * dx * dx
            cov_xy += weight * dx * dy
            cov_yy += weight * dy * dy
        yaw = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
        c, s = math.cos(yaw), math.sin(yaw)
        circles = []
        a = b = 1e-6
        for index in obstacle_indexes:
            obstacle = self.obstacles[index]
            dx, dy = obstacle.x - cx, obstacle.y - cy
            local_x = dx * c + dy * s
            local_y = -dx * s + dy * c
            radius = self.inflated_radius(obstacle)
            circles.append((local_x, local_y, radius))
            a = max(a, abs(local_x) + radius)
            b = max(b, abs(local_y) + radius)

        # 统一放大长短轴，保证每个膨胀圆的离散圆周都落入椭圆。
        scale = 1.0
        for local_x, local_y, radius in circles:
            for sample_index in range(32):
                angle = 2.0 * math.pi * sample_index / 32.0
                px = local_x + radius * math.cos(angle)
                py = local_y + radius * math.sin(angle)
                scale = max(scale, math.hypot(px / a, py / b))
        return cx, cy, a * scale, b * scale, yaw

    @staticmethod
    def ellipse_local_point(x, y, cx, cy, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x - cx, y - cy
        return dx * c + dy * s, -dx * s + dy * c

    def ellipse_tangent_targets(self, pose, ellipse):
        """把椭圆缩放为单位圆后计算左右切点。"""
        cx, cy, a, b, yaw = ellipse
        local_x, local_y = self.ellipse_local_point(
            pose.x, pose.y, cx, cy, yaw
        )
        px, py = local_x / a, local_y / b
        distance2 = px * px + py * py
        if distance2 <= 1.0 + 1e-9:
            return []
        root = math.sqrt(distance2 - 1.0)
        c, s = math.cos(yaw), math.sin(yaw)
        targets = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            unit_x = (px - sign * py * root) / distance2
            unit_y = (py + sign * px * root) / distance2
            tangent_x, tangent_y = a * unit_x, b * unit_y
            tx = cx + tangent_x * c - tangent_y * s
            ty = cy + tangent_x * s + tangent_y * c
            dx, dy = tx - pose.x, ty - pose.y
            norm = math.hypot(dx, dy)
            tx += self.tangent_extension * dx / norm
            ty += self.tangent_extension * dy / norm
            targets.append((side, tx, ty))
        return targets

    def blocking_obstacles(self, pose):
        """返回目标导向矩形中按前缘距离排序的阻挡圆。"""
        if self.goal_rectangle is None:
            return []
        dx = self.goal.x - pose.x
        dy = self.goal.y - pose.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return []
        ux, uy = dx / distance, dy / distance
        half_width = self.goal_rectangle.width / 2.0
        start_s = self.goal_rectangle.forward_offset
        end_s = start_s + self.goal_rectangle.length
        blocking = []
        for index, obstacle in enumerate(self.obstacles):
            rel_x = obstacle.x - pose.x
            rel_y = obstacle.y - pose.y
            longitudinal = rel_x * ux + rel_y * uy
            lateral = abs(-rel_x * uy + rel_y * ux)
            radius = self.inflated_radius(obstacle)
            if longitudinal + radius < start_s or longitudinal - radius > end_s:
                continue
            if lateral > half_width + radius:
                continue
            # 只有影响中心通行带或目标直连线的圆才触发切向重构。
            core_half_width = min(half_width, self.obstacle_inflation)
            if lateral <= core_half_width + radius:
                blocking.append((longitudinal - radius, index))
        blocking.sort(key=lambda item: item[0])
        return [index for _, index in blocking]

    def tangent_targets(self, pose, obstacle, radius_override=None):
        px, py = float(pose.x), float(pose.y)
        cx, cy = float(obstacle.x), float(obstacle.y)
        radius = (
            self.inflated_radius(obstacle)
            if radius_override is None else float(radius_override)
        )
        dx, dy = cx - px, cy - py
        distance = math.hypot(dx, dy)
        if distance <= radius + 1e-6:
            return []
        center_angle = math.atan2(dy, dx)
        # 从外部点观察，切线方向与“点→圆心”方向的夹角为 asin(R / D)。
        offset = math.asin(radius / distance)
        tangent_length = math.sqrt(max(0.0, distance * distance - radius * radius))
        targets = []
        for side, angle in (("left", center_angle + offset), ("right", center_angle - offset)):
            tx = px + tangent_length * math.cos(angle)
            ty = py + tangent_length * math.sin(angle)
            # 沿切线方向越过圆的侧面，避免矩形只指到擦边点。
            tangent_dx = tx - px
            tangent_dy = ty - py
            tangent_norm = math.hypot(tangent_dx, tangent_dy)
            tx += self.tangent_extension * tangent_dx / tangent_norm
            ty += self.tangent_extension * tangent_dy / tangent_norm
            targets.append((side, tx, ty))
        return targets

    def select_tangent_target(self, pose, obstacle_index):
        obstacle_indexes = (
            (obstacle_index,)
            if self.cluster_shape == "single_circle"
            else self.overlapping_cluster(obstacle_index)
        )
        ellipse = self.cluster_enclosing_ellipse(obstacle_indexes)
        best = None
        for side, tx, ty in self.ellipse_tangent_targets(pose, ellipse):
            if not self.inside_bounds(tx, ty):
                continue
            if not self.segment_is_clear(
                (pose.x, pose.y), (tx, ty), ignored_indexes=obstacle_indexes
            ):
                continue
            heading = math.atan2(ty - pose.y, tx - pose.x)
            turn_cost = abs(self.normalize_angle(heading - pose.yaw))
            cost = (
                math.hypot(tx - pose.x, ty - pose.y)
                + math.hypot(self.goal.x - tx, self.goal.y - ty)
                + self.turn_cost_weight * turn_cost
            )
            candidate = (
                cost, side, tx, ty, obstacle_indexes,
                *ellipse,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    def active_obstacle_passed(self, pose):
        guidance = self.tangent_guidance
        if not guidance.active:
            return True
        local_start = self.ellipse_local_point(
            pose.x, pose.y, guidance.cluster_x, guidance.cluster_y,
            guidance.ellipse_yaw,
        )
        local_goal = self.ellipse_local_point(
            self.goal.x, self.goal.y, guidance.cluster_x, guidance.cluster_y,
            guidance.ellipse_yaw,
        )
        distance = self.point_segment_distance(
            (0.0, 0.0),
            (local_start[0] / guidance.ellipse_a,
             local_start[1] / guidance.ellipse_b),
            (local_goal[0] / guidance.ellipse_a,
             local_goal[1] / guidance.ellipse_b),
        )
        return distance > 1.02

    def update_tangent_guidance(self, pose):
        if not self.use_tangent_guidance or not self.obstacles:
            self.restore_rectangle_size()
            self.blocking_obstacle_count = 0
            self.nearest_blocking_distance = math.inf
            self.sampling_target = (float(self.goal.x), float(self.goal.y))
            return

        if self.tangent_guidance.active and not self.active_obstacle_passed(pose):
            if not self.rectangle_shrunk:
                blocking = self.blocking_obstacles(pose)
                self.maybe_shrink_rectangle(pose, blocking)
            self.sampling_target = (
                self.tangent_guidance.target_x,
                self.tangent_guidance.target_y,
            )
            return

        self.tangent_guidance = TangentGuidance()
        self.shrink_decision_made = False
        self.restore_rectangle_size()
        blocking = self.blocking_obstacles(pose)
        self.maybe_shrink_rectangle(pose, blocking)
        if self.rectangle_shrunk:
            # 缩小后重新确认仍位于新矩形内的阻挡圆及其先后顺序。
            blocking = self.blocking_obstacles(pose)
        self.blocking_obstacle_count = len(blocking)
        if not blocking:
            self.sampling_target = (float(self.goal.x), float(self.goal.y))
            return

        # 多圆情况下，按沿目标方向最先遇到的圆逐个处理。
        for obstacle_index in blocking:
            selected = self.select_tangent_target(pose, obstacle_index)
            if selected is None:
                continue
            (
                _, side, tx, ty, obstacle_indexes,
                cluster_x, cluster_y, ellipse_a, ellipse_b, ellipse_yaw,
            ) = selected
            self.tangent_guidance = TangentGuidance(
                obstacle_index=obstacle_index,
                obstacle_indexes=obstacle_indexes,
                side=side,
                target_x=tx,
                target_y=ty,
                cluster_x=cluster_x,
                cluster_y=cluster_y,
                ellipse_a=ellipse_a,
                ellipse_b=ellipse_b,
                ellipse_yaw=ellipse_yaw,
            )
            self.sampling_target = (tx, ty)
            return

        self.sampling_target = (float(self.goal.x), float(self.goal.y))

    def _normalize_probabilities(self):
        """
        归一化非均匀采样概率。

        如果非均匀采样概率之和小于1，剩余概率分配给均匀采样。
        如果大于1，则按比例压缩。
        """
        values = [
            max(0.0, self.goal_probability),
            max(0.0, self.corridor_probability),
            max(0.0, self.rectangle_probability),
        ]

        total_guided = sum(values)

        if total_guided >= 1.0:
            scale = 0.95 / max(total_guided, 1e-12)

            self.goal_probability = values[0] * scale
            self.corridor_probability = values[1] * scale
            self.rectangle_probability = values[2] * scale
            self.uniform_probability = 0.05
        else:
            self.goal_probability = values[0]
            self.corridor_probability = values[1]
            self.rectangle_probability = values[2]
            self.uniform_probability = 1.0 - total_guided

    def update_probability(
        self,
        iteration,
        current_best_cost=None,
    ):
        """
        根据搜索进度自适应调整采样概率。

        current_best_cost:
            当前RRT*最优路径代价。如果代价下降，认为发生改善。
        """
        if not self.use_hybrid_sampling:
            return

        self.iteration = iteration

        if not self.adaptive_probabilities:
            return

        improved = False

        if current_best_cost is not None:
            if current_best_cost < self.best_cost - 1e-6:
                self.best_cost = current_best_cost
                improved = True

        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if iteration < 500:
            # 搜索初期：保持较强的全局探索能力。
            self.uniform_probability = 0.50
            self.goal_probability = 0.10
            self.corridor_probability = 0.10
            self.rectangle_probability = 0.30

        elif iteration < 2000:
            # 搜索中期：逐渐加强目标方向的利用。
            progress = (iteration - 500) / 1500.0

            self.uniform_probability = 0.50 - 0.25 * progress
            self.goal_probability = 0.10 + 0.05 * progress
            self.corridor_probability = 0.10 + 0.05 * progress
            self.rectangle_probability = (
                1.0
                - self.uniform_probability
                - self.goal_probability
                - self.corridor_probability
            )

        else:
            # 搜索后期：重点优化已有方向，但仍保留随机探索。
            self.uniform_probability = 0.20
            self.goal_probability = 0.15
            self.corridor_probability = 0.15
            self.rectangle_probability = 0.50

        # 长时间无改善时，暂时重新增加全局随机探索。
        if self.no_improvement_count > 300:
            exploration_boost = min(
                0.25,
                0.05
                + (self.no_improvement_count - 300) / 2000.0,
            )

            guided_total = (
                self.goal_probability
                + self.corridor_probability
                + self.rectangle_probability
            )

            if guided_total > 1e-12:
                scale = max(
                    0.0,
                    1.0
                    - exploration_boost / guided_total,
                )

                self.goal_probability *= scale
                self.corridor_probability *= scale
                self.rectangle_probability *= scale

            self.uniform_probability = (
                1.0
                - self.goal_probability
                - self.corridor_probability
                - self.rectangle_probability
            )

    def sample(self):
        if not self.use_hybrid_sampling:
            return self.uniform_sample()

        r = self.random.random()

        threshold = self.goal_probability

        if r < threshold:
            return self.goal_sample()

        threshold += self.rectangle_probability

        if r < threshold and self.goal_rectangle is not None:
            return self.goal_rectangle_sample()

        threshold += self.corridor_probability

        if r < threshold and self.corridors:
            return self.corridor_sample()

        return self.uniform_sample()

    def uniform_sample(self):
        return (
            self.random.uniform(self.x_min, self.x_max),
            self.random.uniform(self.y_min, self.y_max),
            self.random.uniform(-math.pi, math.pi),
        )

    def goal_sample(self):
        """
        直接采样目标状态。

        对Ackermann车辆而言，目标yaw也需要保留。
        """
        if self.tangent_guidance.active:
            target_x, target_y = self.sampling_target
            yaw = math.atan2(
                target_y - self.goal_rectangle.anchor_y,
                target_x - self.goal_rectangle.anchor_x,
            )
            return target_x, target_y, yaw

        return self.goal.x, self.goal.y, self.goal.yaw

    def goal_rectangle_sample(self):
        """
        在以当前RRT*末端节点为起点、朝向目标的矩形区域中采样。

        矩形局部坐标：
            local_x沿anchor到goal的方向；
            local_y垂直于目标方向。

        将局部坐标旋转并平移到世界坐标系。
        """
        rectangle = self.goal_rectangle

        target_x, target_y = self.sampling_target
        dx = target_x - rectangle.anchor_x
        dy = target_y - rectangle.anchor_y

        goal_distance = math.hypot(dx, dy)

        if goal_distance < 1e-9:
            return self.goal_sample()

        direction_yaw = math.atan2(dy, dx)

        cos_yaw = math.cos(direction_yaw)
        sin_yaw = math.sin(direction_yaw)

        for _ in range(self.max_sample_attempts):
            rectangle_start = rectangle.forward_offset
            rectangle_end = rectangle.forward_offset + rectangle.length

            if self.random.random() < self.near_anchor_probability:
                # 在末端节点前方的近距离区域重点采样。Beta(1, 3)
                # 在0附近密度最高，同时保留少量向前延伸的样本。
                near_start = max(0.0, rectangle_start)
                near_end = min(
                    rectangle_end,
                    near_start
                    + rectangle.length * self.near_anchor_length_ratio,
                )
                if near_end > near_start:
                    ratio = self.random.betavariate(1.0, 3.0)
                    local_x = near_start + ratio * (near_end - near_start)
                else:
                    local_x = self.random.uniform(
                        rectangle_start, rectangle_end
                    )
            else:
                # 保留完整矩形采样，保证全局探索和快速向前推进。
                local_x = self.random.uniform(
                    rectangle_start,
                    rectangle_end,
                )

            # 横向使用截断高斯分布。
            # 相比完全均匀分布，更偏向矩形中心线。
            half_width = rectangle.width / 2.0

            local_y = self.random.gauss(
                0.0,
                rectangle.width / 6.0,
            )

            local_y = max(
                -half_width,
                min(half_width, local_y),
            )

            world_x = (
                rectangle.anchor_x
                + local_x * cos_yaw
                - local_y * sin_yaw
            )

            world_y = (
                rectangle.anchor_y
                + local_x * sin_yaw
                + local_y * cos_yaw
            )

            if not self.inside_bounds(world_x, world_y):
                continue

            sample_to_goal_yaw = math.atan2(
                target_y - world_y,
                target_x - world_x,
            )

            sample_yaw = self.normalize_angle(
                sample_to_goal_yaw
                + self.random.gauss(0.0, self.heading_std)
            )

            return world_x, world_y, sample_yaw

        # 多次采样都越界时，退化为全地图均匀采样。
        return self.uniform_sample()

    def corridor_sample(self):
        corridor = self.random.choice(self.corridors)

        dx = corridor.x2 - corridor.x1
        dy = corridor.y2 - corridor.y1

        length = math.hypot(dx, dy)

        if length < 1e-9:
            return self.uniform_sample()

        corridor_yaw = math.atan2(dy, dx)

        ratio = self.random.random()

        center_x = corridor.x1 + ratio * dx
        center_y = corridor.y1 + ratio * dy

        half_width = corridor.width / 2.0

        offset = self.random.gauss(
            0.0,
            corridor.width / 6.0,
        )

        offset = max(
            -half_width,
            min(half_width, offset),
        )

        x = center_x - offset * math.sin(corridor_yaw)
        y = center_y + offset * math.cos(corridor_yaw)

        if not self.inside_bounds(x, y):
            return self.uniform_sample()

        yaw = self.normalize_angle(
            corridor_yaw
            + self.random.gauss(0.0, self.heading_std)
        )

        return x, y, yaw
