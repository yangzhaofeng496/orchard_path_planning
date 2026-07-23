import math
import random
import time
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
    """朝向目标的矩形采样区域。"""
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
        self, bounds, goal, corridors=None, goal_rectangle: Optional[GoalRectangle] = None,
        use_hybrid_sampling=True, goal_probability=0.10, tangent_probability=0.05,
        corridor_probability=0.15, rectangle_probability=0.30, random_seed=0,
        heading_std_degrees=12.0, max_sample_attempts=20, obstacles=None,
        use_tangent_guidance=True, obstacle_inflation=1.2, tangent_extension=0.3,
        turn_cost_weight=2.0, shrink_probability=0.60, shrink_length_factor=0.70,
        shrink_width_factor=0.65, min_rectangle_length=10.0,
        min_rectangle_width=8.0, shrink_activation_distance=20.0,
        near_anchor_probability=0.60, near_anchor_length_ratio=0.40,
        adaptive_probabilities=False, cluster_shape="ellipse",
        tangent_target_tolerance=1.0, tangent_along_std=0.30,
        tangent_lateral_std=0.20, max_guidance_updates=20,
        max_cluster_obstacles=8, max_cluster_span=14.0,
        tangent_distance_threshold=12.0, single_cluster_tangent_scale=0.30,
        multi_cluster_tangent_scale=1.00, tangent_detour_weight=20.0,
        max_tangent_detour_ratio=1.15, remaining_blocker_weight=2.0,
        side_switch_penalty=3.0, min_uniform_probability=0.20,
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = map(float, bounds)
        self.goal = goal
        self.corridors = list(corridors) if corridors else []
        self.goal_rectangle = goal_rectangle
        self.obstacles = list(obstacles) if obstacles else []

        self.use_hybrid_sampling = bool(use_hybrid_sampling)
        self.use_tangent_guidance = bool(use_tangent_guidance)
        self.obstacle_inflation = max(0.0, float(obstacle_inflation))
        self.tangent_extension = max(0.0, min(2.0, float(tangent_extension)))
        self.turn_cost_weight = max(0.0, float(turn_cost_weight))

        self.tangent_distance_threshold = max(0.0, float(tangent_distance_threshold))
        self.single_cluster_tangent_scale = max(0.0, min(1.0, float(single_cluster_tangent_scale)))
        self.multi_cluster_tangent_scale = max(0.0, min(1.0, float(multi_cluster_tangent_scale)))
        self.tangent_detour_weight = max(0.0, float(tangent_detour_weight))
        self.remaining_blocker_weight = max(0.0, float(remaining_blocker_weight))
        self.side_switch_penalty = max(0.0, float(side_switch_penalty))
        self.min_uniform_probability = max(0.0, min(0.95, float(min_uniform_probability)))

        # 新增：根据簇数量设置最大绕行比
        self.max_detour_single_cluster = 1.10
        self.max_detour_multi_cluster = 1.20

        # 新增：扩展候选数量
        self.extension_candidates = (0.0, 0.2, 0.4, 0.6)

        self.shrink_probability = self._clamp01(shrink_probability)
        self.shrink_length_factor = max(0.05, min(1.0, float(shrink_length_factor)))
        self.shrink_width_factor = max(0.05, min(1.0, float(shrink_width_factor)))
        self.min_rectangle_length = max(0.1, float(min_rectangle_length))
        self.min_rectangle_width = max(0.1, float(min_rectangle_width))
        self.shrink_activation_distance = max(0.0, float(shrink_activation_distance))

        self.near_anchor_probability = self._clamp01(near_anchor_probability)
        self.near_anchor_length_ratio = max(0.05, min(1.0, float(near_anchor_length_ratio)))

        self.adaptive_probabilities = bool(adaptive_probabilities)
        self.cluster_shape = str(cluster_shape)

        self.tangent_target_tolerance = max(0.05, float(tangent_target_tolerance))
        self.tangent_along_std = max(0.0, float(tangent_along_std))
        self.tangent_lateral_std = max(0.0, float(tangent_lateral_std))
        self.max_guidance_updates = max(1, int(max_guidance_updates))
        self.max_cluster_obstacles = max(1, int(max_cluster_obstacles))
        self.max_cluster_span = max(0.1, float(max_cluster_span))

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
        self.guidance_update_count = 0

        self.initial_goal_probability = max(0.0, float(goal_probability))
        self.initial_tangent_probability = max(0.0, float(tangent_probability))
        self.initial_corridor_probability = max(0.0, float(corridor_probability))
        self.initial_rectangle_probability = max(0.0, float(rectangle_probability))

        self.goal_probability = self.initial_goal_probability
        self.tangent_probability = self.initial_tangent_probability
        self.corridor_probability = self.initial_corridor_probability
        self.rectangle_probability = self.initial_rectangle_probability
        self.uniform_probability = 0.0
        self._normalize_probabilities(min_uniform=0.05)

        self.random = random.Random(random_seed)
        self.heading_std = math.radians(float(heading_std_degrees))
        self.max_sample_attempts = max(1, int(max_sample_attempts))

        self.iteration = 0
        self.no_improvement_count = 0
        self.best_cost = math.inf
        self.has_feasible_path = False

        self.current_anchor_pose = None
        self.direct_blocking_indexes = []
        self.direct_blocking_cluster_count = 0
        self.previous_guidance_cluster = ()
        self.previous_guidance_side = None
        # 保留切向引导的完整调整历史，供实验可视化和调试使用。
        self.tangent_guidance_history = []
        self.tangent_sample_history = []

        self._cluster_cache = {}
        self._ellipse_cache = {}

        self.stats = {
            "uniform": 0,
            "goal": 0,
            "tangent": 0,
            "rectangle": 0,
            "corridor": 0,
            "tangent_fallback": 0,
            "tangent_sample_attempts": 0,
            "tangent_sample_success": 0,
            "guidance_created": 0,
            "guidance_reused": 0,
            "guidance_released": 0,
            "guidance_expired": 0,
            "cluster_builds": 0,
            "ellipse_builds": 0,
            "max_cluster_size": 0,
            "tangent_geometry_time": 0.0,
            "direct_unblocked_updates": 0,
            "single_cluster_updates": 0,
            "multi_cluster_updates": 0,
            "tangent_disabled_after_solution": 0,
            "tangent_disabled_by_distance": 0,
            "tangent_disabled_unblocked": 0,
            "tangent_side_switches": 0,
            "tangent_candidate_rejected_detour": 0,
            "tangent_candidate_passed_cluster": 0,
            "post_solution_samples": 0,
            "effective_tangent_probability_sum": 0.0,
            "tangent_candidate_total": 0,
            "tangent_candidate_valid": 0,
            "tangent_candidate_over_detour_limit": 0,
        }

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0.0 if key == "tangent_geometry_time" else 0

    def get_stats(self):
        result = dict(self.stats)
        result.update({
            "goal_probability": self.goal_probability,
            "tangent_probability": self.tangent_probability,
            "rectangle_probability": self.rectangle_probability,
            "corridor_probability": self.corridor_probability,
            "uniform_probability": self.uniform_probability,
            "guidance_active": self.tangent_guidance.active,
            "guidance_updates": self.guidance_update_count,
            "blocking_obstacle_count": self.blocking_obstacle_count,
            "nearest_blocking_distance": self.nearest_blocking_distance,
            "has_feasible_path": self.has_feasible_path,
            "direct_blocking_count": len(self.direct_blocking_indexes),
            "direct_blocking_cluster_count": self.direct_blocking_cluster_count,
            "previous_guidance_side": self.previous_guidance_side,
            "previous_guidance_cluster": self.previous_guidance_cluster,
        })

        # 计算当前有效概率
        effective_probs = self.effective_sampling_probabilities()
        result.update({
            "effective_goal_probability": effective_probs["goal"],
            "effective_tangent_probability": effective_probs["tangent"],
            "effective_uniform_probability": effective_probs["uniform"],
        })

        return result

    def set_feasible_path_found(self, found=True):
        """设置是否找到可行路径。"""
        found = bool(found)
        if found and not self.has_feasible_path:
            self.has_feasible_path = True
            self.stats["tangent_disabled_after_solution"] += 1
            self.clear_tangent_guidance()
            self.restore_rectangle_size()
            self.blocking_obstacle_count = 0
            self.nearest_blocking_distance = math.inf
            self.sampling_target = (float(self.goal.x), float(self.goal.y))
            self.direct_blocking_indexes = []
            self.direct_blocking_cluster_count = 0
            self.previous_guidance_cluster = ()
            self.previous_guidance_side = None
        elif not found:
            self.has_feasible_path = False
            self.previous_guidance_cluster = ()
            self.previous_guidance_side = None

    def inside_bounds(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def point_is_free(self, x, y, ignored_indexes=()):
        ignored = set(ignored_indexes)

        for index, obstacle in enumerate(self.obstacles):
            if index in ignored:
                continue

            if math.hypot(x - obstacle.x, y - obstacle.y) <= self.inflated_radius(obstacle):
                return False

        return True

    def update_rectangle_anchor(self, pose):
        """移动矩形锚点并更新切向引导。"""
        if not hasattr(self, '_anchor_update_logged'):
            self._anchor_update_logged = True
            print(f"[RectangleAnchor] 首次调用 update_rectangle_anchor")
            print(f"  goal_rectangle={self.goal_rectangle is not None}")

        if self.goal_rectangle is not None:
            self.goal_rectangle.anchor_x = float(pose.x)
            self.goal_rectangle.anchor_y = float(pose.y)

        self.update_tangent_guidance(pose)

    def inflated_radius(self, obstacle):
        return max(1e-6, float(obstacle.radius) + self.obstacle_inflation)

    def restore_rectangle_size(self):
        if self.goal_rectangle is None:
            return

        self.goal_rectangle.length = self.base_rectangle_length
        self.goal_rectangle.width = self.base_rectangle_width
        self.rectangle_shrunk = False

    def obstacle_edge_distance(self, pose, obstacle):
        center_distance = math.hypot(obstacle.x - pose.x, obstacle.y - pose.y)
        return max(0.0, center_distance - self.inflated_radius(obstacle))

    def maybe_shrink_rectangle(self, pose, obstacle_indexes):
        obstacle_count = len(obstacle_indexes)
        self.blocking_obstacle_count = obstacle_count

        # 计算局部的最近阻挡距离，不覆盖 self.nearest_blocking_distance
        nearest_distance = min(
            (
                self.obstacle_edge_distance(pose, self.obstacles[index])
                for index in obstacle_indexes
            ),
            default=math.inf,
        )

        if self.goal_rectangle is None or obstacle_count <= 2:
            return False

        if nearest_distance > self.shrink_activation_distance:
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

        ratio = ((px - ax) * dx + (py - ay) * dy) / length2
        ratio = max(0.0, min(1.0, ratio))
        closest_x = ax + ratio * dx
        closest_y = ay + ratio * dy

        return math.hypot(px - closest_x, py - closest_y)

    def segment_is_clear(self, start, end, ignored_index=None, ignored_indexes=()):
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
        """构建局部重叠簇，并限制数量和空间跨度。"""
        if seed_index in self._cluster_cache:
            return self._cluster_cache[seed_index]

        self.stats["cluster_builds"] += 1
        seed = self.obstacles[seed_index]
        candidates = []

        for index, obstacle in enumerate(self.obstacles):
            distance = math.hypot(
                obstacle.x - seed.x,
                obstacle.y - seed.y,
            )

            if distance <= self.max_cluster_span:
                candidates.append((distance, index))

        candidates.sort(key=lambda item: item[0])
        cluster = {seed_index}
        pending = [seed_index]

        while pending and len(cluster) < self.max_cluster_obstacles:
            current = pending.pop(0)
            obstacle_a = self.obstacles[current]

            for _, index in candidates:
                if index in cluster:
                    continue

                obstacle_b = self.obstacles[index]
                center_distance = math.hypot(
                    obstacle_a.x - obstacle_b.x,
                    obstacle_a.y - obstacle_b.y,
                )

                if center_distance <= (
                    self.inflated_radius(obstacle_a)
                    + self.inflated_radius(obstacle_b)
                ):
                    cluster.add(index)
                    pending.append(index)

                    if len(cluster) >= self.max_cluster_obstacles:
                        break

        result = tuple(sorted(cluster))
        self._cluster_cache[seed_index] = result
        self.stats["max_cluster_size"] = max(
            self.stats["max_cluster_size"], len(result)
        )

        return result

    def cluster_enclosing_ellipse(self, obstacle_indexes):
        """构造包含障碍簇所有膨胀圆的定向椭圆。"""
        key = tuple(obstacle_indexes)

        if key in self._ellipse_cache:
            return self._ellipse_cache[key]

        self.stats["ellipse_builds"] += 1

        weights = [
            self.inflated_radius(self.obstacles[index])
            for index in obstacle_indexes
        ]

        total = sum(weights)

        if total <= 1e-12:
            return None

        cx = sum(
            self.obstacles[index].x * weight
            for index, weight in zip(obstacle_indexes, weights)
        ) / total

        cy = sum(
            self.obstacles[index].y * weight
            for index, weight in zip(obstacle_indexes, weights)
        ) / total

        cov_xx = 0.0
        cov_xy = 0.0
        cov_yy = 0.0

        for index, weight in zip(obstacle_indexes, weights):
            dx = self.obstacles[index].x - cx
            dy = self.obstacles[index].y - cy
            cov_xx += weight * dx * dx
            cov_xy += weight * dx * dy
            cov_yy += weight * dy * dy

        yaw = 0.5 * math.atan2(
            2.0 * cov_xy,
            cov_xx - cov_yy,
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        circles = []
        axis_a = 1e-6
        axis_b = 1e-6

        for index in obstacle_indexes:
            obstacle = self.obstacles[index]
            dx = obstacle.x - cx
            dy = obstacle.y - cy

            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            radius = self.inflated_radius(obstacle)

            circles.append((local_x, local_y, radius))
            axis_a = max(axis_a, abs(local_x) + radius)
            axis_b = max(axis_b, abs(local_y) + radius)

        scale = 1.0

        for local_x, local_y, radius in circles:
            for sample_index in range(24):
                angle = 2.0 * math.pi * sample_index / 24.0
                px = local_x + radius * math.cos(angle)
                py = local_y + radius * math.sin(angle)
                scale = max(
                    scale,
                    math.hypot(px / axis_a, py / axis_b),
                )

        ellipse = (
            cx,
            cy,
            axis_a * scale,
            axis_b * scale,
            yaw,
        )

        if not all(math.isfinite(value) for value in ellipse):
            return None

        self._ellipse_cache[key] = ellipse
        return ellipse

    @staticmethod
    def ellipse_local_point(x, y, cx, cy, yaw):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx = x - cx
        dy = y - cy

        return (
            dx * cos_yaw + dy * sin_yaw,
            -dx * sin_yaw + dy * cos_yaw,
        )

    def ellipse_tangent_targets(self, pose, ellipse):
        if ellipse is None:
            return []

        cx, cy, axis_a, axis_b, yaw = ellipse

        if axis_a <= 1e-6 or axis_b <= 1e-6:
            return []

        local_x, local_y = self.ellipse_local_point(
            pose.x, pose.y, cx, cy, yaw
        )

        px = local_x / axis_a
        py = local_y / axis_b
        distance2 = px * px + py * py

        if not math.isfinite(distance2):
            return []

        if distance2 <= 1.0 + 1e-9:
            return []

        root = math.sqrt(max(0.0, distance2 - 1.0))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        targets = []

        for side, sign in (("left", 1.0), ("right", -1.0)):
            unit_x = (px - sign * py * root) / distance2
            unit_y = (py + sign * px * root) / distance2

            tangent_x = axis_a * unit_x
            tangent_y = axis_b * unit_y

            tx = cx + tangent_x * cos_yaw - tangent_y * sin_yaw
            ty = cy + tangent_x * sin_yaw + tangent_y * cos_yaw

            dx = tx - pose.x
            dy = ty - pose.y
            norm = math.hypot(dx, dy)

            if norm <= 1e-9 or not math.isfinite(norm):
                continue

            tx += self.tangent_extension * dx / norm
            ty += self.tangent_extension * dy / norm

            if math.isfinite(tx) and math.isfinite(ty):
                targets.append((side, tx, ty))

        return targets

    def blocking_obstacles(self, pose):
        """返回目标导向矩形中的阻挡圆。"""
        if self.goal_rectangle is None:
            return []

        dx = self.goal.x - pose.x
        dy = self.goal.y - pose.y
        distance = math.hypot(dx, dy)

        if distance <= 1e-9:
            return []

        ux = dx / distance
        uy = dy / distance

        half_width = self.goal_rectangle.width / 2.0
        start_s = self.goal_rectangle.forward_offset
        end_s = start_s + self.goal_rectangle.length

        core_half_width = min(
            half_width,
            self.obstacle_inflation,
        )

        blocking = []

        for index, obstacle in enumerate(self.obstacles):
            rel_x = obstacle.x - pose.x
            rel_y = obstacle.y - pose.y

            longitudinal = rel_x * ux + rel_y * uy
            lateral = abs(-rel_x * uy + rel_y * ux)
            radius = self.inflated_radius(obstacle)

            if longitudinal + radius < start_s:
                continue

            if longitudinal - radius > end_s:
                continue

            if lateral > half_width + radius:
                continue

            if lateral <= core_half_width + radius:
                blocking.append((longitudinal - radius, index))

        blocking.sort(key=lambda item: item[0])
        return [index for _, index in blocking]

    def direct_blocking_obstacles(self, pose):
        """返回当前pose到最终goal的直线路径上的阻挡障碍物。"""
        dx = self.goal.x - pose.x
        dy = self.goal.y - pose.y
        distance = math.hypot(dx, dy)

        if distance <= 1e-9:
            return []

        blocking = []

        for index, obstacle in enumerate(self.obstacles):
            segment_distance = self.point_segment_distance(
                (obstacle.x, obstacle.y),
                (pose.x, pose.y),
                (self.goal.x, self.goal.y)
            )

            radius = self.inflated_radius(obstacle)

            if segment_distance <= radius:
                # 计算障碍物在pose到goal方向上的投影距离
                rel_x = obstacle.x - pose.x
                rel_y = obstacle.y - pose.y
                projection = (rel_x * dx + rel_y * dy) / distance

                # 确保障碍物在有限线段范围内
                if 0 <= projection <= distance:
                    # 计算前缘距离（障碍物最近点到pose的距离）
                    edge_distance = max(0.0, projection - radius)
                    blocking.append((edge_distance, index))

        blocking.sort(key=lambda item: item[0])
        result = [index for _, index in blocking]

        # 调试日志：只在第一次检测到阻挡时打印
        if result and not hasattr(self, '_first_blocking_logged'):
            self._first_blocking_logged = True
            print(f"[DirectBlocking] 首次检测到 {len(result)} 个阻挡障碍物: {result}")

        return result

    def count_direct_blocking_clusters(self, blocking_indexes):
        """统计直线阻挡障碍物的簇数量。"""
        if not blocking_indexes:
            return 0

        cluster_keys = set()
        for obstacle_index in blocking_indexes:
            if self.cluster_shape == "single_circle":
                cluster_key = (obstacle_index,)
            else:
                cluster_key = tuple(self.overlapping_cluster(obstacle_index))
            cluster_keys.add(cluster_key)

        return len(cluster_keys)

    def segment_intersects_ellipse(self, start, end, ellipse, margin=1.02):
        """判断线段是否与椭圆相交。"""
        if ellipse is None:
            return False

        cx, cy, axis_a, axis_b, yaw = ellipse

        if axis_a <= 1e-6 or axis_b <= 1e-6:
            return False

        # 将起点和终点转换到椭圆局部坐标
        local_start = self.ellipse_local_point(start[0], start[1], cx, cy, yaw)
        local_end = self.ellipse_local_point(end[0], end[1], cx, cy, yaw)

        # 映射到单位圆
        unit_start = (local_start[0] / axis_a, local_start[1] / axis_b)
        unit_end = (local_end[0] / axis_a, local_end[1] / axis_b)

        # 计算单位圆原点到线段的距离
        distance = self.point_segment_distance(
            (0.0, 0.0),
            unit_start,
            unit_end
        )

        return distance <= margin
        """单圆切向目标接口。"""
        px = float(pose.x)
        py = float(pose.y)
        cx = float(obstacle.x)
        cy = float(obstacle.y)

        radius = (
            self.inflated_radius(obstacle)
            if radius_override is None
            else max(1e-6, float(radius_override))
        )

        dx = cx - px
        dy = cy - py
        distance = math.hypot(dx, dy)

        if distance <= radius + 1e-6:
            return []

        ratio = max(-1.0, min(1.0, radius / distance))
        center_angle = math.atan2(dy, dx)
        offset = math.asin(ratio)

        tangent_length = math.sqrt(
            max(0.0, distance * distance - radius * radius)
        )

        targets = []

        for side, angle in (
            ("left", center_angle + offset),
            ("right", center_angle - offset),
        ):
            tx = px + tangent_length * math.cos(angle)
            ty = py + tangent_length * math.sin(angle)

            tangent_dx = tx - px
            tangent_dy = ty - py
            tangent_norm = math.hypot(tangent_dx, tangent_dy)

            if tangent_norm <= 1e-9:
                continue

            tx += self.tangent_extension * tangent_dx / tangent_norm
            ty += self.tangent_extension * tangent_dy / tangent_norm

            if math.isfinite(tx) and math.isfinite(ty):
                targets.append((side, tx, ty))

        return targets

    def select_tangent_target_from_cluster(
        self,
        pose,
        obstacle_index,
        obstacle_indexes,
    ):
        """选择最优切向目标，使用二维几何绕行比评价。"""
        start_time = time.perf_counter()
        ellipse = self.cluster_enclosing_ellipse(obstacle_indexes)

        # 计算直线距离用于绕行比
        direct_distance = math.hypot(self.goal.x - pose.x, self.goal.y - pose.y)

        # 根据簇数量确定最大绕行比
        cluster_count = len(obstacle_indexes)
        max_detour_ratio = (
            self.max_detour_single_cluster if cluster_count == 1
            else self.max_detour_multi_cluster
        )

        all_candidates = []

        # 对每侧生成多个延伸距离的候选
        for side, base_tx, base_ty in self.ellipse_tangent_targets(pose, ellipse):
            # 计算从base_tx, base_ty到pose的方向向量
            dx = base_tx - pose.x
            dy = base_ty - pose.y
            base_norm = math.hypot(dx, dy)

            if base_norm <= 1e-9 or not math.isfinite(base_norm):
                continue

            ux = dx / base_norm
            uy = dy / base_norm

            # 测试不同的延伸比例
            for extension_ratio in self.extension_candidates:
                self.stats["tangent_candidate_total"] += 1

                # 计算延伸后的候选点
                extension_distance = self.tangent_extension * extension_ratio
                tx = base_tx + extension_distance * ux
                ty = base_ty + extension_distance * uy

                # 基本有效性检查
                if not self.inside_bounds(tx, ty):
                    continue

                if not self.point_is_free(tx, ty):
                    continue

                if not math.isfinite(tx) or not math.isfinite(ty):
                    continue

                # 检查pose到候选点的连线是否被其他障碍物阻挡
                if not self.segment_is_clear(
                    (pose.x, pose.y),
                    (tx, ty),
                    ignored_indexes=obstacle_indexes,
                ):
                    continue

                self.stats["tangent_candidate_valid"] += 1

                # 计算二维几何绕行比
                pose_to_target = math.hypot(tx - pose.x, ty - pose.y)
                target_to_goal = math.hypot(self.goal.x - tx, self.goal.y - ty)
                candidate_distance = pose_to_target + target_to_goal
                detour_ratio = candidate_distance / max(direct_distance, 1e-6)

                # 检查候选点到goal是否已经不再穿过当前障碍簇
                passes_cluster = not self.segment_intersects_ellipse(
                    (tx, ty),
                    (self.goal.x, self.goal.y),
                    ellipse,
                    margin=1.02
                )

                if passes_cluster:
                    self.stats["tangent_candidate_passed_cluster"] += 1

                # 计算剩余阻挡障碍物数量
                remaining_blockers = []
                for idx in self.direct_blocking_indexes:
                    if idx not in obstacle_indexes:
                        segment_distance = self.point_segment_distance(
                            (self.obstacles[idx].x, self.obstacles[idx].y),
                            (tx, ty),
                            (self.goal.x, self.goal.y)
                        )
                        if segment_distance <= self.inflated_radius(self.obstacles[idx]):
                            remaining_blockers.append(idx)

                # 计算候选代价（纯二维几何，不考虑yaw和转向）
                cost = (
                    candidate_distance
                    + self.tangent_detour_weight * max(0.0, detour_ratio - 1.0)
                    + self.remaining_blocker_weight * len(remaining_blockers)
                )

                # 侧面切换惩罚
                cluster_key = tuple(obstacle_indexes)
                if (cluster_key == self.previous_guidance_cluster and
                    self.previous_guidance_side is not None and
                    side != self.previous_guidance_side):
                    cost += self.side_switch_penalty

                # 越过椭圆的候选优先（降低代价）
                if passes_cluster:
                    cost -= 5.0

                all_candidates.append((
                    cost,
                    detour_ratio,
                    passes_cluster,
                    side,
                    tx,
                    ty,
                    obstacle_indexes,
                    *ellipse,
                    len(remaining_blockers),
                ))

        self.stats["tangent_geometry_time"] += (
            time.perf_counter() - start_time
        )

        if not all_candidates:
            return None

        # 优先从越过椭圆的候选中选择
        passed_candidates = [c for c in all_candidates if c[2]]

        if passed_candidates:
            # 从越过椭圆的候选中选择绕行比最小的
            passed_candidates.sort(key=lambda c: c[1])

            # 如果最优候选的绕行比在限制内，直接选择
            if passed_candidates[0][1] <= max_detour_ratio:
                best = passed_candidates[0]
            else:
                # 都超过限制，选择绕行比最小的
                self.stats["tangent_candidate_over_detour_limit"] += 1
                best = passed_candidates[0]
        else:
            # 没有越过椭圆的候选，从所有候选中选择
            # 优先选择不超过绕行比限制的
            valid_detour_candidates = [c for c in all_candidates if c[1] <= max_detour_ratio]

            if valid_detour_candidates:
                # 从绕行比合格的候选中选择代价最小的
                valid_detour_candidates.sort(key=lambda c: c[0])
                best = valid_detour_candidates[0]
            else:
                # 所有候选都超过绕行比限制，拒绝所有候选并记录
                self.stats["tangent_candidate_rejected_detour"] += 1
                return None

        # 返回格式：(cost, side, tx, ty, obstacle_indexes, cx, cy, a, b, yaw, remaining_count)
        return (
            best[0],      # cost
            best[3],      # side
            best[4],      # tx
            best[5],      # ty
            best[6],      # obstacle_indexes
            best[7],      # cluster_x
            best[8],      # cluster_y
            best[9],      # ellipse_a
            best[10],     # ellipse_b
            best[11],     # ellipse_yaw
            best[2],      # passes_cluster
            best[12],     # remaining_blockers
        )

    def select_tangent_target(self, pose, obstacle_index):
        obstacle_indexes = (
            (obstacle_index,)
            if self.cluster_shape == "single_circle"
            else self.overlapping_cluster(obstacle_index)
        )

        return self.select_tangent_target_from_cluster(
            pose,
            obstacle_index,
            obstacle_indexes,
        )

    def active_obstacle_passed(self, pose):
        guidance = self.tangent_guidance

        if not guidance.active:
            return True

        if guidance.target_x is None or guidance.target_y is None:
            return True

        # 检查目标是否越界或进入障碍物
        if not self.inside_bounds(guidance.target_x, guidance.target_y):
            return True

        if not self.point_is_free(guidance.target_x, guidance.target_y):
            return True

        # 检查目标是否在车辆后方
        target_heading = math.atan2(
            guidance.target_y - pose.y,
            guidance.target_x - pose.x
        )
        heading_error = abs(self.normalize_angle(target_heading - pose.yaw))
        if heading_error > math.radians(120):
            return True

        target_distance = math.hypot(
            pose.x - guidance.target_x,
            pose.y - guidance.target_y,
        )

        if target_distance <= self.tangent_target_tolerance:
            return True

        required_values = (
            guidance.cluster_x,
            guidance.cluster_y,
            guidance.ellipse_a,
            guidance.ellipse_b,
            guidance.ellipse_yaw,
        )

        if any(value is None for value in required_values):
            return True

        if guidance.ellipse_a <= 1e-6 or guidance.ellipse_b <= 1e-6:
            return True

        local_start = self.ellipse_local_point(
            pose.x,
            pose.y,
            guidance.cluster_x,
            guidance.cluster_y,
            guidance.ellipse_yaw,
        )

        local_goal = self.ellipse_local_point(
            self.goal.x,
            self.goal.y,
            guidance.cluster_x,
            guidance.cluster_y,
            guidance.ellipse_yaw,
        )

        distance = self.point_segment_distance(
            (0.0, 0.0),
            (
                local_start[0] / guidance.ellipse_a,
                local_start[1] / guidance.ellipse_b,
            ),
            (
                local_goal[0] / guidance.ellipse_a,
                local_goal[1] / guidance.ellipse_b,
            ),
        )

        return distance > 1.02

    def clear_tangent_guidance(self, expired=False):
        if self.tangent_guidance.active:
            stat_name = (
                "guidance_expired"
                if expired
                else "guidance_released"
            )
            self.stats[stat_name] += 1

        self.tangent_guidance = TangentGuidance()
        self.guidance_update_count = 0
        self.sampling_target = (
            float(self.goal.x),
            float(self.goal.y),
        )
        # 清除切向引导时，允许下次选择新的侧面
        if not self.has_feasible_path:
            self.previous_guidance_cluster = ()
            self.previous_guidance_side = None

    def update_tangent_guidance(self, pose):
        # 保存当前锚点pose
        self.current_anchor_pose = pose

        # 调试日志：只在第一次调用时打印
        if not hasattr(self, '_first_update_logged'):
            self._first_update_logged = True
            print(f"[TangentGuidance] 首次调用 update_tangent_guidance:")
            print(f"  use_tangent_guidance={self.use_tangent_guidance}")
            print(f"  obstacles={len(self.obstacles) if self.obstacles else 0}")
            print(f"  has_feasible_path={self.has_feasible_path}")

        # 如果不使用切向引导、无障碍物或已找到可行路径，清除引导
        if (
            not self.use_tangent_guidance
            or self.tangent_probability <= 0.0
            or not self.obstacles
            or self.has_feasible_path
        ):
            self.clear_tangent_guidance()
            self.restore_rectangle_size()
            self.blocking_obstacle_count = 0
            self.nearest_blocking_distance = math.inf
            self.direct_blocking_indexes = []
            self.direct_blocking_cluster_count = 0
            return

        # 计算直线阻挡障碍物
        self.direct_blocking_indexes = self.direct_blocking_obstacles(pose)
        self.direct_blocking_cluster_count = self.count_direct_blocking_clusters(
            self.direct_blocking_indexes
        )

        # 更新最近阻挡距离
        if self.direct_blocking_indexes:
            self.nearest_blocking_distance = min(
                self.obstacle_edge_distance(pose, self.obstacles[index])
                for index in self.direct_blocking_indexes
            )
        else:
            self.nearest_blocking_distance = math.inf

        # 如果无遮挡，清除引导
        if not self.direct_blocking_indexes:
            self.stats["direct_unblocked_updates"] += 1
            self.clear_tangent_guidance()
            self.sampling_target = (float(self.goal.x), float(self.goal.y))
            self.restore_rectangle_size()
            self.blocking_obstacle_count = 0
            return

        # 统计簇数量
        if self.direct_blocking_cluster_count == 1:
            self.stats["single_cluster_updates"] += 1
        elif self.direct_blocking_cluster_count >= 2:
            self.stats["multi_cluster_updates"] += 1

        # 检查已有引导是否仍然有效
        if self.tangent_guidance.active:
            self.guidance_update_count += 1

            expired = (
                self.guidance_update_count > self.max_guidance_updates
            )

            passed = self.active_obstacle_passed(pose)

            # 检查是否到达切向目标附近
            if self.tangent_guidance.target_x is not None and self.tangent_guidance.target_y is not None:
                target_distance = math.hypot(
                    pose.x - self.tangent_guidance.target_x,
                    pose.y - self.tangent_guidance.target_y,
                )
                reached_target = (
                    target_distance <= self.tangent_target_tolerance
                )
            else:
                reached_target = False

            if not expired and not passed and not reached_target:
                self.stats["guidance_reused"] += 1

                # 更新矩形缩放（基于矩形阻挡，而非直线阻挡）
                if not self.rectangle_shrunk:
                    blocking = self.blocking_obstacles(pose)
                    self.maybe_shrink_rectangle(pose, blocking)

                self.sampling_target = (
                    self.tangent_guidance.target_x,
                    self.tangent_guidance.target_y,
                )
                return

            self.clear_tangent_guidance(expired=expired)

        # 重置矩形缩放状态
        self.shrink_decision_made = False
        self.restore_rectangle_size()

        # 基于矩形阻挡更新矩形大小
        blocking = self.blocking_obstacles(pose)
        self.maybe_shrink_rectangle(pose, blocking)

        if self.rectangle_shrunk:
            blocking = self.blocking_obstacles(pose)

        self.blocking_obstacle_count = len(blocking)

        # 从直线阻挡障碍物中选择切向目标
        visited_clusters = set()

        for obstacle_index in self.direct_blocking_indexes:
            obstacle_indexes = (
                (obstacle_index,)
                if self.cluster_shape == "single_circle"
                else self.overlapping_cluster(obstacle_index)
            )

            cluster_key = tuple(obstacle_indexes)

            if cluster_key in visited_clusters:
                continue

            visited_clusters.add(cluster_key)

            selected = self.select_tangent_target_from_cluster(
                pose,
                obstacle_index,
                obstacle_indexes,
            )

            if selected is None:
                continue

            (
                _,
                side,
                tx,
                ty,
                selected_indexes,
                cluster_x,
                cluster_y,
                ellipse_a,
                ellipse_b,
                ellipse_yaw,
                passes_cluster,
                remaining_blockers,
            ) = selected

            self.tangent_guidance = TangentGuidance(
                obstacle_index=obstacle_index,
                obstacle_indexes=selected_indexes,
                side=side,
                target_x=tx,
                target_y=ty,
                cluster_x=cluster_x,
                cluster_y=cluster_y,
                ellipse_a=ellipse_a,
                ellipse_b=ellipse_b,
                ellipse_yaw=ellipse_yaw,
            )

            # 调试日志：切向引导创建成功
            print(f"[TangentGuidance] ✓ 创建切向引导 - 侧面:{side}, 目标:({tx:.2f}, {ty:.2f}), 簇:{len(selected_indexes)}个障碍物")

            self.guidance_update_count = 0
            self.sampling_target = (tx, ty)
            self.stats["guidance_created"] += 1
            self.tangent_guidance_history.append({
                "iteration": int(self.iteration),
                "anchor_x": float(pose.x),
                "anchor_y": float(pose.y),
                "target_x": float(tx),
                "target_y": float(ty),
                "side": str(side),
                "obstacle_indexes": tuple(int(i) for i in selected_indexes),
                "cluster_x": float(cluster_x),
                "cluster_y": float(cluster_y),
                "ellipse_a": float(ellipse_a),
                "ellipse_b": float(ellipse_b),
                "ellipse_yaw": float(ellipse_yaw),
            })

            # 记录侧面切换
            if (cluster_key == self.previous_guidance_cluster and
                self.previous_guidance_side is not None and
                side != self.previous_guidance_side):
                self.stats["tangent_side_switches"] += 1

            # 更新上次引导状态
            self.previous_guidance_cluster = cluster_key
            self.previous_guidance_side = side

            return

        # 无法找到合适的切向目标
        self.sampling_target = (
            float(self.goal.x),
            float(self.goal.y),
        )

    def _normalize_probabilities(self, min_uniform=0.05):
        min_uniform = max(
            0.0,
            min(0.95, float(min_uniform)),
        )

        values = [
            max(0.0, self.goal_probability),
            max(0.0, self.tangent_probability),
            max(0.0, self.corridor_probability),
            max(0.0, self.rectangle_probability),
        ]

        total_guided = sum(values)
        guided_budget = 1.0 - min_uniform

        if total_guided > guided_budget and total_guided > 1e-12:
            scale = guided_budget / total_guided
            values = [
                value * scale
                for value in values
            ]

        self.goal_probability = values[0]
        self.tangent_probability = values[1]
        self.corridor_probability = values[2]
        self.rectangle_probability = values[3]

        self.uniform_probability = max(
            0.0,
            1.0 - sum(values),
        )

    def set_probabilities(
        self,
        goal,
        tangent,
        corridor,
        rectangle,
        min_uniform=0.05,
    ):
        self.goal_probability = max(
            0.0,
            float(goal),
        )
        self.tangent_probability = max(
            0.0,
            float(tangent),
        )
        self.corridor_probability = max(
            0.0,
            float(corridor),
        )
        self.rectangle_probability = max(
            0.0,
            float(rectangle),
        )

        self._normalize_probabilities(
            min_uniform=min_uniform
        )

    def update_probability(
        self,
        iteration,
        current_best_cost=None,
    ):
        if not self.use_hybrid_sampling:
            return

        self.iteration = int(iteration)

        if not self.adaptive_probabilities:
            return

        improved = False

        if (
            current_best_cost is not None
            and current_best_cost < self.best_cost - 1e-6
        ):
            self.best_cost = current_best_cost
            improved = True

        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if iteration < 500:
            goal = min(
                self.initial_goal_probability,
                0.12,
            )
            tangent = min(
                self.initial_tangent_probability,
                0.15,
            )
            corridor = min(
                self.initial_corridor_probability,
                0.08,
            )
            rectangle = min(
                self.initial_rectangle_probability,
                0.18,
            )

        elif iteration < 2000:
            goal = min(
                self.initial_goal_probability,
                0.18,
            )
            tangent = min(
                self.initial_tangent_probability,
                0.22,
            )
            corridor = min(
                self.initial_corridor_probability,
                0.10,
            )
            rectangle = min(
                self.initial_rectangle_probability,
                0.22,
            )

        else:
            goal = min(
                max(self.initial_goal_probability, 0.15),
                0.22,
            )
            tangent = min(
                self.initial_tangent_probability,
                0.20,
            )
            corridor = min(
                self.initial_corridor_probability,
                0.10,
            )
            rectangle = min(
                self.initial_rectangle_probability,
                0.20,
            )

        if self.no_improvement_count > 300:
            tangent *= 0.50
            rectangle *= 0.70
            corridor *= 0.80
            goal *= 0.90

        self.set_probabilities(
            goal=goal,
            tangent=tangent,
            corridor=corridor,
            rectangle=rectangle,
            min_uniform=0.20,
        )

    def effective_sampling_probabilities(self):
        """计算当前有效的采样概率。"""
        effective_goal = self.goal_probability
        effective_rectangle = (
            self.rectangle_probability
            if self.goal_rectangle is not None else 0.0
        )
        effective_corridor = (
            self.corridor_probability
            if self.corridors else 0.0
        )

        # 根据条件计算切向概率
        if self.should_use_tangent_sampling():
            # 使用调用方配置的概率。此前这里把单/多簇概率硬编码为
            # 0.02/0.05，导致 tangent_probability=0.20 等消融配置实际
            # 不生效，切向采样常常只触发一两次。
            effective_tangent = max(0.0, self.tangent_probability)
        else:
            effective_tangent = 0.0

        # 计算均匀采样概率（被取消的切向概率全部转给Uniform）
        effective_uniform = (
            1.0
            - effective_goal
            - effective_tangent
            - effective_rectangle
            - effective_corridor
        )

        # 如果总和超过1，压缩引导概率，保证至少min_uniform_probability给Uniform
        total_guided = effective_goal + effective_tangent + effective_rectangle + effective_corridor
        if total_guided > 1.0 - self.min_uniform_probability:
            scale = (1.0 - self.min_uniform_probability) / max(total_guided, 1e-12)
            effective_goal *= scale
            effective_tangent *= scale
            effective_rectangle *= scale
            effective_corridor *= scale
            effective_uniform = self.min_uniform_probability

        # 确保总和为1
        effective_uniform = max(
            0.0,
            1.0 - effective_goal - effective_tangent - effective_rectangle - effective_corridor
        )

        return {
            "goal": effective_goal,
            "tangent": effective_tangent,
            "rectangle": effective_rectangle,
            "corridor": effective_corridor,
            "uniform": effective_uniform,
        }

    def should_use_tangent_sampling(self):
        """
        判断是否应该启用切向采样。
        条件：
        1. use_tangent_guidance=True
        2. tangent_probability>0
        3. 当前尚未找到可行路径
        4. tangent_guidance.active
        5. 当前pose到goal的直线确实存在阻挡障碍物
        6. 最近阻挡障碍物边缘距离不超过12m
        7. 当前切向目标有效并位于地图范围内
        """
        if not self.use_tangent_guidance:
            return False

        if self.tangent_probability <= 0:
            return False

        if self.has_feasible_path:
            return False

        if not self.tangent_guidance.active:
            return False

        if not self.direct_blocking_indexes:
            return False

        if self.nearest_blocking_distance > 12.0:
            self.stats["tangent_disabled_by_distance"] += 1
            return False

        if (self.tangent_guidance.target_x is None or
            self.tangent_guidance.target_y is None):
            return False

        if not self.inside_bounds(
            self.tangent_guidance.target_x,
            self.tangent_guidance.target_y
        ):
            return False

        return True

    def sample(self):
        if not self.use_hybrid_sampling:
            self.stats["uniform"] += 1
            return self.uniform_sample()

        # 获取有效概率
        probs = self.effective_sampling_probabilities()

        # 记录有效切向概率
        self.stats["effective_tangent_probability_sum"] += probs["tangent"]

        # 如果找到首次解，记录后优化采样
        if self.has_feasible_path:
            self.stats["post_solution_samples"] += 1

        random_value = self.random.random()
        threshold = probs["goal"]

        if random_value < threshold:
            self.stats["goal"] += 1
            return self.goal_sample()

        threshold += probs["tangent"]

        if random_value < threshold:
            if probs["tangent"] > 0 and self.tangent_guidance.active:
                self.stats["tangent"] += 1
                return self.tangent_sample()

            self.stats["tangent_fallback"] += 1
            self.stats["uniform"] += 1
            return self.uniform_sample()

        threshold += probs["rectangle"]

        if (
            random_value < threshold
            and self.goal_rectangle is not None
        ):
            self.stats["rectangle"] += 1
            return self.goal_rectangle_sample()

        threshold += probs["corridor"]

        if (
            random_value < threshold
            and self.corridors
        ):
            self.stats["corridor"] += 1
            return self.corridor_sample()

        self.stats["uniform"] += 1
        return self.uniform_sample()

    def uniform_sample(self):
        return (
            self.random.uniform(
                self.x_min,
                self.x_max,
            ),
            self.random.uniform(
                self.y_min,
                self.y_max,
            ),
            self.random.uniform(
                -math.pi,
                math.pi,
            ),
        )

    def goal_sample(self):
        return (
            self.goal.x,
            self.goal.y,
            self.goal.yaw,
        )

    def tangent_sample(self):
        """
        在切向目标周围采样一个小区域，而不是每次返回完全相同的点。
        yaw设置为采样点指向最终目标的方向。
        """
        target_x, target_y = self.sampling_target

        if not self.tangent_guidance.active:
            self.stats["tangent_fallback"] += 1
            return self.uniform_sample()

        if self.goal_rectangle is not None:
            anchor_x = self.goal_rectangle.anchor_x
            anchor_y = self.goal_rectangle.anchor_y
        else:
            anchor_x = self.goal.x
            anchor_y = self.goal.y

        dx = target_x - anchor_x
        dy = target_y - anchor_y
        distance = math.hypot(dx, dy)

        if distance <= 1e-9:
            self.stats["tangent_fallback"] += 1
            return self.uniform_sample()

        ux = dx / distance
        uy = dy / distance
        nx = -uy
        ny = ux

        for _ in range(self.max_sample_attempts):
            self.stats["tangent_sample_attempts"] += 1

            along_offset = self.random.gauss(
                0.0,
                self.tangent_along_std,
            )

            lateral_offset = self.random.gauss(
                0.0,
                self.tangent_lateral_std,
            )

            x = (
                target_x
                + along_offset * ux
                + lateral_offset * nx
            )

            y = (
                target_y
                + along_offset * uy
                + lateral_offset * ny
            )

            if not self.inside_bounds(x, y):
                continue

            if not self.point_is_free(x, y):
                continue

            # 检查不在锚点后方太远
            if self.current_anchor_pose is not None:
                to_sample_x = x - self.current_anchor_pose.x
                to_sample_y = y - self.current_anchor_pose.y
                forward_projection = (
                    to_sample_x * math.cos(self.current_anchor_pose.yaw) +
                    to_sample_y * math.sin(self.current_anchor_pose.yaw)
                )
                if forward_projection < -2.0:
                    continue

            # yaw设置为采样点指向最终目标的方向（全局规划不考虑转向代价）
            yaw = math.atan2(
                self.goal.y - y,
                self.goal.x - x
            )

            self.stats["tangent_sample_success"] += 1
            self.tangent_sample_history.append({
                "iteration": int(self.iteration),
                "x": float(x),
                "y": float(y),
                "target_x": float(target_x),
                "target_y": float(target_y),
            })
            return x, y, yaw

        self.stats["tangent_fallback"] += 1
        return self.uniform_sample()

    def goal_rectangle_sample(self):
        rectangle = self.goal_rectangle

        if rectangle is None:
            return self.uniform_sample()

        target_x, target_y = self.sampling_target
        dx = target_x - rectangle.anchor_x
        dy = target_y - rectangle.anchor_y
        goal_distance = math.hypot(dx, dy)

        if goal_distance < 1e-9:
            return self.goal_sample()

        direction_yaw = math.atan2(dy, dx)
        cos_yaw = math.cos(direction_yaw)
        sin_yaw = math.sin(direction_yaw)

        rectangle_start = rectangle.forward_offset
        rectangle_end = (
            rectangle.forward_offset
            + rectangle.length
        )

        for _ in range(self.max_sample_attempts):
            if (
                self.random.random()
                < self.near_anchor_probability
            ):
                near_start = max(
                    0.0,
                    rectangle_start,
                )

                near_end = min(
                    rectangle_end,
                    near_start
                    + rectangle.length
                    * self.near_anchor_length_ratio,
                )

                if near_end > near_start:
                    ratio = self.random.betavariate(
                        1.0,
                        3.0,
                    )

                    local_x = (
                        near_start
                        + ratio
                        * (near_end - near_start)
                    )
                else:
                    local_x = self.random.uniform(
                        rectangle_start,
                        rectangle_end,
                    )
            else:
                local_x = self.random.uniform(
                    rectangle_start,
                    rectangle_end,
                )

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

            if not self.inside_bounds(
                world_x,
                world_y,
            ):
                continue

            sample_to_goal_yaw = math.atan2(
                target_y - world_y,
                target_x - world_x,
            )

            sample_yaw = self.normalize_angle(
                sample_to_goal_yaw
                + self.random.gauss(
                    0.0,
                    self.heading_std,
                )
            )

            return (
                world_x,
                world_y,
                sample_yaw,
            )

        return self.uniform_sample()

    def corridor_sample(self):
        if not self.corridors:
            return self.uniform_sample()

        corridor = self.random.choice(
            self.corridors
        )

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

        x = (
            center_x
            - offset * math.sin(corridor_yaw)
        )

        y = (
            center_y
            + offset * math.cos(corridor_yaw)
        )

        if not self.inside_bounds(x, y):
            return self.uniform_sample()

        yaw = self.normalize_angle(
            corridor_yaw
            + self.random.gauss(
                0.0,
                self.heading_std,
            )
        )

        return x, y, yaw
