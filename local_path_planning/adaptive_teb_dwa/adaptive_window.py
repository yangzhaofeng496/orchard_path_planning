"""Obstacle-, curvature-, and speed-aware reference window selection."""

from dataclasses import dataclass
import math
from typing import List

from ..base import CircleObstacle, Pose, VehicleState


@dataclass
class AdaptiveWindowConfig:
    min_lookahead: float = 3.0
    max_lookahead: float = 12.0
    base_lookahead: float = 7.0
    sensing_radius: float = 6.0
    density_reference: float = 0.22
    curvature_reference: float = 0.25
    speed_gain: float = 1.2
    density_gain: float = 3.0
    curvature_gain: float = 3.0
    free_space_gain: float = 1.5


@dataclass
class WindowSelection:
    reference_path: List[Pose]
    lookahead_distance: float
    closest_index: int
    obstacle_density: float
    free_space_ratio: float
    path_curvature: float


class AdaptiveWindowSelector:
    def __init__(self, config: AdaptiveWindowConfig):
        self.config = config

    def select(
        self,
        global_path: List[Pose],
        state: VehicleState,
        obstacles: List[CircleObstacle],
        window_scale: float = 1.0,
    ) -> WindowSelection:
        if not global_path:
            raise ValueError("global_path 不能为空")
        closest = min(
            range(len(global_path)),
            key=lambda i: math.hypot(global_path[i].x - state.x, global_path[i].y - state.y),
        )
        nearby = [
            obstacle for obstacle in obstacles
            if math.hypot(obstacle.x - state.x, obstacle.y - state.y)
            <= self.config.sensing_radius + obstacle.radius
        ]
        area = math.pi * self.config.sensing_radius ** 2
        occupied = sum(math.pi * obstacle.radius ** 2 for obstacle in nearby)
        density = min(1.0, occupied / max(area * self.config.density_reference, 1e-9))
        nearest_clearance = min(
            (math.hypot(o.x - state.x, o.y - state.y) - o.radius for o in nearby),
            default=self.config.sensing_radius,
        )
        free_ratio = min(1.0, max(0.0, nearest_clearance / self.config.sensing_radius))
        curvature = self._path_curvature(global_path, closest, 12)
        curvature_level = min(1.0, curvature / max(self.config.curvature_reference, 1e-9))

        lookahead = (
            self.config.base_lookahead
            + self.config.speed_gain * max(0.0, state.speed)
            - self.config.density_gain * density
            - self.config.curvature_gain * curvature_level
            + self.config.free_space_gain * free_ratio
        ) * window_scale
        lookahead = min(self.config.max_lookahead, max(self.config.min_lookahead, lookahead))
        reference = self._slice_by_arc_length(global_path, state, closest, lookahead)
        return WindowSelection(reference, lookahead, closest, density, free_ratio, curvature)

    @staticmethod
    def _path_curvature(path: List[Pose], start: int, count: int) -> float:
        subset = path[start:min(len(path), start + count)]
        values = []
        for first, middle, last in zip(subset[:-2], subset[1:-1], subset[2:]):
            a = math.hypot(middle.x - first.x, middle.y - first.y)
            b = math.hypot(last.x - middle.x, last.y - middle.y)
            c = math.hypot(last.x - first.x, last.y - first.y)
            cross = ((middle.x - first.x) * (last.y - first.y)
                     - (middle.y - first.y) * (last.x - first.x))
            if a * b * c > 1e-9:
                values.append(abs(2.0 * cross / (a * b * c)))
        return max(values, default=0.0)

    @staticmethod
    def _slice_by_arc_length(path, state, start, distance):
        result = [Pose(state.x, state.y, state.yaw)]
        travelled = 0.0
        previous = result[0]
        for pose in path[start:]:
            segment = math.hypot(pose.x - previous.x, pose.y - previous.y)
            if segment <= 1e-9:
                continue
            if travelled + segment >= distance:
                ratio = (distance - travelled) / segment
                x = previous.x + ratio * (pose.x - previous.x)
                y = previous.y + ratio * (pose.y - previous.y)
                yaw = math.atan2(pose.y - previous.y, pose.x - previous.x)
                result.append(Pose(x, y, yaw))
                break
            result.append(Pose(pose.x, pose.y, pose.yaw))
            travelled += segment
            previous = pose
        if len(result) == 1 and start < len(path):
            result.append(path[start])
        return result
