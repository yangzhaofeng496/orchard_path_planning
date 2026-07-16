"""Curvature-constrained corner smoothing and velocity look-ahead."""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

from .shortcut import CollisionChecker

Point = Tuple[float, float]


@dataclass
class SmoothedPath:
    points: List[Point]
    yaws: List[float]
    curvatures: List[float]
    speeds: List[float]
    corner_indices: List[int]


class CurvatureSmoother:
    """Round polyline corners and generate a dynamically feasible speed profile."""

    def __init__(
        self,
        collision_checker: Optional[CollisionChecker] = None,
        max_curvature: float = 0.22,
        interpolation_spacing: float = 0.20,
        corner_angle_threshold: float = math.radians(8.0),
        corner_blend_distance: float = 2.0,
        max_lateral_accel: float = 1.2,
        max_speed: float = 2.0,
        max_accel: float = 1.0,
        max_decel: float = 1.0,
        lookahead_distance: float = 5.0,
        start_speed: float = 0.0,
        end_speed: float = 0.0,
        verbose: bool = False,
    ):
        if max_curvature <= 0 or interpolation_spacing <= 0:
            raise ValueError("max_curvature 和 interpolation_spacing 必须大于 0")
        self.collision_checker = collision_checker
        self.max_curvature = max_curvature
        self.spacing = interpolation_spacing
        self.corner_threshold = corner_angle_threshold
        self.blend_distance = corner_blend_distance
        self.max_lateral_accel = max_lateral_accel
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.lookahead_distance = lookahead_distance
        self.start_speed = start_speed
        self.end_speed = end_speed
        self.verbose = verbose

    def smooth(self, path: List[Point]) -> SmoothedPath:
        points = self._deduplicate(path)
        if len(points) < 2:
            return SmoothedPath(points, [0.0] * len(points), [0.0] * len(points), [0.0] * len(points), [])

        rounded, corner_locations = self._round_corners(points)
        sampled = self._resample(rounded)
        yaws = self._compute_yaws(sampled)
        curvatures = self._compute_curvatures(sampled)
        speeds = self._velocity_lookahead(sampled, curvatures)
        corner_indices = [self._nearest_index(sampled, p) for p in corner_locations]

        if self.verbose:
            max_kappa = max((abs(k) for k in curvatures), default=0.0)
            print(
                f"[Curvature] corners={len(corner_locations)}, points={len(sampled)}, "
                f"max_curvature={max_kappa:.4f} 1/m, "
                f"speed=[{min(speeds):.2f}, {max(speeds):.2f}] m/s"
            )
        return SmoothedPath(sampled, yaws, curvatures, speeds, corner_indices)

    def _round_corners(self, path: List[Point]):
        trims = [0.0] * len(path)
        turns = [0.0] * len(path)
        for i in range(1, len(path) - 1):
            incoming = self._unit(path[i - 1], path[i])
            outgoing = self._unit(path[i], path[i + 1])
            turn = math.acos(np.clip(incoming[0] * outgoing[0] + incoming[1] * outgoing[1], -1.0, 1.0))
            turns[i] = turn
            if turn >= self.corner_threshold:
                # Circular fillet: tangent distance = R * tan(turn / 2).
                required = math.tan(min(turn, math.radians(170.0)) / 2.0) / self.max_curvature
                trims[i] = min(
                    max(self.blend_distance, required),
                    0.90 * self._distance(path[i - 1], path[i]),
                    0.90 * self._distance(path[i], path[i + 1]),
                )

        # Adjacent corners share a segment. Scale both trims only when their
        # requested transition lengths overlap on that segment.
        for i in range(1, len(path) - 2):
            available = 0.90 * self._distance(path[i], path[i + 1])
            requested = trims[i] + trims[i + 1]
            if requested > available and requested > 1e-9:
                scale = available / requested
                trims[i] *= scale
                trims[i + 1] *= scale

        output = [path[0]]
        corners = []
        for i in range(1, len(path) - 1):
            previous, vertex, following = path[i - 1], path[i], path[i + 1]
            incoming = self._unit(previous, vertex)
            outgoing = self._unit(vertex, following)
            turn = turns[i]
            if turn < self.corner_threshold:
                output.append(vertex)
                continue

            trim = trims[i]
            if trim <= self.spacing:
                output.append(vertex)
                continue

            entry = (vertex[0] - incoming[0] * trim, vertex[1] - incoming[1] * trim)
            exit_ = (vertex[0] + outgoing[0] * trim, vertex[1] + outgoing[1] * trim)
            radius = trim / max(math.tan(turn / 2.0), 1e-9)
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            direction = 1.0 if cross >= 0.0 else -1.0
            normal = (-incoming[1] * direction, incoming[0] * direction)
            center = (entry[0] + normal[0] * radius, entry[1] + normal[1] * radius)
            start_angle = math.atan2(entry[1] - center[1], entry[0] - center[0])
            arc_length = radius * turn
            count = max(4, int(arc_length / self.spacing) + 1)
            curve = [
                (
                    center[0] + radius * math.cos(start_angle + direction * turn * t),
                    center[1] + radius * math.sin(start_angle + direction * turn * t),
                )
                for t in np.linspace(0.0, 1.0, count)
            ]
            candidate = [output[-1]] + curve
            if self._collision_free(candidate):
                if self._distance(output[-1], entry) > 1e-8:
                    output.append(entry)
                output.extend(curve[1:])
                corners.append(vertex)
            else:
                output.append(vertex)
                if self.verbose:
                    print(f"[Curvature] corner {i} smoothing rejected by collision checker")
        output.append(path[-1])
        return self._deduplicate(output), corners

    def _velocity_lookahead(self, points: List[Point], curvatures: List[float]):
        n = len(points)
        curve_limits = [
            min(self.max_speed, math.sqrt(self.max_lateral_accel / max(abs(k), 1e-9)))
            for k in curvatures
        ]
        # Propagate the minimum upcoming curvature speed over the look-ahead horizon.
        targets = curve_limits.copy()
        for i in range(n):
            distance = 0.0
            j = i
            while j + 1 < n and distance < self.lookahead_distance:
                distance += self._distance(points[j], points[j + 1])
                j += 1
                targets[i] = min(targets[i], curve_limits[j])

        speeds = targets
        speeds[0] = min(speeds[0], self.start_speed)
        for i in range(1, n):
            ds = self._distance(points[i - 1], points[i])
            speeds[i] = min(speeds[i], math.sqrt(speeds[i - 1] ** 2 + 2.0 * self.max_accel * ds))
        speeds[-1] = min(speeds[-1], self.end_speed)
        for i in range(n - 2, -1, -1):
            ds = self._distance(points[i], points[i + 1])
            speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2.0 * self.max_decel * ds))
        return speeds

    def _resample(self, points: List[Point]):
        cumulative = [0.0]
        for a, b in zip(points[:-1], points[1:]):
            cumulative.append(cumulative[-1] + self._distance(a, b))
        if cumulative[-1] <= 1e-9:
            return [points[0]]
        targets = list(np.arange(0.0, cumulative[-1], self.spacing)) + [cumulative[-1]]
        result, segment = [], 0
        for target in targets:
            while segment < len(points) - 2 and target > cumulative[segment + 1]:
                segment += 1
            length = cumulative[segment + 1] - cumulative[segment]
            ratio = 0.0 if length <= 1e-12 else (target - cumulative[segment]) / length
            a, b = points[segment], points[segment + 1]
            result.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
        return self._deduplicate(result)

    @staticmethod
    def _compute_yaws(points):
        if len(points) < 2:
            return [0.0] * len(points)
        yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(points[:-1], points[1:])]
        return yaws + [yaws[-1]]

    @staticmethod
    def _compute_curvatures(points):
        if len(points) < 3:
            return [0.0] * len(points)
        values = [0.0]
        for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
            ab = CurvatureSmoother._distance(a, b)
            bc = CurvatureSmoother._distance(b, c)
            ac = CurvatureSmoother._distance(a, c)
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            values.append(0.0 if ab * bc * ac <= 1e-12 else 2.0 * cross / (ab * bc * ac))
        values.append(0.0)
        return values

    def _collision_free(self, points):
        return self.collision_checker is None or all(
            self.collision_checker.check_line(a, b) for a, b in zip(points[:-1], points[1:])
        )

    @staticmethod
    def _unit(a, b):
        length = CurvatureSmoother._distance(a, b)
        return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)

    @staticmethod
    def _distance(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _deduplicate(points):
        result = []
        for point in points:
            point = (float(point[0]), float(point[1]))
            if not result or CurvatureSmoother._distance(result[-1], point) > 1e-8:
                result.append(point)
        return result

    @staticmethod
    def _nearest_index(points, target):
        return min(range(len(points)), key=lambda i: CurvatureSmoother._distance(points[i], target))
