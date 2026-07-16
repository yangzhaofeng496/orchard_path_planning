"""Use the existing DWA planner as an execution-feasibility evaluator."""

from dataclasses import dataclass, replace
import math
from typing import List

from ..base import CircleObstacle, LocalPlanResult, Pose, VehicleState
from ..config import DWAConfig
from ..dwa import DWAPlanner


@dataclass
class DWAEvaluation:
    score: float
    collision_risk: float
    tracking_error: float
    motion_error: float
    feasible: bool
    reason: str
    min_clearance: float


class DWAFeedbackEvaluator:
    def __init__(
        self,
        config: DWAConfig,
        bounds: tuple,
        max_tracking_error: float = 1.0,
        min_clearance: float = 0.25,
        min_score: float = 0.45,
        min_progress_speed: float = 0.03,
    ):
        self.config = replace(config)
        self.bounds = bounds
        self.max_tracking_error = max_tracking_error
        self.required_clearance = min_clearance
        self.min_score = min_score
        self.min_progress_speed = min_progress_speed
        self.last_result = None

    def evaluate(
        self,
        teb_result: LocalPlanResult,
        state: VehicleState,
        obstacles: List[CircleObstacle],
    ) -> DWAEvaluation:
        planner = DWAPlanner(replace(self.config), self.bounds)
        planner.set_global_path(teb_result.trajectory)
        result = planner.plan(state, obstacles)
        self.last_result = result
        if result.best is None:
            return DWAEvaluation(0.0, 1.0, math.inf, math.inf, False,
                                 "DWA_NO_REACHABLE_CONTROL", 0.0)

        candidate = result.best
        # DWA核心仍按原代价选择；反馈层避免把“原地停车”当作可执行成功。
        if candidate.control.speed < self.min_progress_speed:
            moving_candidates = [
                item for item in result.candidates
                if item.valid and item.control.speed >= self.min_progress_speed
            ]
            if moving_candidates:
                candidate = min(moving_candidates, key=lambda item: item.cost)
                self.last_result.best = candidate
            else:
                return DWAEvaluation(
                    0.0, 1.0, math.inf, 1.0, False,
                    "DWA_STALLED", float(candidate.clearance),
                )

        dwa_trajectory = candidate.trajectory
        tracking = self._mean_tracking_error(dwa_trajectory, teb_result.trajectory)
        clearance = float(candidate.clearance)
        if not math.isfinite(clearance):
            clearance = 1e6
        collision_risk = 1.0 / (1.0 + max(0.0, clearance))
        motion_error = (
            abs(candidate.control.speed - teb_result.control.speed)
            / max(self.config.max_speed, 1e-6)
            + abs(candidate.control.steering - teb_result.control.steering)
            / max(math.radians(self.config.max_steer_deg), 1e-6)
        ) / 2.0
        max_curvature = self._max_curvature(teb_result.trajectory)
        allowed_curvature = math.tan(math.radians(self.config.max_steer_deg)) / self.config.wheel_base
        curvature_excess = max(0.0, max_curvature / max(allowed_curvature, 1e-9) - 1.0)
        motion_error += curvature_excess
        tracking_score = max(0.0, 1.0 - tracking / max(self.max_tracking_error, 1e-6))
        clearance_score = min(1.0, max(0.0, clearance) / max(self.required_clearance, 1e-6))
        motion_score = max(0.0, 1.0 - motion_error)
        score = 0.45 * tracking_score + 0.35 * clearance_score + 0.20 * motion_score

        if max_curvature > allowed_curvature + 1e-3:
            reason = "KINEMATIC_INFEASIBLE"
        elif clearance < self.required_clearance:
            reason = "HIGH_COLLISION_RISK"
        elif tracking > self.max_tracking_error:
            reason = "TRACKING_INFEASIBLE"
        elif score < self.min_score:
            reason = "LOW_EXECUTION_SCORE"
        else:
            reason = "OK"
        return DWAEvaluation(score, collision_risk, tracking, motion_error,
                             reason == "OK", reason, clearance)

    @staticmethod
    def _mean_tracking_error(candidate, reference):
        if not candidate or not reference:
            return math.inf
        return sum(
            min(math.hypot(p.x - q.x, p.y - q.y) for q in reference)
            for p in candidate
        ) / len(candidate)

    @staticmethod
    def _max_curvature(path):
        values = []
        for first, second in zip(path[:-1], path[1:]):
            distance = math.hypot(second.x - first.x, second.y - first.y)
            if distance > 1e-8:
                dyaw = (second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi
                values.append(abs(dyaw / distance))
        return max(values, default=0.0)
