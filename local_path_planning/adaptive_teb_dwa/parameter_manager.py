"""Bounded feedback policy for TEB weights and adaptive-window scale."""

from dataclasses import dataclass

from ..config import TEBConfig
from .dwa_feedback import DWAEvaluation


@dataclass
class FeedbackConfig:
    obstacle_multiplier: float = 1.5
    kinematic_multiplier: float = 1.35
    omega_multiplier: float = 1.2
    shrink_factor: float = 0.78
    expand_factor: float = 1.12
    min_window_scale: float = 0.55
    max_window_scale: float = 1.35
    max_weight_multiplier: float = 5.0


@dataclass
class ParameterAdjustment:
    window_scale: float
    obstacle_weight: float
    kinematic_weight: float
    omega_weight: float
    reason: str


class ParameterManager:
    def __init__(self, teb_config: TEBConfig, config: FeedbackConfig):
        self.base = teb_config
        self.config = config
        self.window_scale = 1.0
        self.obstacle_weight = teb_config.w_obstacle
        self.kinematic_weight = teb_config.w_kinematics
        self.omega_weight = teb_config.w_omega

    def adjust(self, evaluation: DWAEvaluation, obstacle_density: float):
        if evaluation.reason == "HIGH_COLLISION_RISK":
            self.obstacle_weight *= self.config.obstacle_multiplier
            self.window_scale *= self.config.shrink_factor
        elif evaluation.reason == "DWA_STALLED":
            self.obstacle_weight *= self.config.obstacle_multiplier
            self.kinematic_weight *= 1.10
            self.window_scale *= self.config.shrink_factor
        elif evaluation.reason in (
            "TRACKING_INFEASIBLE", "KINEMATIC_INFEASIBLE",
            "DWA_NO_REACHABLE_CONTROL", "TEB_OPTIMIZATION_FAILED",
        ):
            self.kinematic_weight *= self.config.kinematic_multiplier
            self.omega_weight *= self.config.omega_multiplier
            self.window_scale *= self.config.shrink_factor
        elif evaluation.reason == "LOW_EXECUTION_SCORE":
            self.kinematic_weight *= 1.15
            self.window_scale *= self.config.shrink_factor
        elif evaluation.feasible and evaluation.collision_risk < 0.15 and obstacle_density < 0.25:
            self.window_scale *= self.config.expand_factor

        maximum = self.config.max_weight_multiplier
        self.obstacle_weight = min(self.base.w_obstacle * maximum, self.obstacle_weight)
        self.kinematic_weight = min(self.base.w_kinematics * maximum, self.kinematic_weight)
        self.omega_weight = min(self.base.w_omega * maximum, self.omega_weight)
        self.window_scale = min(self.config.max_window_scale,
                                max(self.config.min_window_scale, self.window_scale))
        return ParameterAdjustment(
            self.window_scale, self.obstacle_weight, self.kinematic_weight,
            self.omega_weight, evaluation.reason,
        )
