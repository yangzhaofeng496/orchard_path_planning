"""Thin adapter around the existing TEBPlanner."""

from dataclasses import replace
from typing import List

from ..base import CircleObstacle, LocalPlanResult, Pose, VehicleState
from ..config import TEBConfig
from ..teb import TEBPlanner


class TEBInterface:
    def __init__(self, config: TEBConfig, bounds: tuple):
        self.config = replace(config)
        self.bounds = bounds
        self.last_planner = None

    def optimize(
        self, reference: List[Pose], state: VehicleState, obstacles: List[CircleObstacle]
    ) -> LocalPlanResult | None:
        # A fresh wrapper prevents the previous adaptive window from leaking nodes.
        self.last_planner = TEBPlanner(replace(self.config), self.bounds)
        self.last_planner.set_global_path(reference)
        return self.last_planner.plan(state, obstacles)

    def update_weights(self, obstacle: float, kinematic: float, omega: float):
        self.config.w_obstacle = obstacle
        self.config.w_kinematics = kinematic
        self.config.w_omega = omega
