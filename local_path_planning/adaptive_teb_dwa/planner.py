"""Main adaptive TEB-DWA orchestration; contains no TEB/DWA core algorithm."""

from dataclasses import dataclass, field
import math
import time
from typing import List

from ..base import CircleObstacle, Pose, VehicleState
from ..config import DWAConfig, TEBConfig
from .adaptive_window import AdaptiveWindowConfig, AdaptiveWindowSelector, WindowSelection
from .dwa_feedback import DWAEvaluation, DWAFeedbackEvaluator
from .parameter_manager import FeedbackConfig, ParameterAdjustment, ParameterManager
from .teb_interface import TEBInterface


@dataclass
class AdaptivePlannerConfig:
    max_feedback_iterations: int = 6
    accept_best_score: bool = True
    debug_log: bool = True


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    v: float
    steering: float


@dataclass
class PlannerStatistics:
    calls: int = 0
    successes: int = 0
    dwa_rejections: int = 0
    teb_reoptimizations: int = 0
    total_computation_time: float = 0.0

    @property
    def success_rate(self):
        return self.successes / self.calls if self.calls else 0.0


@dataclass
class AdaptivePlannerResult:
    success: bool
    trajectory: List[TrajectoryPoint]
    window: WindowSelection | None
    evaluation: DWAEvaluation | None
    adjustments: List[ParameterAdjustment] = field(default_factory=list)
    feedback_iterations: int = 0
    computation_time: float = 0.0
    message: str = ""


class AdaptiveTEBDWAPlanner:
    def __init__(
        self,
        teb_config: TEBConfig,
        dwa_config: DWAConfig,
        bounds: tuple,
        planner_config: AdaptivePlannerConfig | None = None,
        window_config: AdaptiveWindowConfig | None = None,
        feedback_config: FeedbackConfig | None = None,
    ):
        self.config = planner_config or AdaptivePlannerConfig()
        self.window_selector = AdaptiveWindowSelector(window_config or AdaptiveWindowConfig())
        self.teb = TEBInterface(teb_config, bounds)
        self.dwa = DWAFeedbackEvaluator(dwa_config, bounds)
        self.parameters = ParameterManager(teb_config, feedback_config or FeedbackConfig())
        self.global_path: List[Pose] = []
        self.statistics = PlannerStatistics()

    def set_global_path(self, path: List[Pose]):
        if len(path) < 2:
            raise ValueError("global_path 至少需要两个点")
        self.global_path = list(path)

    def plan(self, robot_state: VehicleState, local_costmap) -> AdaptivePlannerResult:
        started = time.perf_counter()
        self.statistics.calls += 1
        obstacles = self._extract_obstacles(local_costmap)
        if not self.global_path:
            return self._finish(False, [], None, None, [], 0, started, "尚未设置全局路径")

        best = None
        adjustments = []
        for iteration in range(1, self.config.max_feedback_iterations + 1):
            window = self.window_selector.select(
                self.global_path, robot_state, obstacles, self.parameters.window_scale
            )
            teb_result = self.teb.optimize(window.reference_path, robot_state, obstacles)
            if teb_result is None:
                evaluation = DWAEvaluation(
                    0.0, 1.0, math.inf, math.inf, False, "TEB_OPTIMIZATION_FAILED", 0.0
                )
            else:
                evaluation = self.dwa.evaluate(teb_result, robot_state, obstacles)
                if best is None or evaluation.score > best[0].score:
                    best = (evaluation, teb_result, window)

            self._log(iteration, window, evaluation)
            if teb_result is not None and evaluation.feasible:
                trajectory = self._make_trajectory(teb_result.trajectory, teb_result.control)
                self._apply_dwa_validated_first_control(trajectory)
                self.statistics.successes += 1
                return self._finish(
                    True, trajectory, window, evaluation, adjustments, iteration, started, "OK"
                )

            self.statistics.dwa_rejections += int(teb_result is not None)
            if iteration < self.config.max_feedback_iterations:
                self.statistics.teb_reoptimizations += 1
                adjustment = self.parameters.adjust(evaluation, window.obstacle_density)
                adjustments.append(adjustment)
                self.teb.update_weights(
                    adjustment.obstacle_weight,
                    adjustment.kinematic_weight,
                    adjustment.omega_weight,
                )

        if best is not None and self.config.accept_best_score:
            evaluation, teb_result, window = best
            trajectory = self._make_trajectory(teb_result.trajectory, teb_result.control)
            return self._finish(
                False, trajectory, window, evaluation, adjustments,
                self.config.max_feedback_iterations, started,
                f"DWA拒绝，保留最佳候选供诊断: {evaluation.reason}",
            )
        return self._finish(
            False, [], window, evaluation, adjustments,
            self.config.max_feedback_iterations, started, evaluation.reason,
        )

    def _make_trajectory(self, poses, control):
        result = []
        nodes = getattr(self.teb.last_planner, "teb_nodes", [])
        for index, pose in enumerate(poses):
            if index + 1 < len(poses):
                following = poses[index + 1]
                ds = math.hypot(following.x - pose.x, following.y - pose.y)
                dyaw = self._normalize(following.yaw - pose.yaw)
                steering = 0.0 if ds < 1e-8 else math.atan(self.teb.config.wheel_base * dyaw / ds)
                dt = nodes[index].dt if index < len(nodes) else self.teb.config.dt
                speed = min(self.teb.config.max_speed, ds / max(dt, 1e-3))
            elif result:
                steering = result[-1].steering
                speed = result[-1].v
            else:
                steering = control.steering
                speed = control.speed
            result.append(TrajectoryPoint(pose.x, pose.y, pose.yaw, speed, steering))
        return result

    def _apply_dwa_validated_first_control(self, trajectory):
        """Use DWA's reachable first control while retaining the TEB spatial trajectory."""
        dwa_result = self.dwa.last_result
        if not trajectory or dwa_result is None or dwa_result.best is None:
            return
        validated = dwa_result.best.control
        trajectory[0].v = validated.speed
        trajectory[0].steering = validated.steering

    @staticmethod
    def _extract_obstacles(costmap) -> List[CircleObstacle]:
        if costmap is None:
            return []
        if isinstance(costmap, (list, tuple)):
            return list(costmap)
        if hasattr(costmap, "get_obstacles"):
            return list(costmap.get_obstacles())
        if hasattr(costmap, "obstacles"):
            return list(costmap.obstacles)
        raise TypeError("local_costmap 需为障碍物列表，或实现 get_obstacles()/.obstacles")

    def _finish(self, success, trajectory, window, evaluation, adjustments,
                iterations, started, message):
        elapsed = time.perf_counter() - started
        self.statistics.total_computation_time += elapsed
        return AdaptivePlannerResult(
            success, trajectory, window, evaluation, adjustments,
            iterations, elapsed, message,
        )

    def _log(self, iteration, window, evaluation):
        if self.config.debug_log:
            print(
                f"[AdaptiveTEB-DWA] iter={iteration}, Ld={window.lookahead_distance:.2f}, "
                f"density={window.obstacle_density:.2f}, curvature={window.path_curvature:.3f}, "
                f"score={evaluation.score:.3f}, feasible={evaluation.feasible}, "
                f"reason={evaluation.reason}"
            )

    @staticmethod
    def _normalize(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
