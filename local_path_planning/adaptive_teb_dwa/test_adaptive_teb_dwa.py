import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_path_planning.base import CircleObstacle, Pose, VehicleState
from local_path_planning.base import Control, LocalPlanResult
from local_path_planning.config import DWAConfig
from local_path_planning.config import TEBConfig
from local_path_planning.adaptive_teb_dwa.adaptive_window import (
    AdaptiveWindowConfig,
    AdaptiveWindowSelector,
)
from local_path_planning.adaptive_teb_dwa.dwa_feedback import DWAEvaluation
from local_path_planning.adaptive_teb_dwa.dwa_feedback import DWAFeedbackEvaluator
from local_path_planning.adaptive_teb_dwa.parameter_manager import FeedbackConfig, ParameterManager
from local_path_planning.adaptive_teb_dwa.planner import AdaptiveTEBDWAPlanner


class CostmapStub:
    def __init__(self, obstacles):
        self.obstacles = obstacles


class AdaptiveFusionTest(unittest.TestCase):
    def setUp(self):
        self.path = [Pose(5.0 + i * 0.5, 5.0, 0.0) for i in range(30)]
        self.state = VehicleState(5.0, 5.0, 0.0, 0.5, 0.0)

    def test_dense_environment_shortens_window(self):
        selector = AdaptiveWindowSelector(AdaptiveWindowConfig())
        free = selector.select(self.path, self.state, [])
        dense = selector.select(
            self.path,
            self.state,
            [CircleObstacle(6.0 + i * 0.3, 5.5, 0.3) for i in range(8)],
        )
        self.assertLess(dense.lookahead_distance, free.lookahead_distance)

    def test_collision_feedback_increases_obstacle_weight(self):
        config = TEBConfig()
        manager = ParameterManager(config, FeedbackConfig())
        evaluation = DWAEvaluation(0.1, 0.9, 0.2, 0.1, False,
                                   "HIGH_COLLISION_RISK", 0.1)
        adjustment = manager.adjust(evaluation, obstacle_density=0.8)
        self.assertGreater(adjustment.obstacle_weight, config.w_obstacle)
        self.assertLess(adjustment.window_scale, 1.0)

    def test_costmap_adapters(self):
        obstacles = [CircleObstacle(1.0, 2.0, 0.3)]
        self.assertEqual(AdaptiveTEBDWAPlanner._extract_obstacles(obstacles), obstacles)
        self.assertEqual(
            AdaptiveTEBDWAPlanner._extract_obstacles(CostmapStub(obstacles)), obstacles
        )

    def test_zero_speed_best_is_replaced_by_moving_candidate(self):
        stopped = SimpleNamespace(
            control=Control(0.0, 0.0), trajectory=self.path[:3],
            valid=True, clearance=2.0, cost=1.0,
        )
        moving = SimpleNamespace(
            control=Control(0.05, 0.0), trajectory=self.path[:3],
            valid=True, clearance=2.0, cost=1.1,
        )
        dwa_result = SimpleNamespace(
            best=stopped, candidates=[stopped, moving],
            local_goal=self.path[2], nearest_path_index=0,
        )
        fake_planner = SimpleNamespace(
            set_global_path=lambda _path: None,
            plan=lambda _state, _obstacles: dwa_result,
        )
        teb_result = LocalPlanResult(True, Control(0.5, 0.0), self.path[:3], 0.0)
        evaluator = DWAFeedbackEvaluator(DWAConfig(), (0, 30, 0, 30))
        with patch(
            'local_path_planning.adaptive_teb_dwa.dwa_feedback.DWAPlanner',
            return_value=fake_planner,
        ):
            evaluation = evaluator.evaluate(teb_result, self.state, [])
        self.assertIs(dwa_result.best, moving)
        self.assertNotEqual(evaluation.reason, 'DWA_STALLED')

    def test_dwa_validated_control_replaces_zero_teb_first_speed(self):
        from local_path_planning.adaptive_teb_dwa.planner import TrajectoryPoint
        planner = object.__new__(AdaptiveTEBDWAPlanner)
        planner.dwa = SimpleNamespace(last_result=SimpleNamespace(
            best=SimpleNamespace(control=Control(0.05, 0.1))
        ))
        trajectory = [TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.0)]
        planner._apply_dwa_validated_first_control(trajectory)
        self.assertEqual(trajectory[0].v, 0.05)
        self.assertEqual(trajectory[0].steering, 0.1)


if __name__ == "__main__":
    unittest.main()
