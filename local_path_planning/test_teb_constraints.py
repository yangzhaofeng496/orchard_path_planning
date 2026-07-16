import math
import unittest

from local_path_planning.base import CircleObstacle, Pose, VehicleState
from local_path_planning.config import TEBConfig
from local_path_planning.teb import TEBPlanner


class TEBConstraintTest(unittest.TestCase):
    def make_planner(self):
        config = TEBConfig(
            num_samples=15,
            max_iterations=100,
            lookahead_distance=6.0,
            vehicle_front_length=1.2,
            vehicle_rear_length=0.5,
            vehicle_width=1.0,
            vehicle_safety_margin=0.1,
            obstacle_min_dist=0.5,
            obstacle_influence_dist=2.0,
        )
        planner = TEBPlanner(config, bounds=(0.0, 20.0, 0.0, 12.0))
        planner.set_global_path([Pose(2.0 + i * 16.0 / 19.0, 6.0, 0.0) for i in range(20)])
        return planner

    def test_local_window_and_dynamic_constraints(self):
        planner = self.make_planner()
        state = VehicleState(2.0, 6.0, 0.0, 0.5, 0.0)
        result = planner.plan(state, [])
        self.assertIsNotNone(result)
        self.assertGreater(result.cost, 0.0)
        self.assertLessEqual(planner.teb_nodes[-1].x - state.x, 6.01)

        variables = planner._pack_variables()
        self.assertGreaterEqual(planner._speed_constraint(variables).min(), -1e-4)
        self.assertGreaterEqual(planner._acceleration_constraint(variables).min(), -1e-4)
        self.assertGreaterEqual(planner._steering_constraint(variables).min(), -1e-4)
        self.assertGreaterEqual(planner._forward_constraint(variables).min(), -1e-4)
        self.assertGreaterEqual(planner._progress_constraint(variables).min(), -1e-4)

    def test_obstacle_scenario_returns_collision_free_trajectory(self):
        planner = self.make_planner()
        state = VehicleState(2.0, 6.0, 0.0, 0.5, 0.0)
        obstacles = [CircleObstacle(10.0, 6.0, 1.5)]
        result = planner.plan(state, obstacles)
        self.assertIsNotNone(result)
        self.assertIsNone(planner._check_trajectory_collision(obstacles))
        self.assertLessEqual(result.control.speed, planner.config.max_speed + 1e-6)
        self.assertLessEqual(
            abs(result.control.steering),
            math.radians(planner.config.max_steer_deg) + 1e-6,
        )


if __name__ == '__main__':
    unittest.main()
