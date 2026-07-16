import unittest

from local_path_planning.base import CircleObstacle
from local_path_planning.config import TEBConfig
from local_path_planning.teb import TEBNode, TEBPlanner


class TEBCollisionTest(unittest.TestCase):
    def setUp(self):
        config = TEBConfig(
            vehicle_front_length=2.0,
            vehicle_rear_length=1.0,
            vehicle_width=1.0,
            vehicle_safety_margin=0.2,
            collision_check_resolution=0.1,
        )
        self.planner = TEBPlanner(config, bounds=(0.0, 20.0, 0.0, 20.0))

    def test_obstacle_inside_vehicle_footprint(self):
        obstacle = CircleObstacle(6.5, 5.0, 0.2)
        clearance = self.planner._vehicle_obstacle_clearance(5.0, 5.0, 0.0, obstacle)
        self.assertLessEqual(clearance, 0.0)

    def test_front_overhang_collision_is_detected(self):
        obstacle = CircleObstacle(7.1, 5.0, 0.2)
        clearance = self.planner._vehicle_obstacle_clearance(5.0, 5.0, 0.0, obstacle)
        self.assertLessEqual(clearance, 0.0)

    def test_collision_between_sparse_nodes_is_detected(self):
        self.planner.teb_nodes = [
            TEBNode(5.0, 5.0, 0.0, 0.1),
            TEBNode(15.0, 5.0, 0.0, 0.1),
        ]
        collision = self.planner._check_trajectory_collision(
            [CircleObstacle(10.0, 5.0, 0.5)],
            max_path_distance=10.0,
        )
        self.assertIsNotNone(collision)
        self.assertEqual(collision[0], 0)

    def test_distant_collision_does_not_trigger_immediate_stop(self):
        self.planner.teb_nodes = [
            TEBNode(5.0, 5.0, 0.0, 0.1),
            TEBNode(15.0, 5.0, 0.0, 0.1),
        ]
        collision = self.planner._check_trajectory_collision(
            [CircleObstacle(12.0, 5.0, 0.5)],
            max_path_distance=3.0,
        )
        self.assertIsNone(collision)

    def test_clear_trajectory_is_accepted(self):
        self.planner.teb_nodes = [
            TEBNode(5.0, 5.0, 0.0, 0.1),
            TEBNode(15.0, 5.0, 0.0, 0.1),
        ]
        collision = self.planner._check_trajectory_collision(
            [CircleObstacle(10.0, 10.0, 0.5)]
        )
        self.assertIsNone(collision)


if __name__ == '__main__':
    unittest.main()
