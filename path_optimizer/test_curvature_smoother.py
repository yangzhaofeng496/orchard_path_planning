import math
import unittest

from path_optimizer.curvature_smoother import CurvatureSmoother


class FreeSpace:
    def check_line(self, _first, _second):
        return True


class CurvatureSmootherTest(unittest.TestCase):
    def setUp(self):
        self.smoother = CurvatureSmoother(
            collision_checker=FreeSpace(),
            max_curvature=0.23,
            interpolation_spacing=0.10,
            corner_blend_distance=2.0,
            max_speed=2.0,
            max_accel=1.0,
            max_decel=1.2,
            start_speed=0.0,
            end_speed=0.0,
        )

    def test_right_angle_respects_curvature(self):
        result = self.smoother.smooth([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)])
        self.assertEqual(result.points[0], (0.0, 0.0))
        self.assertAlmostEqual(result.points[-1][0], 5.0)
        self.assertAlmostEqual(result.points[-1][1], 5.0)
        self.assertEqual(len(result.corner_indices), 1)
        self.assertLessEqual(max(map(abs, result.curvatures)), 0.23 + 1e-3)

    def test_velocity_profile_respects_longitudinal_limits(self):
        result = self.smoother.smooth([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)])
        self.assertEqual(result.speeds[0], 0.0)
        self.assertEqual(result.speeds[-1], 0.0)
        for i in range(1, len(result.points)):
            ds = math.dist(result.points[i - 1], result.points[i])
            self.assertLessEqual(
                result.speeds[i] ** 2 - result.speeds[i - 1] ** 2,
                2.0 * self.smoother.max_accel * ds + 1e-8,
            )
            self.assertLessEqual(
                result.speeds[i - 1] ** 2 - result.speeds[i] ** 2,
                2.0 * self.smoother.max_decel * ds + 1e-8,
            )

    def test_straight_path_remains_straight(self):
        result = self.smoother.smooth([(0.0, 0.0), (10.0, 0.0)])
        self.assertEqual(result.corner_indices, [])
        self.assertTrue(all(abs(point[1]) < 1e-12 for point in result.points))
        self.assertTrue(all(abs(kappa) < 1e-12 for kappa in result.curvatures))


if __name__ == '__main__':
    unittest.main()
