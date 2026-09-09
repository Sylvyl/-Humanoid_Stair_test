import numpy as np
import unittest

from x2_stair_autonomy.models import StairProfile
from x2_stair_autonomy.perception import depth_to_points, estimate_stairs


def staircase(direction="up", riser=0.17, tread=0.28, width=1.1, steps=6):
    rng = np.random.default_rng(11)
    rows = []
    for index in range(steps + 1):
        z = index * riser if direction == "up" else (steps - index) * riser
        x0 = 0.35 + index * tread
        xs = rng.uniform(x0, x0 + tread * 0.92, 220)
        ys = rng.uniform(-width / 2, width / 2, 220)
        zs = rng.normal(z, 0.002, 220)
        rows.append(np.column_stack((xs, ys, zs)))
    return np.vstack(rows)


class PerceptionTests(unittest.TestCase):
    def test_estimates_supported_ascent(self):
        result = estimate_stairs(staircase("up"))
        self.assertTrue(result.supported)
        self.assertEqual(result.direction, "up")
        self.assertLessEqual(abs(result.riser_m - 0.17), 0.01)
        self.assertLessEqual(abs(result.tread_m - 0.28), 0.015)
        self.assertGreaterEqual(result.width_m, 0.9)
        self.assertGreaterEqual(result.confidence, 0.8)

    def test_estimates_supported_descent(self):
        result = estimate_stairs(staircase("down"))
        self.assertTrue(result.supported)
        self.assertEqual(result.direction, "down")

    def test_rejects_out_of_profile_geometry(self):
        result = estimate_stairs(staircase("up", riser=0.23))
        self.assertTrue(result.available)
        self.assertFalse(result.supported)
        self.assertIn("outside", result.rejection_reason)

    def test_rejects_small_cloud(self):
        result = estimate_stairs(np.zeros((10, 3)), StairProfile())
        self.assertFalse(result.available)
        self.assertIn("insufficient", result.rejection_reason)

    def test_depth_projection_requires_explicit_transform(self):
        depth = np.ones((4, 4), dtype=float)
        points = depth_to_points(depth, fx=2, fy=2, cx=1.5, cy=1.5,
                                 camera_to_base=np.eye(4), stride=2)
        self.assertEqual(points.shape, (4, 3))
        with self.assertRaises(ValueError):
            depth_to_points(depth, fx=0, fy=2, cx=0, cy=0,
                            camera_to_base=np.eye(4))
