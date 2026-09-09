import unittest

from x2_stair_autonomy.calibration import evaluate_scenes
from x2_stair_autonomy.models import StairProfile


def stair_points(direction="up", riser=0.17, tread=0.28, width=1.0):
    points = []
    sign = 1 if direction == "up" else -1
    for step in range(5):
        z = sign * step * riser
        x0 = 0.45 + step * tread
        for xi in range(12):
            for yi in range(14):
                points.append([x0 + xi * tread / 12, -width / 2 + yi * width / 13, z])
    return points


class CalibrationTests(unittest.TestCase):
    def test_acceptance_requires_supported_and_rejection_scenes(self):
        profile = StairProfile(min_confidence=0.5)
        report = evaluate_scenes([
            {"id": "up", "points_xyz": stair_points(), "expected": {
                "supported": True, "direction": "up", "riser_m": 0.17,
                "tread_m": 0.28, "yaw_deg": 0.0,
            }},
            {"id": "empty", "points_xyz": [], "expected": {"supported": False}},
        ], profile)
        self.assertTrue(report["passed"], report["failures"])

    def test_wrong_direction_fails(self):
        profile = StairProfile(min_confidence=0.5)
        report = evaluate_scenes([
            {"id": "wrong", "points_xyz": stair_points(), "expected": {
                "supported": True, "direction": "down", "riser_m": 0.17,
                "tread_m": 0.28, "yaw_deg": 0.0,
            }},
            {"id": "empty", "points_xyz": [], "expected": {"supported": False}},
        ], profile)
        self.assertFalse(report["passed"])
        self.assertTrue(any("direction mismatch" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()
