import unittest

from x2_stair_autonomy.recording import recording_topics


class RecordingTests(unittest.TestCase):
    def test_missing_required_sensor_rejected(self):
        with self.assertRaisesRegex(ValueError, "joints"):
            recording_topics({"roles": {"depth": "/d", "rgb": "/r", "lidar": "/l", "imu": "/i"}})

    def test_optional_power_is_not_fabricated(self):
        inventory = {
            "roles": {"depth": "/d", "rgb": "/r", "lidar": "/l", "imu": "/i", "joints": "/j", "power": None},
            "topics": {"/x2/stairs/safety": ["x2_stair_interfaces/msg/SafetyStatus"]},
        }
        topics = recording_topics(inventory)
        self.assertEqual(topics[:5], ["/d", "/r", "/l", "/i", "/j"])
        self.assertIn("/x2/stairs/safety", topics)
        self.assertNotIn("/aima/hal/pmu/state", topics)


if __name__ == "__main__":
    unittest.main()
