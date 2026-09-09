import unittest

from x2_stair_autonomy.bag_replay import matching_scene, validate_annotations


class BagReplayTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "schema_version": 1,
            "lidar_frame_approved": True,
            "imu_frame_approved": True,
            "scenes": [
                {"id": "approach", "start_ns": 100, "end_ns": 200, "expected": {"supported": True}}
            ],
        }

    def test_requires_approved_transforms(self):
        self.document["lidar_frame_approved"] = False
        with self.assertRaisesRegex(ValueError, "LiDAR-to-base"):
            validate_annotations(self.document)

    def test_timestamp_selects_scene(self):
        scenes = validate_annotations(self.document)
        self.assertEqual(matching_scene(scenes, 150)["id"], "approach")
        self.assertIsNone(matching_scene(scenes, 250))

    def test_overlapping_annotations_are_rejected(self):
        scenes = validate_annotations(self.document)
        scenes.append({"id": "overlap", "start_ns": 125, "end_ns": 175, "expected": {}})
        with self.assertRaisesRegex(ValueError, "overlap"):
            matching_scene(scenes, 150)


if __name__ == "__main__":
    unittest.main()
