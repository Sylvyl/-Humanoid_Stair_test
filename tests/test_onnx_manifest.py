import unittest

from training.x2_stairs.validate_onnx import manifest_blockers


class OnnxManifestTests(unittest.TestCase):
    def test_incomplete_contract_is_blocked(self):
        blockers = manifest_blockers({"status": "DRAFT"}, {"approved": False})
        self.assertIn("manifest status is not VENDOR_APPROVED", blockers)
        self.assertIn("vendor approval is not approved", blockers)

    def test_complete_static_contract_passes(self):
        manifest = {
            "status": "VENDOR_APPROVED",
            "joint_order": ["left", "right"],
            "action_size": 2,
            "observation_size": 3,
            "inputs": [{"name": "observation", "size": 3}],
            "outputs": [{"name": "action", "size": 2}],
            "command_publish_rate_hz": 100,
            "action_scale": [0.1, 0.1],
        }
        approval = {"approved": True, "command_rate_hz": 100}
        self.assertEqual(manifest_blockers(manifest, approval), [])


if __name__ == "__main__":
    unittest.main()
