import json
from pathlib import Path
import unittest


class PolicyManifestTests(unittest.TestCase):
    def test_policy_tensor_sizes_are_explicit(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "training/config/policy_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(item["size"] for item in manifest["inputs"]), manifest["observation_size"])
        self.assertEqual(sum(item["size"] for item in manifest["outputs"]), manifest["action_size"])
        self.assertEqual(manifest["status"], "UNAPPROVED_SHADOW_ONLY")
        self.assertIsNone(manifest["command_publish_rate_hz"])
