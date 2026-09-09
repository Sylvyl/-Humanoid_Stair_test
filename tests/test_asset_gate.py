import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training.x2_stairs.isaaclab_env import REQUIRED_REVIEWS, validated_x2_usd


class AssetGateTests(unittest.TestCase):
    def test_missing_asset_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "X2_VALIDATED_USD is unset"):
                validated_x2_usd()

    def test_complete_hashed_approval_passes(self):
        with TemporaryDirectory() as directory:
            usd = Path(directory) / "x2.usd"
            usd.write_bytes(b"approved-test-usd")
            approval = {
                "approved": True, "approval_reference": "AGI-ASSET-1",
                "sdk": "aimdk v1.0.0-ga424add",
                "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
                "joint_order": [f"joint_{index}" for index in range(29)],
                "reviews": {name: True for name in REQUIRED_REVIEWS},
            }
            usd.with_suffix(".usd.vendor-approved.json").write_text(json.dumps(approval), encoding="utf-8")
            with patch.dict(os.environ, {"X2_VALIDATED_USD": str(usd)}):
                self.assertEqual(validated_x2_usd(), usd)


if __name__ == "__main__":
    unittest.main()
