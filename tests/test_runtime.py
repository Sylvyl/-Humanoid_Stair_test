import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from x2_stair_autonomy.runtime import RuntimeState


class RuntimeTests(unittest.TestCase):
    def test_training_runs_and_compatibility_are_loaded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "training").mkdir()
            (root / "validation").mkdir()
            (root / "config/stair_profiles.json").write_text(json.dumps({"profiles": [{
                "profile_id": "test", "min_riser_m": 0.15, "max_riser_m": 0.19,
                "min_tread_m": 0.25, "max_tread_m": 0.32, "min_width_m": 0.9,
                "min_landing_m": 0.9, "min_confidence": 0.8, "approved": False,
            }]}), encoding="utf-8")
            (root / "training/runs.json").write_text('[{"id":"smoke"}]', encoding="utf-8")
            (root / "validation/pc2_compatibility.json").write_text(json.dumps({
                "passed_shadow": True, "passed_hardware": True,
                "roles": {"power": "/pmu"}, "topics": {"/pmu": ["PmuState"]},
            }), encoding="utf-8")
            state = RuntimeState(root)
            self.assertEqual(state.training_runs[0]["id"], "smoke")
            self.assertEqual(state.compatibility_summary()["roles"]["power"], "/pmu")
            self.assertTrue(any(reason.startswith("release gate:") for reason in state.refresh_safety().blockers()))


if __name__ == "__main__":
    unittest.main()
