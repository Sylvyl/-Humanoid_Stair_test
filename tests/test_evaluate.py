import os
import unittest

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_HAS_SDK = bool(os.environ.get("AIMDK_SDK_ROOT"))


@unittest.skipUnless(_HAS_DEPS and _HAS_SDK, "requires mujoco, torch, and AIMDK_SDK_ROOT")
class EvaluateTests(unittest.TestCase):
    def test_report_schema_and_bounds(self):
        from training.x2_stairs.evaluate import evaluate
        from training.x2_stairs.ppo_train import ActorCritic
        from x2_stair_autonomy.policy_contract import PolicyContract
        from pathlib import Path

        manifest_path = Path(__file__).resolve().parents[1] / "training" / "config" / "policy_manifest.json"
        contract = PolicyContract.from_path(manifest_path)
        model = ActorCritic(contract.observation_size, contract.action_size, contract.action_clip)

        report = evaluate(model, contract, "single_step_up", episodes=3, max_steps=20, base_seed=0)

        self.assertEqual(report["episodes"], 3)
        self.assertGreaterEqual(report["success_rate"], 0.0)
        self.assertLessEqual(report["success_rate"], 1.0)
        self.assertIn("up", report["by_direction"])
        self.assertEqual(report["by_direction"]["up"]["episodes"], 3)
        # single_step_up always forces direction "up" -- never mixes in "down".
        self.assertNotIn("down", report["by_direction"])


if __name__ == "__main__":
    unittest.main()
