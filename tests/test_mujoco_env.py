import os
import unittest

import numpy as np

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

_HAS_SDK = bool(os.environ.get("AIMDK_SDK_ROOT"))


@unittest.skipUnless(_HAS_MUJOCO and _HAS_SDK, "requires `mujoco` and AIMDK_SDK_ROOT")
class StairMujocoEnvTests(unittest.TestCase):
    def setUp(self):
        from training.x2_stairs.mujoco_env import StairMujocoEnv
        self.env = StairMujocoEnv()

    def test_reset_observation_matches_manifest_contract(self):
        values = self.env.reset(seed=1)
        flat = self.env.contract.observation(values)
        self.assertEqual(flat.shape, (self.env.contract.observation_size,))
        self.assertTrue(np.isfinite(flat).all())

    def test_step_returns_finite_reward_and_valid_termination(self):
        self.env.reset(seed=2)
        action = np.zeros(self.env.contract.action_size, dtype=np.float32)
        result = self.env.step(action)
        self.assertTrue(np.isfinite(result.reward))
        self.assertIn(
            result.termination,
            (None, "base_contact", "non_foot_collision", "excessive_tilt",
             "joint_limit", "stale_observation", "stair_profile_violation"),
        )

    def test_force_stale_termination_hook(self):
        self.env.reset(seed=3)
        self.env.force_stale_termination = True
        result = self.env.step(np.zeros(self.env.contract.action_size, dtype=np.float32))
        self.assertEqual(result.termination, "stale_observation")
        self.assertTrue(result.done)

    def test_reset_is_reproducible_for_a_given_seed(self):
        first = self.env.reset(seed=42)
        second = self.env.reset(seed=42)
        self.assertEqual(self.env.direction, self.env.task_spec.sample_domain(42)["direction"])
        np.testing.assert_allclose(first["joint_position_error"], second["joint_position_error"])


if __name__ == "__main__":
    unittest.main()
