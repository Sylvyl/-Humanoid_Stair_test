import unittest

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class GaeTests(unittest.TestCase):
    def test_gae_matches_hand_computation(self):
        from training.x2_stairs.ppo_train import compute_gae

        # 1 env, 3 steps, gamma=lambda=1 (undiscounted): with value=0
        # everywhere, GAE(1) reduces to the plain reward-to-go.
        rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        dones = np.array([[False], [False], [False]])
        last_values = np.array([0.0], dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones, last_values, gamma=1.0, lambda_=1.0)
        np.testing.assert_allclose(advantages[:, 0], [3.0, 2.0, 1.0])
        np.testing.assert_allclose(returns[:, 0], [3.0, 2.0, 1.0])

    def test_done_cuts_advantage_propagation(self):
        from training.x2_stairs.ppo_train import compute_gae

        rewards = np.array([[1.0], [1.0]], dtype=np.float32)
        values = np.array([[0.0], [0.0]], dtype=np.float32)
        dones = np.array([[True], [False]])
        last_values = np.array([0.0], dtype=np.float32)
        advantages, _returns = compute_gae(rewards, values, dones, last_values, gamma=1.0, lambda_=1.0)
        # Step 0 is terminal: its advantage must not include step 1's reward.
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class ActorCriticTests(unittest.TestCase):
    def test_forward_and_act_shapes_respect_action_clip(self):
        from training.x2_stairs.ppo_train import ActorCritic

        model = ActorCritic(obs_size=144, action_size=29, action_clip=1.0)
        obs = torch.zeros(5, 144)
        mean, value = model.forward(obs)
        self.assertEqual(tuple(mean.shape), (5, 29))
        self.assertEqual(tuple(value.shape), (5,))

        action, log_prob, act_value = model.act(obs)
        self.assertEqual(tuple(action.shape), (5, 29))
        self.assertEqual(tuple(log_prob.shape), (5,))
        self.assertEqual(tuple(act_value.shape), (5,))
        self.assertTrue(bool(torch.all(action <= 1.0)) and bool(torch.all(action >= -1.0)))

    def test_evaluate_shapes(self):
        from training.x2_stairs.ppo_train import ActorCritic

        model = ActorCritic(obs_size=144, action_size=29, action_clip=1.0)
        obs = torch.zeros(3, 144)
        action = torch.zeros(3, 29)
        log_prob, entropy, value = model.evaluate(obs, action)
        self.assertEqual(tuple(log_prob.shape), (3,))
        self.assertEqual(tuple(entropy.shape), (3,))
        self.assertEqual(tuple(value.shape), (3,))


if __name__ == "__main__":
    unittest.main()
