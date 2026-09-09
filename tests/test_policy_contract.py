import json
from pathlib import Path
import unittest

import numpy as np

from x2_stair_autonomy.policy_contract import PolicyContract


class PolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.manifest = json.loads((root / "training/config/policy_manifest.json").read_text(encoding="utf-8"))

    def values(self):
        return {item["name"]: np.zeros(item["size"], dtype=np.float32) for item in self.manifest["inputs"]}

    def test_manifest_order_is_preserved(self):
        contract = PolicyContract(self.manifest)
        values = self.values()
        values[self.manifest["inputs"][0]["name"]][0] = 2.0
        self.assertEqual(contract.observation(values)[0], 2.0)

    def test_locked_or_stale_returns_deterministic_zero(self):
        contract = PolicyContract(self.manifest)
        runner = lambda _: np.ones((1, contract.action_size), dtype=np.float32)
        result = contract.infer(runner, self.values(), sensors_fresh=True, release_unlocked=False, heartbeat_fresh=True)
        self.assertFalse(result.valid)
        np.testing.assert_array_equal(result.action, np.zeros(contract.action_size))

    def test_malformed_and_nonfinite_actions_return_zero(self):
        contract = PolicyContract(self.manifest)
        values = self.values()
        values[contract.inputs[0]["name"]][0] = np.nan
        result = contract.infer(lambda _: np.ones(contract.action_size), values,
                                sensors_fresh=True, release_unlocked=True, heartbeat_fresh=True)
        self.assertFalse(result.valid)
        np.testing.assert_array_equal(result.action, np.zeros(contract.action_size))

    def test_valid_action_is_clipped(self):
        contract = PolicyContract(self.manifest)
        result = contract.infer(lambda _: np.full((1, contract.action_size), 4.0), self.values(),
                                sensors_fresh=True, release_unlocked=True, heartbeat_fresh=True)
        self.assertTrue(result.valid)
        self.assertTrue(np.all(result.action == contract.action_clip))


if __name__ == "__main__":
    unittest.main()
