import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import mujoco  # noqa: F401
    import onnxruntime  # noqa: F401
    import onnx
    from onnx import TensorProto, helper
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_HAS_SDK = bool(os.environ.get("AIMDK_SDK_ROOT"))


def _write_zero_policy(path: Path, obs_size: int, action_size: int) -> None:
    """A trivial all-zero-weight ONNX policy, just to exercise the pipeline."""
    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [obs_size, action_size], [0.0] * (obs_size * action_size)
    )
    node = helper.make_node("MatMul", ["observation", "W"], ["action"])
    graph = helper.make_graph(
        [node],
        "zero_policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, obs_size])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_size])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


@unittest.skipUnless(_HAS_DEPS and _HAS_SDK, "requires mujoco, onnxruntime, onnx, and AIMDK_SDK_ROOT")
class MujocoClosedLoopTests(unittest.TestCase):
    def test_zero_policy_report_is_well_formed(self):
        from training.x2_stairs.mujoco_closed_loop import _make_runner, run_episode
        from training.x2_stairs.mujoco_env import StairMujocoEnv
        from x2_stair_autonomy.policy_contract import PolicyContract

        manifest_path = Path(__file__).resolve().parents[1] / "training" / "config" / "policy_manifest.json"
        contract = PolicyContract.from_path(manifest_path)
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "zero_policy.onnx"
            _write_zero_policy(model_path, contract.observation_size, contract.action_size)
            runner = _make_runner(model_path)
            env = StairMujocoEnv()

            episode = run_episode(env, contract, runner, seed=1, max_steps=20)
            self.assertIn("termination", episode)
            self.assertGreaterEqual(episode["limit_violations"], 0)

            fault_episode = run_episode(env, contract, runner, seed=2, max_steps=20, fault_at_step=5)
            self.assertTrue(fault_episode["fault_zeroed_action"])


if __name__ == "__main__":
    unittest.main()
