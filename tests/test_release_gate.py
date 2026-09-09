import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from x2_stair_autonomy.release_gate import evaluate_release_gate


def write_json(root: Path, relative: str, value: dict):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReleaseGateTests(unittest.TestCase):
    def test_missing_evidence_fails_closed(self):
        with TemporaryDirectory() as directory:
            report = evaluate_release_gate(Path(directory))
            self.assertFalse(report.unlocked)
            self.assertIn("vendor approval record is missing or unapproved", report.blockers)
            self.assertIn("stair policy ONNX artifact is missing", report.blockers)

    def test_suspended_stage_unlocks_only_with_complete_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "training/artifacts/stair_policy.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"test-policy")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            write_json(root, "config/vendor_approval.json", {
                "approved": True, "sdk": "aimdk v1.0.0-ga424add",
                "joint_limits_reviewed": True, "stop_matrix_reviewed": True,
                "command_rate_hz": 500, "onnx_contract_sha256": digest,
                "approval_reference": "AGI-1", "robot_serial": "X2-TEST",
                "firmware": "approved", "approved_at": "2026-01-01T00:00:00Z",
                "approved_by": "AgiBot",
            })
            write_json(root, "training/config/policy_manifest.json", {
                "status": "VENDOR_APPROVED", "sdk": "aimdk v1.0.0-ga424add",
                "action_size": 2, "observation_size": 3,
                "inputs": [{"name": "obs", "size": 3}],
                "outputs": [{"name": "action", "size": 2}],
                "joint_order": ["j0", "j1"], "action_scale": 0.1,
                "command_publish_rate_hz": 500,
            })
            write_json(root, "config/stair_profiles.json", {"profiles": [{"approved": True}]})
            write_json(root, "validation/pc2_compatibility.json", {
                "passed_hardware": True, "host": {"architecture": "aarch64"},
                "environment": {"ros_distro": "humble"},
            })
            write_json(root, "validation/perception_report.json", {"passed": True})
            write_json(root, "validation/simulation_report.json", {
                "passed": True, "up": {"episodes": 1000, "success_rate": 0.95},
                "down": {"episodes": 1000, "success_rate": 0.96},
                "fault_stop_passed": True,
            })
            write_json(root, "validation/onnx_report.json", {
                "passed": True, "mujoco_closed_loop_passed": True,
            })
            signed = {name: {"passed": True, "operator": "op", "spotter": "spot", "reviewed_at": "2026-01-01T00:00:00Z"}
                      for name in ("read_only_pc2", "shadow_replay")}
            write_json(root, "validation/hardware_stages.json", {"stages": signed})
            self.assertTrue(evaluate_release_gate(root).unlocked)

    def test_next_stage_requires_suspended_stage_signature(self):
        with TemporaryDirectory() as directory:
            report = evaluate_release_gate(Path(directory), "gantry_flat_ground")
            self.assertIn("prior hardware stage is unsigned: suspended_joint_output", report.blockers)


if __name__ == "__main__":
    unittest.main()
