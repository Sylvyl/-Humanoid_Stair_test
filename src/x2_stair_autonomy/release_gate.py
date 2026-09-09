from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SDK_BASELINE = "aimdk v1.0.0-ga424add"
HARDWARE_STAGES = (
    "read_only_pc2",
    "shadow_replay",
    "suspended_joint_output",
    "gantry_flat_ground",
    "isolated_step",
    "three_step_rig",
    "known_flight",
    "three_standard_staircases",
)


@dataclass(frozen=True, slots=True)
class GateReport:
    target_stage: str
    unlocked: bool
    blockers: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_release_gate(root: Path, target_stage: str = "suspended_joint_output") -> GateReport:
    if target_stage not in HARDWARE_STAGES:
        raise ValueError(f"unknown hardware stage: {target_stage}")

    target_index = HARDWARE_STAGES.index(target_stage)
    approval = _json(root / "config" / "vendor_approval.json")
    manifest = _json(root / "training" / "config" / "policy_manifest.json")
    perception = _json(root / "validation" / "perception_report.json")
    simulation = _json(root / "validation" / "simulation_report.json")
    parity = _json(root / "validation" / "onnx_report.json")
    compatibility = _json(root / "validation" / "pc2_compatibility.json")
    stages = _json(root / "validation" / "hardware_stages.json")
    profiles = _json(root / "config" / "stair_profiles.json")
    model = root / "training" / "artifacts" / "stair_policy.onnx"
    model_hash = _file_sha256(model)
    configured_profiles = profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []
    blockers: list[str] = []

    # Read-only work has no policy or vendor gate. Every output-capable stage does.
    if target_index >= HARDWARE_STAGES.index("suspended_joint_output"):
        if not (
            compatibility.get("passed_hardware") is True
            and compatibility.get("host", {}).get("architecture") in {"aarch64", "arm64"}
            and compatibility.get("environment", {}).get("ros_distro") == "humble"
        ):
            blockers.append("live PC2 compatibility report has not passed")
        if not configured_profiles or configured_profiles[0].get("approved") is not True:
            blockers.append("stair profile is not approved")
        if approval.get("approved") is not True:
            blockers.append("vendor approval record is missing or unapproved")
        for identity_field in ("approval_reference", "robot_serial", "firmware", "approved_at", "approved_by"):
            if not approval.get(identity_field):
                blockers.append(f"vendor approval field is missing: {identity_field}")
        if approval.get("sdk") != SDK_BASELINE:
            blockers.append("vendor approval SDK does not match the AimDK baseline")
        if not approval.get("joint_limits_reviewed"):
            blockers.append("vendor joint limits/order are not reviewed")
        if not approval.get("stop_matrix_reviewed"):
            blockers.append("vendor stop matrix is not reviewed")
        if not isinstance(approval.get("command_rate_hz"), (int, float)) or approval.get("command_rate_hz", 0) <= 0:
            blockers.append("vendor command rate is missing")
        if manifest.get("status") != "VENDOR_APPROVED":
            blockers.append("policy manifest is not VENDOR_APPROVED")
        if manifest.get("sdk") != SDK_BASELINE:
            blockers.append("policy manifest SDK does not match the AimDK baseline")
        joint_order = manifest.get("joint_order", [])
        if not isinstance(joint_order, list) or any(not isinstance(name, str) or not name for name in joint_order):
            joint_order = []
        if len(joint_order) != manifest.get("action_size") or len(set(joint_order)) != len(joint_order):
            blockers.append("policy joint order is incomplete")
        if sum(item.get("size", 0) for item in manifest.get("inputs", [])) != manifest.get("observation_size"):
            blockers.append("policy observation tensor contract is inconsistent")
        if sum(item.get("size", 0) for item in manifest.get("outputs", [])) != manifest.get("action_size"):
            blockers.append("policy action tensor contract is inconsistent")
        if manifest.get("command_publish_rate_hz") != approval.get("command_rate_hz"):
            blockers.append("policy and vendor command rates do not match")
        action_scale = manifest.get("action_scale")
        scalar_scale = isinstance(action_scale, (int, float)) and action_scale > 0
        vector_scale = (
            isinstance(action_scale, list)
            and len(action_scale) == manifest.get("action_size")
            and all(isinstance(value, (int, float)) and value > 0 for value in action_scale)
        )
        if not (scalar_scale or vector_scale):
            blockers.append("policy action scale is missing")
        if model_hash is None:
            blockers.append("stair policy ONNX artifact is missing")
        elif model_hash != approval.get("onnx_contract_sha256"):
            blockers.append("ONNX artifact hash does not match vendor approval")
        if perception.get("passed") is not True:
            blockers.append("perception/replay acceptance report has not passed")
        if not (
            simulation.get("passed") is True
            and simulation.get("up", {}).get("episodes", 0) >= 1000
            and simulation.get("down", {}).get("episodes", 0) >= 1000
            and simulation.get("up", {}).get("success_rate", 0) >= 0.95
            and simulation.get("down", {}).get("success_rate", 0) >= 0.95
            and simulation.get("fault_stop_passed") is True
        ):
            blockers.append("simulation acceptance thresholds have not passed")
        if not (parity.get("passed") is True and parity.get("mujoco_closed_loop_passed") is True):
            blockers.append("ONNX parity/MuJoCo report has not passed")

    completed = stages.get("stages", {}) if isinstance(stages.get("stages"), dict) else {}
    for prior_stage in HARDWARE_STAGES[:target_index]:
        record = completed.get(prior_stage, {})
        if not (
            isinstance(record, dict)
            and record.get("passed") is True
            and record.get("operator")
            and record.get("spotter")
            and record.get("reviewed_at")
        ):
            blockers.append(f"prior hardware stage is unsigned: {prior_stage}")

    evidence = {
        "sdk": SDK_BASELINE,
        "policy_sha256": model_hash,
        "vendor_approval": bool(approval.get("approved")),
        "pc2_compatibility": bool(compatibility.get("passed_hardware")),
        "stair_profile_approved": bool(configured_profiles and configured_profiles[0].get("approved")),
        "perception_report": bool(perception.get("passed")),
        "simulation_report": bool(simulation.get("passed")),
        "onnx_report": bool(parity.get("passed")),
        "signed_prior_stages": [name for name in HARDWARE_STAGES[:target_index] if completed.get(name, {}).get("passed") is True],
    }
    return GateReport(target_stage, not blockers, tuple(blockers), evidence)
