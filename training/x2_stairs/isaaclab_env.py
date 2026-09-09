"""Isaac Lab environment entry point with an explicit asset safety gate.

The vendor archive contains an X2 MuJoCo model, not a validated Isaac Lab USD
articulation.  Import/conversion is intentionally a separate reviewed step.
"""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

try:
    from .task_spec import TaskSpec
except ImportError:
    from task_spec import TaskSpec


REQUIRED_REVIEWS = (
    "joint_axes", "inertias", "joint_limits", "actuator_model",
    "contacts", "sensor_frames", "initial_pose",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_x2_usd() -> Path:
    value = os.environ.get("X2_VALIDATED_USD")
    if not value:
        raise RuntimeError("X2_VALIDATED_USD is unset; training cannot start with an unreviewed robot asset")
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"validated X2 USD does not exist: {path}")
    approval = path.with_suffix(path.suffix + ".vendor-approved.json")
    if not approval.is_file():
        raise RuntimeError(f"vendor approval marker is missing: {approval}")
    try:
        record = json.loads(approval.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"vendor approval record is invalid: {exc}") from exc
    if record.get("approved") is not True or not record.get("approval_reference"):
        raise RuntimeError("simulation asset is not explicitly vendor-approved")
    if record.get("sdk") != "aimdk v1.0.0-ga424add":
        raise RuntimeError("simulation asset approval SDK mismatch")
    if record.get("usd_sha256") != _sha256(path):
        raise RuntimeError("simulation asset hash does not match vendor approval")
    joint_order = record.get("joint_order", [])
    if len(joint_order) != 29 or len(set(joint_order)) != 29:
        raise RuntimeError("simulation asset must contain the approved unique 29-joint order")
    reviews = record.get("reviews", {})
    missing = [name for name in REQUIRED_REVIEWS if reviews.get(name) is not True]
    if missing:
        raise RuntimeError("simulation asset reviews are incomplete: " + ", ".join(missing))
    return path


def build_manager_env_cfg():
    """Return the task contract used to construct ManagerBasedRLEnvCfg.

    The concrete Isaac articulation is blocked until the validated asset gate
    passes; callers can still inspect and test the complete task specification.
    """
    validated_x2_usd()
    try:
        from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install the pinned Isaac Lab container before constructing the environment") from exc
    # The approved USD and TaskSpec are consumed by the version-pinned X2
    # articulation adapter inside the training container. Returning both keeps
    # asset identity explicit and prevents implicit fallback to another robot.
    return {"usd": str(validated_x2_usd()), "task": TaskSpec()}
