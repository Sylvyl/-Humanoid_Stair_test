"""Real AgiBot X2 joint facts, cited to their exact AimDK SDK source.

Every number here is transcribed from the vendor's own reference RL-deployment
example, not guessed. Sources (relative to ``AIMDK_SDK_ROOT``):

- ``extra/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml``:
  ``seq``/``action_seq`` (joint order), ``default_dof_pos``, ``kps``, ``kds``,
  ``dt``, ``action_scale``.
- ``extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/
  model_info/x2.xml``: per-joint ``range`` (position limit) and
  ``actuatorfrcrange`` (torque limit), and the foot/pelvis body names.

``default_dof_pos``/``kps``/``kds`` are flat 29-element arrays in the YAML;
this module assumes they are positionally aligned with ``seq`` (the standard
convention for this style of legged-robot deploy config). That alignment is
not independently vendor-confirmed and should be spot-checked against a real
robot/telemetry before any of these numbers are relied on for anything beyond
local simulation.

``ACTION_SCALE_REFERENCE`` is the vendor *example*'s scalar residual-action
scale. It is a reasonable default for our own MuJoCo sim, but the actual
action scale for a trained stair policy remains a training decision recorded
in ``training/config/policy_manifest.json`` (``action_scale``), which stays
``null`` until real training and vendor approval happen.

This module never touches the robot: it only exposes static facts and file
paths. It is imported solely by training/simulation tooling.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- extra/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml: seq / action_seq ---
JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

# --- motion_control.yaml: default_dof_pos (positionally aligned with `seq`) ---
_DEFAULT_DOF_POS = (
    -0.3120, -0.3120, 0.0000,
    0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000,
    0.6690, 0.6690,
    0.2000, 0.2000,
    -0.3630, -0.3630,
    0.2000, -0.2000,
    0.0000, 0.0000,
    0.0000, 0.0000,
    -0.3000, -0.3000,
    0.0000, 0.0000,
    0.0000, 0.0000,
    0.0000, 0.0000,
)

# --- motion_control.yaml: kps / kds (positionally aligned with `seq`) ---
_KP = (
    120.0000, 120.0000, 40.1792, 100.0000, 100.0000, 200.0000, 100.0000,
    100.0000, 200.0000, 150.0000, 150.0000, 50.0000, 50.0000, 40.0000,
    40.0000, 50.0000, 50.0000, 40.0000, 40.0000, 50.0000, 50.0000,
    50.0000, 50.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000,
)
_KD = (
    5.0000, 5.0000, 2.5579, 4.0000, 4.0000, 2.0000, 4.0000, 4.0000, 2.0000,
    5.0000, 5.0000, 3.0000, 3.0000, 2.0000, 2.0000, 3.0000, 3.0000, 2.0000,
    2.0000, 3.0000, 3.0000, 3.0000, 3.0000, 2.0000, 2.0000, 2.0000, 2.0000,
    2.0000, 2.0000,
)

assert len(JOINT_ORDER) == 29 == len(_DEFAULT_DOF_POS) == len(_KP) == len(_KD)
assert len(set(JOINT_ORDER)) == 29, "joint order must not contain duplicates"

DEFAULT_DOF_POS: dict[str, float] = dict(zip(JOINT_ORDER, _DEFAULT_DOF_POS))
KP: dict[str, float] = dict(zip(JOINT_ORDER, _KP))
KD: dict[str, float] = dict(zip(JOINT_ORDER, _KD))

CONTROL_DT_S = 0.02  # motion_control.yaml: rl_config.dt (50 Hz)
ACTION_SCALE_REFERENCE = 0.25  # motion_control.yaml: rl_config.action_seq action_scale

# --- x2.xml: <joint ... range="lo hi" actuatorfrcrange="-t t"/> (not a clean
# left/right sign-mirror for every joint, e.g. ankle_roll/wrist_roll -- values
# below are transcribed explicitly rather than derived). ---
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-2.704, 2.556), "right_hip_pitch_joint": (-2.704, 2.556),
    "left_hip_roll_joint": (-0.235, 2.906), "right_hip_roll_joint": (-2.906, 0.235),
    "left_hip_yaw_joint": (-1.684, 3.43), "right_hip_yaw_joint": (-3.43, 1.684),
    "left_knee_joint": (0.0, 2.4073), "right_knee_joint": (0.0, 2.4073),
    "left_ankle_pitch_joint": (-0.803, 0.453), "right_ankle_pitch_joint": (-0.803, 0.453),
    "left_ankle_roll_joint": (-0.262, 0.262), "right_ankle_roll_joint": (-0.2625, 0.2625),
    "waist_yaw_joint": (-3.43, 2.382), "waist_pitch_joint": (-0.314, 0.314), "waist_roll_joint": (-0.488, 0.488),
    "left_shoulder_pitch_joint": (-3.08, 2.04), "right_shoulder_pitch_joint": (-3.08, 2.04),
    "left_shoulder_roll_joint": (-0.061, 2.993), "right_shoulder_roll_joint": (-2.993, 0.061),
    "left_shoulder_yaw_joint": (-2.556, 2.556), "right_shoulder_yaw_joint": (-2.556, 2.556),
    "left_elbow_joint": (-2.3556, 0.0), "right_elbow_joint": (-2.3556, 0.0),
    "left_wrist_yaw_joint": (-2.556, 2.556), "right_wrist_yaw_joint": (-2.556, 2.556),
    "left_wrist_pitch_joint": (-0.558, 0.558), "right_wrist_pitch_joint": (-0.558, 0.558),
    "left_wrist_roll_joint": (-1.571, 0.724), "right_wrist_roll_joint": (-0.724, 1.571),
}
JOINT_TORQUE_LIMITS_NM: dict[str, float] = {
    "left_hip_pitch_joint": 120.0, "right_hip_pitch_joint": 120.0,
    "left_hip_roll_joint": 120.0, "right_hip_roll_joint": 120.0,
    "left_hip_yaw_joint": 120.0, "right_hip_yaw_joint": 120.0,
    "left_knee_joint": 120.0, "right_knee_joint": 120.0,
    "left_ankle_pitch_joint": 36.0, "right_ankle_pitch_joint": 36.0,
    "left_ankle_roll_joint": 24.0, "right_ankle_roll_joint": 24.0,
    "waist_yaw_joint": 120.0, "waist_pitch_joint": 48.0, "waist_roll_joint": 48.0,
    "left_shoulder_pitch_joint": 36.0, "right_shoulder_pitch_joint": 36.0,
    "left_shoulder_roll_joint": 36.0, "right_shoulder_roll_joint": 36.0,
    "left_shoulder_yaw_joint": 24.0, "right_shoulder_yaw_joint": 24.0,
    "left_elbow_joint": 24.0, "right_elbow_joint": 24.0,
    "left_wrist_yaw_joint": 24.0, "right_wrist_yaw_joint": 24.0,
    "left_wrist_pitch_joint": 4.8, "right_wrist_pitch_joint": 4.8,
    "left_wrist_roll_joint": 4.8, "right_wrist_roll_joint": 4.8,
}
assert set(JOINT_LIMITS_RAD) == set(JOINT_TORQUE_LIMITS_NM) == set(JOINT_ORDER)

# x2.xml: <body name="pelvis" .../> is the floating base (<freejoint/>); the two
# terminal leg bodies are the only surfaces allowed to contact terrain.
BASE_BODY = "pelvis"
FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

_ROBOT_MODEL_ID = "lx2501_3_t2d5"
_SDK_ROOT_ENV = "AIMDK_SDK_ROOT"


def resolve_sdk_path(*parts: str) -> Path:
    """Resolve a path inside the extracted AimDK SDK artifacts.

    Mirrors the ``X2_VALIDATED_USD``-style pattern used by ``isaaclab_env.py``:
    an external, reviewed asset is referenced by environment variable and never
    copied into this repository. Raises ``RuntimeError`` (fail closed) when the
    variable is unset or the path doesn't exist.
    """
    root = os.environ.get(_SDK_ROOT_ENV)
    if not root:
        raise RuntimeError(
            f"{_SDK_ROOT_ENV} is unset; point it at the extracted "
            "aimdk-aarch64-*-artifacts directory to use SDK-derived simulation assets"
        )
    path = Path(root).joinpath(*parts)
    if not path.exists():
        raise RuntimeError(f"expected AimDK SDK path does not exist: {path}")
    return path


def x2_mjcf_path() -> Path:
    """Path to the vendor's real X2 MJCF robot model (``x2.xml``)."""
    return resolve_sdk_path(
        "extra", "x2_rl_deploy", "x2_rl_deploy_mujoco", "configuration", "robot",
        _ROBOT_MODEL_ID, "model_info", "x2.xml",
    )
