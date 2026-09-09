"""Local MuJoCo stair environment for the AgiBot X2 Ultra.

Built on the vendor's real X2 robot model (`robot_reference.x2_mjcf_path()`)
and the residual-position-PD control convention used by AgiBot's own on-robot
RL controller (`extra/x2_rl_deploy/x2_rl_deploy_controller/src/
motion_control_node.cc`): ``target = default_dof_pos + action * action_scale``
tracked through per-joint PD gains, matching the existing
``PolicyContract`` residual-action design in
``src/x2_stair_autonomy/policy_contract.py``.

This module only ever talks to a local MuJoCo physics instance. It is never
imported by the runtime dashboard/ROS bridge and cannot reach the real robot.
Requires ``pip install -r training/requirements-sim.txt`` and the
``AIMDK_SDK_ROOT`` environment variable (see ``robot_reference``).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _src  # noqa: F401  (adds src/ to sys.path)
from x2_stair_autonomy.models import StairProfile
from x2_stair_autonomy.policy_contract import PolicyContract

from . import robot_reference as ref
from . import terrain

try:
    from .task_spec import TaskSpec
except ImportError:  # pragma: no cover - script-style invocation
    from task_spec import TaskSpec

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "config" / "policy_manifest.json"


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Install `mujoco` (see training/requirements-sim.txt) to use StairMujocoEnv"
        ) from exc
    return mujoco


def gravity_projection(quat_wxyz) -> np.ndarray:
    """Projected-gravity vector in the base frame.

    Identical formula to the vendor's real on-robot controller:
    ``MotionControlNode::getGravityOrientation`` in
    ``x2_rl_deploy_controller/src/motion_control_node.cc`` (~lines 232-243),
    kept identical here for sim/real observation consistency.
    """
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


@dataclass
class StepResult:
    """``observation`` is the named-field Mapping the environment measured,
    e.g. ``{"joint_position_error": ..., "projected_gravity": ..., ...}`` --
    exactly the shape ``PolicyContract.observation()``/``PolicyContract.infer()``
    expect. It is deliberately *not* pre-flattened here so callers always go
    through the one real assembly path in ``policy_contract.py`` instead of a
    second, possibly-diverging copy of it."""

    observation: dict
    reward: float
    done: bool
    termination: str | None
    reward_terms: dict


class StairMujocoEnv:
    """One stair-climbing episode with a gym-style ``reset()``/``step()`` API."""

    def __init__(self, task_spec: TaskSpec | None = None, profile: StairProfile | None = None):
        self._mujoco = _require_mujoco()
        self._x2_xml = ref.x2_mjcf_path()
        self.task_spec = task_spec or TaskSpec()
        self.profile = profile or StairProfile()
        self.contract = PolicyContract.from_path(_MANIFEST_PATH)

        self.joint_order = list(ref.JOINT_ORDER)
        self.default_qpos = np.array([ref.DEFAULT_DOF_POS[j] for j in self.joint_order], dtype=np.float32)
        self.kp = np.array([ref.KP[j] for j in self.joint_order], dtype=np.float32)
        self.kd = np.array([ref.KD[j] for j in self.joint_order], dtype=np.float32)
        self.torque_limits = np.array([ref.JOINT_TORQUE_LIMITS_NM[j] for j in self.joint_order], dtype=np.float32)
        self.joint_limits_lo = np.array([ref.JOINT_LIMITS_RAD[j][0] for j in self.joint_order], dtype=np.float32)
        self.joint_limits_hi = np.array([ref.JOINT_LIMITS_RAD[j][1] for j in self.joint_order], dtype=np.float32)
        self.action_scale = ref.ACTION_SCALE_REFERENCE

        # Fault-injection test hook for mujoco_closed_loop.py; PolicyContract's
        # own sensors_fresh/heartbeat_fresh gate (already covered by
        # tests/test_policy_contract.py) is exercised separately at the
        # inference call site, not duplicated here.
        self.force_stale_termination = False

        self.model = None
        self.data = None
        self.layout = None
        self.domain = None
        self.direction = "up"
        self.previous_action = np.zeros(self.contract.action_size, dtype=np.float32)
        self._actuator_ids = None
        self._joint_qpos_adr = None
        self._joint_dof_adr = None
        self._foot_body_ids = None
        self._base_body_id = None
        self._prev_foot_pos = {}
        self._actuator_strength = 1.0
        self._progress_ref = 0.0

    # -- model construction ---------------------------------------------------
    def _build_model(self, layout: terrain.StairLayout):
        mujoco = self._mujoco
        scene_xml = (
            '<mujoco model="x2_stairs_scene">\n'
            '  <include file="x2.xml"/>\n'
            '  <worldbody>\n'
            '    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.3 0.5 0.6 1"/>\n'
            f"    {terrain.to_mjcf(layout)}\n"
            "  </worldbody>\n"
            "</mujoco>\n"
        )
        # Written alongside x2.xml so its own relative <mesh>/meshdir resolution
        # behaves exactly like the vendor's own working scene.xml.
        temp_path = self._x2_xml.parent / f"_x2_stairs_scene_{uuid.uuid4().hex}.xml"
        temp_path.write_text(scene_xml, encoding="utf-8")
        try:
            return mujoco.MjModel.from_xml_path(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    def _bind_indices(self) -> None:
        model = self.model
        self._joint_qpos_adr = np.array([model.joint(name).qposadr[0] for name in self.joint_order])
        self._joint_dof_adr = np.array([model.joint(name).dofadr[0] for name in self.joint_order])
        self._actuator_ids = np.array([model.actuator(f"motor_{name}").id for name in self.joint_order])
        self._foot_body_ids = [model.body(name).id for name in ref.FOOT_BODIES]
        self._base_body_id = model.body(ref.BASE_BODY).id

    # -- episode lifecycle ------------------------------------------------------
    def reset(self, seed: int, *, force_direction: str | None = None) -> dict:
        domain = self.task_spec.sample_domain(seed, force_direction=force_direction)
        self.domain = domain
        self.direction = domain["direction"]
        self.layout = terrain.build_layout(domain, self.profile)
        self.model = self._build_model(self.layout)
        self.data = self._mujoco.MjData(self.model)
        self._bind_indices()
        self._randomize_dynamics(domain)

        self.data.qpos[self._joint_qpos_adr] = self.default_qpos
        if self.direction == "up":
            base_x, base_quat = 0.15, (1.0, 0.0, 0.0, 0.0)
        else:
            base_x, base_quat = self.layout.end_x - 0.15, (0.0, 0.0, 0.0, 1.0)
        self.data.qpos[0] = base_x
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.68 + terrain.height_at(self.layout, base_x)
        self.data.qpos[3:7] = base_quat
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        self.previous_action = np.zeros(self.contract.action_size, dtype=np.float32)
        self.force_stale_termination = False
        self._prev_foot_pos = {body_id: np.array(self.data.xpos[body_id]) for body_id in self._foot_body_ids}
        self._progress_ref = self._progress_coordinate()
        return self._observation_values()

    def _randomize_dynamics(self, domain: dict) -> None:
        model = self.model
        model.geom_friction[:, 0] *= float(domain.get("friction", 1.0))
        model.body_mass[:] *= float(domain.get("mass_scale", 1.0))
        payload = float(domain.get("payload_kg", 0.0))
        if payload > 0:
            model.body_mass[self._base_body_id] += payload
        self._actuator_strength = float(domain.get("actuator_strength", 1.0))

    # -- physics step -------------------------------------------------------------
    def step(self, action) -> StepResult:
        mujoco = self._mujoco
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            -self.contract.action_clip, self.contract.action_clip,
        )
        target = self.default_qpos + action * self.action_scale
        substeps = max(1, round(ref.CONTROL_DT_S / self.model.opt.timestep))
        for _ in range(substeps):
            qpos = self.data.qpos[self._joint_qpos_adr]
            qvel = self.data.qvel[self._joint_dof_adr]
            torque = self._actuator_strength * (self.kp * (target - qpos) - self.kd * qvel)
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self._actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)

        termination = self._check_termination()
        reward, reward_terms = self._reward(action, termination)
        observation = self._observation_values(previous_action=action)
        self._prev_foot_pos = {body_id: np.array(self.data.xpos[body_id]) for body_id in self._foot_body_ids}
        self.previous_action = action
        return StepResult(observation, reward, termination is not None, termination, reward_terms)

    # -- observation --------------------------------------------------------------
    def _observation_values(self, previous_action: np.ndarray | None = None) -> dict:
        """Named observation fields matching ``policy_manifest.json``'s
        ``inputs``. Callers pass this straight to
        ``PolicyContract.observation()``/``PolicyContract.infer()``."""
        qpos = self.data.qpos[self._joint_qpos_adr]
        qvel = self.data.qvel[self._joint_dof_adr]
        return {
            "joint_position_error": qpos - self.default_qpos,
            "joint_velocity": qvel,
            "projected_gravity": gravity_projection(self.data.qpos[3:7]),
            "base_angular_velocity": self.data.qvel[3:6],
            "previous_action": self.previous_action if previous_action is None else previous_action,
            "velocity_command": self._velocity_command(),
            "terrain_height_samples": self._height_samples(),
        }

    def _velocity_command(self) -> np.ndarray:
        # "Forward" is always +x in the direction the robot is spawned facing
        # (toward the stairs), regardless of world-frame direction.
        return np.array([0.35, 0.0, 0.0], dtype=np.float32)

    def _height_samples(self) -> np.ndarray:
        base_x = float(self.data.qpos[0])
        base_z = float(self.data.qpos[2])
        forward_sign = 1.0 if self.direction == "up" else -1.0
        offsets = np.linspace(-0.2, 1.4, 8)
        samples = [
            terrain.height_at(self.layout, base_x + forward_sign * dx) - base_z
            for _lateral in range(6)
            for dx in offsets
        ]
        return np.array(samples, dtype=np.float32)

    # -- termination / contacts ----------------------------------------------------
    def _contacting_bodies(self) -> set[int]:
        bodies = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 == 0 and b2 != 0:
                bodies.add(b2)
            elif b2 == 0 and b1 != 0:
                bodies.add(b1)
        return bodies

    def _check_termination(self) -> str | None:
        if self.force_stale_termination:
            return "stale_observation"
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "excessive_tilt"
        gravity = gravity_projection(self.data.qpos[3:7])
        if gravity[2] > -0.5:  # more than ~60 degrees from upright
            return "excessive_tilt"
        qpos = self.data.qpos[self._joint_qpos_adr]
        margin = 0.02
        if np.any(qpos < self.joint_limits_lo - margin) or np.any(qpos > self.joint_limits_hi + margin):
            return "joint_limit"
        contacting = self._contacting_bodies()
        if self._base_body_id in contacting:
            return "base_contact"
        if contacting - set(self._foot_body_ids):
            return "non_foot_collision"
        profile, layout = self.profile, self.layout
        if not (
            profile.min_riser_m <= layout.riser_m <= profile.max_riser_m
            and profile.min_tread_m <= layout.tread_m <= profile.max_tread_m
            and layout.width_m >= profile.min_width_m
        ):
            return "stair_profile_violation"
        return None

    # -- reward -----------------------------------------------------------------
    def _progress_coordinate(self) -> float:
        x = float(self.data.qpos[0])
        return x if self.direction == "up" else -x

    def _on_landing(self) -> bool:
        x = float(self.data.qpos[0])
        if self.direction == "up":
            return x >= self.layout.landing.x_leading
        return x <= self.layout.steps[0].x_leading

    def reached_landing(self) -> bool:
        """Public alias of the landing check, for external success evaluation
        (e.g. ``evaluate.py``): the base has reached the far side of the
        stair (top landing for "up", the floor past the bottom step for
        "down")."""
        return self._on_landing()

    def _foot_clearance(self) -> float:
        total = 0.0
        for body_id in self._foot_body_ids:
            pos = self.data.xpos[body_id]
            ground = terrain.height_at(self.layout, float(pos[0]))
            total += min(max(float(pos[2]) - ground, 0.0), 0.15)
        return total

    def _slip(self, contacting: set[int]) -> float:
        total = 0.0
        for body_id in self._foot_body_ids:
            if body_id in contacting:
                delta = (np.array(self.data.xpos[body_id]) - self._prev_foot_pos[body_id]) / ref.CONTROL_DT_S
                total += float(np.linalg.norm(delta[:2]))
        return total

    def _reward(self, action: np.ndarray, termination: str | None) -> tuple[float, dict]:
        contacting = self._contacting_bodies()
        progress = self._progress_coordinate() - self._progress_ref
        self._progress_ref = self._progress_coordinate()
        gravity = gravity_projection(self.data.qpos[3:7])
        feet_down = sum(1 for b in self._foot_body_ids if b in contacting)
        # Rebalanced after this session's training runs first plateaued with
        # progress_along_stairs near zero while the robot just stood and
        # balanced (original 8x progress weight only earned 0.056/step at
        # full commanded speed against ~0.46/step of near-free "upright"
        # reward), then, after raising the progress weight alone, drifted
        # *backward* instead (there was nothing discouraging retreating away
        # from the stairs, and no cost to the episode ending, so a long slow
        # retreat scored as well as a short failed forward attempt). Fixed
        # with two changes together: a meaningful termination penalty (so
        # ending the episode early is never free) makes surviving matter,
        # and a positive-progress bonus specifically rewards moving in the
        # correct direction rather than merely treating "more negative
        # progress" as symmetrically as "more positive progress".
        terms = {
            "progress_along_stairs": 30.0 * progress,
            "forward_progress_bonus": 15.0 * max(0.0, progress),
            "upright": 0.15 * max(0.0, -float(gravity[2])),
            "foot_clearance": 0.05 * self._foot_clearance(),
            "stable_foot_contact": 0.1 if feet_down >= 1 else 0.0,
            "landing_completion": 5.0 if self._on_landing() else 0.0,
            "low_slip": -0.02 * self._slip(contacting),
            "smooth_action": -0.01 * float(np.sum((action - self.previous_action) ** 2)),
            "energy_penalty": -0.001 * float(np.sum(action ** 2)),
            "termination_penalty": -5.0 if termination is not None else 0.0,
        }
        return float(sum(terms.values())), terms
