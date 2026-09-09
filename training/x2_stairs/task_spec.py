from __future__ import annotations

from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class StairDomain:
    riser_m: tuple[float, float] = (0.150, 0.190)
    tread_m: tuple[float, float] = (0.250, 0.320)
    width_m: tuple[float, float] = (0.900, 1.500)
    steps: tuple[int, int] = (3, 16)
    friction: tuple[float, float] = (0.55, 1.10)
    payload_kg: tuple[float, float] = (0.0, 3.0)
    mass_scale: tuple[float, float] = (0.92, 1.08)
    actuator_strength: tuple[float, float] = (0.90, 1.05)
    observation_latency_s: tuple[float, float] = (0.0, 0.040)
    depth_noise_m: tuple[float, float] = (0.0, 0.015)
    imu_noise_rad_s: tuple[float, float] = (0.0, 0.025)
    light_level: tuple[float, float] = (0.15, 1.0)


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[str, ...] = (
        "flat_balance",
        "single_step_up",
        "single_step_down",
        "three_steps",
        "random_straight_flight",
        "noisy_sensor_student",
    )


@dataclass(frozen=True)
class TaskSpec:
    domain: StairDomain = StairDomain()
    curriculum: Curriculum = Curriculum()
    policy_hz: int = 50
    action_size: int = 29
    observation_size: int = 144
    teacher_uses_privileged_terrain: bool = True
    student_uses_height_samples: bool = True
    hard_terminations: tuple[str, ...] = (
        "base_contact",
        "non_foot_collision",
        "excessive_tilt",
        "joint_limit",
        "stale_observation",
        "stair_profile_violation",
    )
    rewards: tuple[str, ...] = (
        "progress_along_stairs",
        "forward_progress_bonus",
        "upright",
        "foot_clearance",
        "stable_foot_contact",
        "landing_completion",
        "low_slip",
        "smooth_action",
        "termination_penalty",
        "energy_penalty",
    )

    def sample_domain(self, seed: int, *, force_direction: str | None = None) -> dict:
        rng = random.Random(seed)
        sampled = {}
        for name, bounds in asdict(self.domain).items():
            if name == "steps":
                sampled[name] = rng.randint(*bounds)
            else:
                sampled[name] = rng.uniform(*bounds)
        sampled["direction"] = force_direction if force_direction in ("up", "down") else rng.choice(("up", "down"))
        return sampled


# Curriculum.stages names a progression; only the entries below have a
# concrete StairDomain wired up (a follow-up session can add the rest).
# "flat_balance" needs a no-stairs env mode (StairMujocoEnv always builds
# stair terrain today); "noisy_sensor_student" needs observation-noise
# injection / teacher-student distillation. Neither exists yet, so
# domain_for_stage raises NotImplementedError naming what's missing rather
# than silently approximating them.
_CURRICULUM_STAGE_DOMAINS: dict[str, StairDomain] = {
    "single_step_up": StairDomain(steps=(1, 1), riser_m=(0.165, 0.175), tread_m=(0.280, 0.300)),
    "single_step_down": StairDomain(steps=(1, 1), riser_m=(0.165, 0.175), tread_m=(0.280, 0.300)),
    "three_steps": StairDomain(steps=(3, 3)),
}
_CURRICULUM_STAGE_DIRECTION: dict[str, str] = {
    "single_step_up": "up",
    "single_step_down": "down",
}
_UNIMPLEMENTED_CURRICULUM_STAGES: dict[str, str] = {
    "flat_balance": "requires a no-stairs env mode",
    "noisy_sensor_student": "requires observation-noise injection / teacher-student distillation",
}


def domain_for_stage(stage: str) -> StairDomain:
    """Narrowed ``StairDomain`` bounds for one named ``Curriculum`` stage."""
    if stage in _CURRICULUM_STAGE_DOMAINS:
        return _CURRICULUM_STAGE_DOMAINS[stage]
    if stage == "random_straight_flight":
        return StairDomain()
    if stage in _UNIMPLEMENTED_CURRICULUM_STAGES:
        raise NotImplementedError(
            f"curriculum stage '{stage}' is not implemented: {_UNIMPLEMENTED_CURRICULUM_STAGES[stage]}"
        )
    raise ValueError(f"unknown curriculum stage: {stage}")


def direction_for_stage(stage: str) -> str | None:
    """Forced episode direction for a stage, or ``None`` to sample randomly."""
    return _CURRICULUM_STAGE_DIRECTION.get(stage)

