"""Synthetic stair terrain for the local MuJoCo environment.

Generalizes the stair block already sketched (commented out) in the vendor's
own ``x2_rl_deploy_mujoco/.../model_info/scene.xml``: stacked
``<geom type="box">`` steps, each box's base at world z=0 and its top surface
at the cumulative riser height, side by side along x. This module makes that
pattern parametric, driven by a sampled ``TaskSpec.sample_domain()`` dict.

Terrain is always generated ascending as x increases, starting after a flat
0.5 m approach zone (matching the vendor example's starting x). Episode
*direction* ("up" spawn at the bottom facing +x, "down" spawn on the landing
facing -x over the same geometry) is a ``StairMujocoEnv`` concern, not a
terrain-shape concern -- kept out of this module so the geometry math stays
simple and direction-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import _src  # noqa: F401  (adds src/ to sys.path)
from x2_stair_autonomy.models import StairProfile

APPROACH_LENGTH_M = 0.5


@dataclass(frozen=True, slots=True)
class StepGeom:
    x_center: float
    z_top: float
    half_depth: float
    half_width: float

    @property
    def half_height(self) -> float:
        return self.z_top / 2.0

    @property
    def x_leading(self) -> float:
        return self.x_center - self.half_depth

    @property
    def x_trailing(self) -> float:
        return self.x_center + self.half_depth


@dataclass(frozen=True, slots=True)
class StairLayout:
    riser_m: float
    tread_m: float
    width_m: float
    steps: tuple[StepGeom, ...]
    landing: StepGeom

    @property
    def top_z(self) -> float:
        return self.landing.z_top

    @property
    def start_x(self) -> float:
        return self.steps[0].x_leading if self.steps else APPROACH_LENGTH_M

    @property
    def end_x(self) -> float:
        return self.landing.x_trailing


def build_layout(domain: dict, profile: StairProfile | None = None) -> StairLayout:
    """Build a stair layout from a ``TaskSpec.sample_domain()`` dict.

    ``domain`` must provide ``riser_m``, ``tread_m``, ``width_m``, and
    ``steps`` (an int step count); this is exactly the shape
    ``TaskSpec.sample_domain(seed)`` already returns.
    """
    profile = profile or StairProfile()
    riser = float(domain["riser_m"])
    tread = float(domain["tread_m"])
    width = float(domain["width_m"])
    step_count = int(domain["steps"])
    if riser <= 0 or tread <= 0 or width <= 0 or step_count < 1:
        raise ValueError("riser_m, tread_m, width_m must be positive and steps >= 1")

    half_depth = tread / 2.0
    half_width = width / 2.0
    steps = tuple(
        StepGeom(
            x_center=APPROACH_LENGTH_M + half_depth + index * tread,
            z_top=riser * (index + 1),
            half_depth=half_depth,
            half_width=half_width,
        )
        for index in range(step_count)
    )
    top_z = steps[-1].z_top
    landing_depth = max(profile.min_landing_m, tread)
    landing = StepGeom(
        x_center=steps[-1].x_trailing + landing_depth / 2.0,
        z_top=top_z,
        half_depth=landing_depth / 2.0,
        half_width=half_width,
    )
    return StairLayout(riser_m=riser, tread_m=tread, width_m=width, steps=steps, landing=landing)


def height_at(layout: StairLayout, x: float) -> float:
    """Ground height at longitudinal position ``x`` (privileged/teacher query).

    Used by the environment to build the ``terrain_height_samples`` observation
    field analytically, without raycasting against the physics scene.
    """
    if x < layout.start_x:
        return 0.0
    for step in layout.steps:
        if x <= step.x_trailing:
            return step.z_top
    if x <= layout.landing.x_trailing:
        return layout.landing.z_top
    return layout.landing.z_top


def _geom_xml(geom: StepGeom) -> str:
    return (
        f'<geom type="box" size="{geom.half_depth:.4f} {geom.half_width:.4f} '
        f'{geom.half_height:.4f}" pos="{geom.x_center:.4f} 0 {geom.half_height:.4f}" '
        f'rgba="0.55 0.55 0.58 1" conaffinity="15" condim="3"/>'
    )


def to_mjcf(layout: StairLayout) -> str:
    """MJCF ``<geom>`` fragments for every step plus the landing, as a string
    ready to be inserted inside a ``<worldbody>`` element."""
    lines = [_geom_xml(step) for step in layout.steps]
    lines.append(_geom_xml(layout.landing))
    return "\n".join(lines)
