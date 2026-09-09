from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from .models import StairEstimate, StairProfile


def depth_to_points(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_to_base: np.ndarray,
    stride: int = 4,
) -> np.ndarray:
    """Project a metric RGB-D image into base coordinates using reviewed extrinsics."""
    image = np.asarray(depth, dtype=float)
    transform = np.asarray(camera_to_base, dtype=float)
    if image.ndim != 2 or transform.shape != (4, 4):
        raise ValueError("depth must be HxW and camera_to_base must be 4x4")
    if min(fx, fy) <= 0 or stride < 1:
        raise ValueError("camera intrinsics and stride must be positive")
    v, u = np.mgrid[0:image.shape[0]:stride, 0:image.shape[1]:stride]
    z = image[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0.15) & (z < 5.0)
    # Optical coordinates: x right, y down, z forward.
    optical = np.column_stack((((u[valid] - cx) * z[valid] / fx),
                               ((v[valid] - cy) * z[valid] / fy), z[valid],
                               np.ones(int(valid.sum()))))
    return (optical @ transform.T)[:, :3]


def _gravity_align(points: np.ndarray, roll_rad: float, pitch_rad: float) -> np.ndarray:
    """Remove measured roll/pitch; ROS coordinates are x forward, y left, z up."""
    cr, sr = math.cos(-roll_rad), math.sin(-roll_rad)
    cp, sp = math.cos(-pitch_rad), math.sin(-pitch_rad)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    return points @ (ry @ rx).T


def _cluster_levels(z_values: np.ndarray, tolerance_m: float = 0.018) -> list[float]:
    if not len(z_values):
        return []
    rounded = np.round(z_values / 0.008).astype(int)
    bins, counts = np.unique(rounded, return_counts=True)
    minimum = max(12, int(len(z_values) * 0.008))
    peaks = sorted(float(b * 0.008) for b, c in zip(bins, counts) if c >= minimum)
    groups: list[list[float]] = []
    for value in peaks:
        if not groups or value - groups[-1][-1] > tolerance_m:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.median(group)) for group in groups]


def estimate_stairs(
    points_xyz: Iterable[Iterable[float]],
    profile: StairProfile | None = None,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> StairEstimate:
    """Estimate a straight staircase from an already synchronized point cloud.

    It is deliberately conservative: uncertain or unsupported geometry is
    returned with ``supported=False`` and can never arm the supervisor.
    """
    profile = profile or StairProfile()
    points = np.asarray(list(points_xyz), dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 150:
        return StairEstimate(rejection_reason="insufficient point cloud")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    points = _gravity_align(points, roll_rad, pitch_rad)
    roi = points[
        (points[:, 0] >= 0.20)
        & (points[:, 0] <= 3.50)
        & (np.abs(points[:, 1]) <= 1.25)
        & (points[:, 2] >= -0.60)
        & (points[:, 2] <= 2.20)
    ]
    if len(roi) < 150:
        return StairEstimate(rejection_reason="not enough points in stair region")

    levels = _cluster_levels(roi[:, 2])
    level_rows: list[tuple[float, float, float, float]] = []
    for level in levels:
        sample = roi[np.abs(roi[:, 2] - level) <= 0.022]
        if len(sample) < 20:
            continue
        level_rows.append(
            (level, float(np.quantile(sample[:, 0], 0.10)), float(np.quantile(sample[:, 0], 0.90)),
             float(np.quantile(sample[:, 1], 0.95) - np.quantile(sample[:, 1], 0.05)))
        )
    if len(level_rows) < 3:
        return StairEstimate(rejection_reason="fewer than three horizontal levels")

    # Sort by the progression coordinate of each visible tread.
    rows = sorted(level_rows, key=lambda row: (row[1] + row[2]) * 0.5)
    centers_x = np.array([(row[1] + row[2]) * 0.5 for row in rows])
    heights = np.array([row[0] for row in rows])
    dz = np.diff(heights)
    valid_risers = np.abs(dz[(np.abs(dz) >= 0.10) & (np.abs(dz) <= 0.25)])
    if len(valid_risers) < 2:
        return StairEstimate(available=True, rejection_reason="riser pattern is inconsistent")
    sign = float(np.median(dz))
    direction = "up" if sign > 0 else "down"
    riser = float(np.median(valid_risers))

    dx = np.abs(np.diff(centers_x))
    valid_treads = dx[(dx >= 0.15) & (dx <= 0.50)]
    if len(valid_treads) < 2:
        return StairEstimate(available=True, direction=direction, riser_m=riser,
                             rejection_reason="tread pattern is inconsistent")
    tread = float(np.median(valid_treads))
    width = float(np.median([row[3] for row in rows]))
    distance = float(max(0.0, min(row[1] for row in rows)))
    top_row = rows[int(np.argmax(heights))]
    bottom_row = rows[int(np.argmin(heights))]
    landing_extent = (top_row[2] - top_row[1]) if direction == "up" else (bottom_row[2] - bottom_row[1])
    landing = landing_extent >= profile.min_landing_m

    consistent_riser = float(np.std(valid_risers)) <= 0.018
    consistent_tread = float(np.std(valid_treads)) <= 0.030
    within = (
        profile.min_riser_m <= riser <= profile.max_riser_m
        and profile.min_tread_m <= tread <= profile.max_tread_m
        and width >= profile.min_width_m
    )
    confidence = 0.35
    confidence += 0.20 if consistent_riser else 0.0
    confidence += 0.20 if consistent_tread else 0.0
    confidence += 0.15 if width >= profile.min_width_m else 0.0
    confidence += 0.10 if len(valid_risers) >= 3 else 0.0
    confidence = round(min(1.0, confidence), 3)

    reason = None
    if not within:
        reason = "geometry is outside the approved straight-stair profile"
    elif confidence < profile.min_confidence:
        reason = "estimate confidence is below the approved threshold"

    return StairEstimate(
        available=True,
        supported=within and confidence >= profile.min_confidence,
        direction=direction,
        distance_m=round(distance, 3),
        yaw_deg=0.0,
        riser_m=round(riser, 3),
        tread_m=round(tread, 3),
        width_m=round(width, 3),
        step_count=int(len(valid_risers)),
        landing_detected=landing,
        confidence=confidence,
        rejection_reason=reason,
        stamp=time.time(),
    )
