"""Puts the sibling ``src/`` package on ``sys.path`` for training/simulation
tooling that reuses ``x2_stair_autonomy`` code (``PolicyContract``,
``StairProfile``) instead of duplicating it.

Import this (for its side effect) before importing anything from
``x2_stair_autonomy``. Never imported by the runtime dashboard/ROS bridge --
only by training/x2_stairs modules that already require optional simulation
dependencies such as ``mujoco``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
