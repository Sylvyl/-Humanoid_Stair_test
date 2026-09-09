from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class InferenceResult:
    action: np.ndarray
    valid: bool
    reason: str


class PolicyContract:
    """Tensor contract and fail-closed inference wrapper; never publishes commands."""

    def __init__(self, manifest: Mapping):
        self.manifest = dict(manifest)
        self.inputs = tuple(self.manifest.get("inputs", ()))
        self.observation_size = int(self.manifest.get("observation_size", 0))
        self.action_size = int(self.manifest.get("action_size", 0))
        self.action_clip = float(self.manifest.get("action_clip", 0.0))
        if self.observation_size <= 0 or self.action_size <= 0 or self.action_clip <= 0:
            raise ValueError("manifest tensor sizes and action_clip must be positive")
        if sum(int(item.get("size", 0)) for item in self.inputs) != self.observation_size:
            raise ValueError("manifest input sizes do not equal observation_size")

    @classmethod
    def from_path(cls, path: Path) -> "PolicyContract":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def observation(self, values: Mapping[str, Sequence[float]]) -> np.ndarray:
        pieces = []
        for item in self.inputs:
            name = str(item["name"])
            expected = int(item["size"])
            if name not in values:
                raise ValueError(f"missing observation field: {name}")
            array = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if array.size != expected:
                raise ValueError(f"{name} has size {array.size}; expected {expected}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            pieces.append(array * np.float32(item.get("scale", 1.0)))
        observation = np.concatenate(pieces).astype(np.float32, copy=False)
        if observation.size != self.observation_size:
            raise ValueError("assembled observation size mismatch")
        return observation

    def infer(
        self,
        runner: Callable[[np.ndarray], np.ndarray],
        values: Mapping[str, Sequence[float]],
        *,
        sensors_fresh: bool,
        release_unlocked: bool,
        heartbeat_fresh: bool,
    ) -> InferenceResult:
        zero = np.zeros(self.action_size, dtype=np.float32)
        if not release_unlocked:
            return InferenceResult(zero, False, "release gate locked")
        if not sensors_fresh:
            return InferenceResult(zero, False, "sensor data stale")
        if not heartbeat_fresh:
            return InferenceResult(zero, False, "operator heartbeat stale")
        try:
            observation = self.observation(values)
            raw = np.asarray(runner(observation.reshape(1, -1)), dtype=np.float32).reshape(-1)
        except Exception as exc:
            return InferenceResult(zero, False, f"inference rejected: {exc}")
        if raw.size != self.action_size:
            return InferenceResult(zero, False, "policy action size mismatch")
        if not np.isfinite(raw).all():
            return InferenceResult(zero, False, "policy produced non-finite action")
        return InferenceResult(np.clip(raw, -self.action_clip, self.action_clip), True, "ok")

