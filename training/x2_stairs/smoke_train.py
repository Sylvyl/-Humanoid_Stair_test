"""Dependency-light contract smoke test; this is not policy training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from task_spec import TaskSpec


def run(episodes: int, seed: int) -> dict:
    spec = TaskSpec()
    rng = np.random.default_rng(seed)
    failures = []
    samples = []
    for index in range(episodes):
        sample = spec.sample_domain(seed + index)
        obs = rng.normal(size=spec.observation_size).astype(np.float32)
        action = np.tanh(rng.normal(size=spec.action_size)).astype(np.float32)
        if obs.shape != (144,) or action.shape != (29,) or not np.isfinite(action).all():
            failures.append(index)
        samples.append(sample)
    return {
        "kind": "contract-smoke",
        "created_at": time.time(),
        "episodes": episodes,
        "seed": seed,
        "passed": not failures,
        "failures": failures,
        "first_domain": samples[0] if samples else None,
        "notice": "No locomotion policy was trained and no robot command was emitted."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.episodes, args.seed)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

