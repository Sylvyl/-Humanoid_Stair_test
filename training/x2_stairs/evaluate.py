"""Deterministic success-rate evaluator.

Nothing before this measured the actual thing the release gate's simulation
acceptance bar cares about (`evaluate_release_gate`'s "1,000 held-out
episodes per direction with >=95% completion") -- prior tooling this session
only tracked reward and survival time, neither of which is "did it actually
complete the step." An episode here counts as a success if the robot's base
reaches the landing (`StairMujocoEnv.reached_landing()`) at any point before
a hard termination fires. Runs the policy's *mean* action (no sampling
noise), matching how the exported ONNX policy behaves at inference time --
not the stochastic rollout used during training.

This is deliberately the same shape as `validation/simulation_report.json`
(`up`/`down` episode counts and success rates) so a large enough run can
graduate directly into that file, but small diagnostic runs (tens to a few
hundred episodes) are far short of the required 1,000/direction and must not
be mistaken for release-gate evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from . import _src  # noqa: F401  (adds src/ to sys.path)
from x2_stair_autonomy.policy_contract import PolicyContract

from .mujoco_env import StairMujocoEnv
from .ppo_train import _MANIFEST_PATH, ActorCritic
from .task_spec import TaskSpec, direction_for_stage, domain_for_stage


def evaluate(
    model: ActorCritic, contract: PolicyContract, stage: str,
    *, episodes: int, max_steps: int, base_seed: int,
) -> dict:
    model.eval()
    task_spec = TaskSpec(domain=domain_for_stage(stage))
    direction = direction_for_stage(stage)
    env = StairMujocoEnv(task_spec=task_spec)

    episode_results = []
    for index in range(episodes):
        seed = base_seed + index
        values = env.reset(seed, force_direction=direction)
        success = False
        termination = None
        steps_taken = max_steps
        for step in range(max_steps):
            obs = contract.observation(values)
            with torch.no_grad():
                mean, _value = model.forward(torch.from_numpy(obs.reshape(1, -1).astype(np.float32)))
            action = mean.squeeze(0).numpy()
            step_result = env.step(action)
            values = step_result.observation
            if env.reached_landing():
                success = True
            if step_result.done:
                termination = step_result.termination
                steps_taken = step + 1
                break
        episode_results.append({
            "seed": seed, "direction": env.direction, "success": success,
            "termination": termination, "steps": steps_taken,
        })

    by_direction: dict[str, dict] = {}
    for direction_name in sorted({r["direction"] for r in episode_results}):
        subset = [r for r in episode_results if r["direction"] == direction_name]
        by_direction[direction_name] = {
            "episodes": len(subset),
            "success_rate": sum(r["success"] for r in subset) / len(subset),
        }
    failure_terminations: dict[str, int] = {}
    for result in episode_results:
        if not result["success"]:
            key = result["termination"] or "max_steps_without_landing"
            failure_terminations[key] = failure_terminations.get(key, 0) + 1

    return {
        "stage": stage,
        "episodes": episodes,
        "max_steps": max_steps,
        "success_rate": sum(r["success"] for r in episode_results) / episodes,
        "mean_steps": float(np.mean([r["steps"] for r in episode_results])),
        "by_direction": by_direction,
        "failure_terminations": failure_terminations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="ActorCritic state_dict (.pt) from ppo_train.py")
    parser.add_argument("--stage", default="single_step_up")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = PolicyContract.from_path(_MANIFEST_PATH)
    model = ActorCritic(contract.observation_size, contract.action_size, contract.action_clip)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    report = evaluate(
        model, contract, args.stage,
        episodes=args.episodes, max_steps=args.max_steps, base_seed=args.seed,
    )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["checkpoint"] = str(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
