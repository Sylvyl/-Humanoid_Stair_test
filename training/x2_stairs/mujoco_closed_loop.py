"""Closed-loop MuJoCo validator: the piece `training/README.md` already
documents and `validate_onnx.py --mujoco-report` already expects, but that
nothing in this repository previously generated.

Runs a candidate ONNX policy through `StairMujocoEnv` for a handful of sanity
episodes, always through the *existing* fail-closed `PolicyContract.infer()`
wrapper (never calling the ONNX session directly), and separately injects a
sensor-staleness fault mid-episode to confirm the effective command goes to
exactly zero -- the `fault_stop_passed` claim the release gate names.

This checks closed-loop *execution correctness* and *fault-stop behavior*,
not full task mastery: the 1000-episode/95%-success acceptance bar lives in
`validation/simulation_report.json`, a separate, much larger gate item that is
explicitly out of scope for this tool.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import _src  # noqa: F401  (adds src/ to sys.path)
from x2_stair_autonomy.policy_contract import PolicyContract

from .mujoco_env import StairMujocoEnv


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Install `onnxruntime` (see training/requirements-sim.txt) to run mujoco_closed_loop"
        ) from exc
    return ort


def _make_runner(model_path: Path):
    ort = _require_onnxruntime()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("policy must have exactly one input and one output tensor")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def runner(observation: np.ndarray) -> np.ndarray:
        return session.run([output_name], {input_name: observation.astype(np.float32)})[0]

    return runner


def run_episode(
    env: StairMujocoEnv,
    contract: PolicyContract,
    runner,
    seed: int,
    *,
    max_steps: int,
    fault_at_step: int | None = None,
) -> dict:
    """Run one episode. If `fault_at_step` is set, sensors are reported stale
    from that step onward and the episode fails fast unless the resulting
    action is exactly zero at every faulted step."""
    values = env.reset(seed)
    limit_violations = 0
    for step in range(max_steps):
        faulted = fault_at_step is not None and step >= fault_at_step
        result = contract.infer(
            runner, values,
            sensors_fresh=not faulted, release_unlocked=True, heartbeat_fresh=True,
        )
        if faulted and not np.array_equal(result.action, np.zeros_like(result.action)):
            return {
                "seed": seed, "passed": False, "fault_zeroed_action": False,
                "steps": step, "termination": None, "limit_violations": limit_violations,
            }
        step_result = env.step(result.action)
        values = step_result.observation
        if step_result.termination == "joint_limit":
            limit_violations += 1
        if step_result.done:
            return {
                "seed": seed,
                "passed": True,
                "fault_zeroed_action": True if fault_at_step is not None else None,
                "steps": step + 1,
                "termination": step_result.termination,
                "direction": env.direction,
                "limit_violations": limit_violations,
            }
    return {
        "seed": seed, "passed": True,
        "fault_zeroed_action": True if fault_at_step is not None else None,
        "steps": max_steps, "termination": None, "direction": env.direction,
        "limit_violations": limit_violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "policy_manifest.json")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fault-step", type=int, default=50, help="step at which the fault-stop episode reports stale sensors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = PolicyContract.from_path(args.manifest)
    runner = _make_runner(args.model)
    env = StairMujocoEnv()

    episodes = [
        run_episode(env, contract, runner, args.seed + index, max_steps=args.max_steps)
        for index in range(args.episodes)
    ]
    fault_episode = run_episode(
        env, contract, runner, args.seed + args.episodes, max_steps=args.max_steps, fault_at_step=args.fault_step,
    )

    limit_violations = sum(ep["limit_violations"] for ep in episodes) + fault_episode["limit_violations"]
    fault_stop_passed = bool(fault_episode.get("fault_zeroed_action"))
    report = {
        "kind": "mujoco-closed-loop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "episodes": episodes,
        "fault_episode": fault_episode,
        "limit_violations": limit_violations,
        "fault_stop_passed": fault_stop_passed,
        "passed": limit_violations == 0 and fault_stop_passed,
        "notice": (
            "Sanity/fault-stop check only; full task-success acceptance lives in "
            "validation/simulation_report.json (1000+ episodes/direction, >=95% success)."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
