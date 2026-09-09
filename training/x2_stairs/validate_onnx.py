from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def manifest_blockers(manifest: dict[str, Any], approval: dict[str, Any]) -> list[str]:
    """Return every static policy-contract blocker instead of failing at the first one."""
    blockers: list[str] = []
    if manifest.get("status") != "VENDOR_APPROVED":
        blockers.append("manifest status is not VENDOR_APPROVED")
    if approval.get("approved") is not True:
        blockers.append("vendor approval is not approved")

    joint_order = manifest.get("joint_order", [])
    action_size = manifest.get("action_size")
    if not isinstance(action_size, int) or action_size <= 0:
        blockers.append("action_size must be a positive integer")
    elif not isinstance(joint_order, list) or len(joint_order) != action_size:
        blockers.append("joint_order length does not match action_size")
    elif len(set(joint_order)) != len(joint_order):
        blockers.append("joint_order contains duplicates")

    observation_size = manifest.get("observation_size")
    inputs = manifest.get("inputs", [])
    if not isinstance(observation_size, int) or observation_size <= 0:
        blockers.append("observation_size must be a positive integer")
    elif not isinstance(inputs, list) or sum(
        item.get("size", 0) for item in inputs if isinstance(item, dict)
    ) != observation_size:
        blockers.append("input field sizes do not match observation_size")

    outputs = manifest.get("outputs", [])
    if isinstance(action_size, int) and (
        not isinstance(outputs, list)
        or sum(item.get("size", 0) for item in outputs if isinstance(item, dict)) != action_size
    ):
        blockers.append("output field sizes do not match action_size")

    manifest_rate = manifest.get("command_publish_rate_hz")
    approved_rate = approval.get("command_rate_hz")
    if not isinstance(manifest_rate, (int, float)) or manifest_rate <= 0:
        blockers.append("manifest command_publish_rate_hz must be positive")
    elif manifest_rate != approved_rate:
        blockers.append("manifest command rate differs from vendor approval")

    scale = manifest.get("action_scale")
    if not isinstance(scale, (int, float, list)):
        blockers.append("action_scale is missing")
    elif isinstance(scale, (int, float)) and scale <= 0:
        blockers.append("action_scale must be positive")
    elif isinstance(scale, list) and (
        len(scale) != action_size or any(not isinstance(v, (int, float)) or v <= 0 for v in scale)
    ):
        blockers.append("action_scale vector must contain one positive value per action")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ONNX parity and the signed stair-policy contract"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="NPZ containing observations and native_actions arrays",
    )
    parser.add_argument("--mujoco-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    manifest = _read_json(args.manifest)
    approval = _read_json(args.approval)
    mujoco = _read_json(args.mujoco_report)
    blockers = manifest_blockers(manifest, approval)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if approval.get("onnx_contract_sha256") != model_hash:
        blockers.append("ONNX SHA-256 differs from vendor-approved contract")
    if not (
        mujoco.get("passed") is True
        and mujoco.get("limit_violations") == 0
        and mujoco.get("fault_stop_passed") is True
    ):
        blockers.append("MuJoCo closed-loop report is not passing")

    max_abs_error: float | None = None
    shape: list[int] | None = None
    input_name: str | None = None
    output_name: str | None = None
    if not blockers:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
            if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                blockers.append("policy must have exactly one input and one output tensor")
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                with np.load(args.reference, allow_pickle=False) as reference:
                    observations = np.asarray(reference["observations"], dtype=np.float32)
                    native_actions = np.asarray(reference["native_actions"], dtype=np.float32)
                expected_obs = manifest["observation_size"]
                expected_actions = manifest["action_size"]
                if observations.ndim != 2 or observations.shape[1] != expected_obs:
                    blockers.append("reference observations have the wrong shape")
                elif native_actions.shape != (observations.shape[0], expected_actions):
                    blockers.append("reference native_actions have the wrong shape")
                elif not np.isfinite(observations).all() or not np.isfinite(native_actions).all():
                    blockers.append("reference arrays contain non-finite values")
                else:
                    onnx_actions = np.asarray(
                        session.run([output_name], {input_name: observations})[0], dtype=np.float32
                    )
                    shape = list(onnx_actions.shape)
                    if onnx_actions.shape != native_actions.shape:
                        blockers.append("ONNX output shape differs from native reference")
                    elif not np.isfinite(onnx_actions).all():
                        blockers.append("ONNX output contains non-finite values")
                    else:
                        max_abs_error = float(np.max(np.abs(onnx_actions - native_actions)))
                        if max_abs_error > args.tolerance:
                            blockers.append("ONNX/native parity exceeds tolerance")
        except Exception as exc:
            blockers.append(f"ONNX validation failed: {type(exc).__name__}: {exc}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not blockers,
        "model_sha256": model_hash,
        "input_name": input_name,
        "output_name": output_name,
        "output_shape": shape,
        "tolerance": args.tolerance,
        "max_abs_error": max_abs_error,
        "mujoco_closed_loop_passed": mujoco.get("passed") is True,
        "blockers": blockers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
