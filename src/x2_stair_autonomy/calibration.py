from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .models import StairProfile
from .perception import estimate_stairs


LIMITS = {"riser_m": 0.010, "tread_m": 0.015, "yaw_deg": 2.0}


def evaluate_scenes(scenes: list[dict[str, Any]], profile: StairProfile | None = None) -> dict[str, Any]:
    profile = profile or StairProfile(approved=False)
    results = []
    errors = {name: [] for name in LIMITS}
    failures: list[str] = []
    for index, scene in enumerate(scenes):
        scene_id = str(scene.get("id", index))
        expected = scene.get("expected", {})
        estimate = estimate_stairs(
            scene.get("points_xyz", []), profile,
            roll_rad=float(scene.get("roll_rad", 0.0)),
            pitch_rad=float(scene.get("pitch_rad", 0.0)),
        )
        expected_supported = bool(expected.get("supported"))
        case = {"id": scene_id, "expected": expected, "estimate": estimate.to_dict(), "passed": True, "errors": {}}
        if expected_supported != estimate.supported:
            case["passed"] = False
            failures.append(f"{scene_id}: supported classification mismatch")
        if expected_supported:
            if expected.get("direction") != estimate.direction:
                case["passed"] = False
                failures.append(f"{scene_id}: direction mismatch")
            for name, limit in LIMITS.items():
                actual = getattr(estimate, name)
                if actual is None or name not in expected:
                    delta = None
                else:
                    delta = abs(float(actual) - float(expected[name]))
                case["errors"][name] = delta
                if delta is not None:
                    errors[name].append(delta)
                if delta is None or delta > limit:
                    case["passed"] = False
                    rendered = "missing" if delta is None else f"{delta:.6f}"
                    failures.append(f"{scene_id}: {name} error {rendered} exceeds {limit:.6f}")
        elif estimate.supported:
            failures.append(f"{scene_id}: unsupported scene was accepted")
        results.append(case)

    maxima = {name: max(values, default=None) for name, values in errors.items()}
    supported_count = sum(bool(scene.get("expected", {}).get("supported")) for scene in scenes)
    rejected_count = len(scenes) - supported_count
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(scenes) and not failures and supported_count > 0 and rejected_count > 0,
        "thresholds": LIMITS,
        "max_errors": maxima,
        "supported_cases": supported_count,
        "rejection_cases": rejected_count,
        "failures": failures,
        "cases": results,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Deterministic X2 stair perception acceptance runner")
    parser.add_argument("dataset", type=Path, help="JSON document containing a scenes array")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    scenes = dataset.get("scenes") if isinstance(dataset, dict) else None
    if not isinstance(scenes, list):
        raise SystemExit("dataset must be an object containing a scenes array")
    report = evaluate_scenes(scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "max_errors", "failures")}, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
