from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


TOPIC_ROLES = {
    "depth": ("/aima/hal/sensor/rgbd_head_front/depth_image",),
    "rgb": ("/aima/hal/sensor/rgbd_head_front/rgb_image/compressed",),
    "lidar": ("/aima/hal/sensor/lidar_chest_front/lidar_pointcloud",),
    "imu": ("/aima/hal/imu/torso/state", "/aima/hal/imu/chest/state"),
    "joints": ("/aima/hal/joint/leg/state",),
    "power": ("/aima/hal/pmu/state", "/aima/hal/battery/state"),
}


def _run(*argv: str, timeout: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": 124, "stdout": "", "stderr": str(exc)}


def collect() -> dict[str, Any]:
    topic_result = _run("ros2", "topic", "list", "-t")
    service_result = _run("ros2", "service", "list", "-t")
    topics: dict[str, list[str]] = {}
    for line in topic_result["stdout"].splitlines():
        if " [" in line and line.endswith("]"):
            name, raw_types = line.split(" [", 1)
            topics[name] = [item.strip() for item in raw_types[:-1].split(",")]
    roles = {role: next((topic for topic in candidates if topic in topics), None) for role, candidates in TOPIC_ROLES.items()}
    required_missing = [role for role in ("depth", "rgb", "lidar", "imu", "joints") if roles[role] is None]
    details = {topic: _run("ros2", "topic", "info", "-v", topic, timeout=6) for topic in roles.values() if topic}
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "host": {"architecture": platform.machine(), "python": platform.python_version(), "hostname": platform.node()},
        "environment": {
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
            "disk": shutil.disk_usage("/agibot/data/home/agi")._asdict() if Path("/agibot/data/home/agi").exists() else None,
        },
        "topics": topics,
        "services_raw": service_result,
        "roles": roles,
        "topic_details": details,
        "required_missing": required_missing,
        "power_source_missing": roles["power"] is None,
        "passed_shadow": not required_missing,
        "passed_hardware": not required_missing and roles["power"] is not None,
    }
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Read-only AimDK PC2 compatibility inventory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = collect()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["passed_shadow"] else 2)


if __name__ == "__main__":
    main()

