from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


REQUIRED_ROLES = ("depth", "rgb", "lidar", "imu", "joints")
DERIVED_TOPICS = ("/x2/stairs/estimate", "/x2/stairs/safety")


def recording_topics(inventory: dict) -> list[str]:
    roles = inventory.get("roles", {})
    missing = [role for role in REQUIRED_ROLES if not roles.get(role)]
    if missing:
        raise ValueError("required sensor roles are missing: " + ", ".join(missing))
    topics = [roles[role] for role in REQUIRED_ROLES]
    if roles.get("power"):
        topics.append(roles["power"])
    available = inventory.get("topics", {})
    topics.extend(topic for topic in DERIVED_TOPICS if topic in available)
    return list(dict.fromkeys(topics))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plan or execute a synchronized shadow rosbag recording")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    topics = recording_topics(inventory)
    output = args.output or Path.home() / "x2dev" / "stair-bags" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = {"read_only": True, "output": str(output), "topics": topics, "power_source_missing": not inventory.get("roles", {}).get("power")}
    print(json.dumps(plan, indent=2))
    if args.execute:
        output.parent.mkdir(parents=True, exist_ok=True)
        raise SystemExit(subprocess.run(["ros2", "bag", "record", "-o", str(output), *topics], check=False).returncode)


if __name__ == "__main__":
    main()

