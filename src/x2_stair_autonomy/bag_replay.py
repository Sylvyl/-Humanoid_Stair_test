from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .calibration import evaluate_scenes


def matching_scene(scenes: list[dict[str, Any]], timestamp_ns: int) -> dict[str, Any] | None:
    matches = [
        scene
        for scene in scenes
        if int(scene.get("start_ns", -1)) <= timestamp_ns <= int(scene.get("end_ns", -1))
    ]
    if len(matches) > 1:
        raise ValueError(f"annotation intervals overlap at {timestamp_ns}")
    return matches[0] if matches else None


def validate_annotations(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != 1:
        raise ValueError("annotation schema_version must be 1")
    if document.get("lidar_frame_approved") is not True:
        raise ValueError("LiDAR-to-base transform has not been approved")
    if document.get("imu_frame_approved") is not True:
        raise ValueError("IMU-to-base transform has not been approved")
    scenes = document.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("annotations must contain a non-empty scenes array")
    for scene in scenes:
        if not isinstance(scene, dict) or not {"id", "start_ns", "end_ns", "expected"} <= scene.keys():
            raise ValueError("each scene requires id, start_ns, end_ns, and expected")
        if int(scene["start_ns"]) > int(scene["end_ns"]):
            raise ValueError(f"scene {scene['id']} has an invalid time interval")
    return scenes


def _euler(orientation: Any) -> tuple[float, float]:
    sinr = 2.0 * (orientation.w * orientation.x + orientation.y * orientation.z)
    cosr = 1.0 - 2.0 * (orientation.x * orientation.x + orientation.y * orientation.y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (orientation.w * orientation.y - orientation.z * orientation.x)))
    return roll, math.asin(sinp)


def _xyz(raw_points: Any) -> list[tuple[float, float, float]]:
    # ROS 2 Humble may return a structured array; field lookup prevents mixed
    # LiDAR datatype fields from changing XYZ column positions.
    if getattr(getattr(raw_points, "dtype", None), "names", None):
        return [(float(p["x"]), float(p["y"]), float(p["z"])) for p in raw_points]
    return [(float(p[0]), float(p[1]), float(p[2])) for p in raw_points]


def replay_bag(bag: Path, annotations: dict[str, Any], max_frames_per_scene: int = 20) -> dict[str, Any]:
    """Read a rosbag deterministically without constructing any ROS publisher."""
    scenes = validate_annotations(annotations)
    lidar_topic = str(annotations["lidar_topic"])
    imu_topic = str(annotations["imu_topic"])
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except ImportError as exc:
        raise RuntimeError("ROS 2 Humble Python bag dependencies are required") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    missing = [topic for topic in (lidar_topic, imu_topic) if topic not in types]
    if missing:
        raise ValueError("bag is missing required topics: " + ", ".join(missing))
    lidar_type = get_message(types[lidar_topic])
    imu_type = get_message(types[imu_topic])

    roll = pitch = 0.0
    have_imu = False
    counts = {str(scene["id"]): 0 for scene in scenes}
    frames: list[dict[str, Any]] = []
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic == imu_topic:
            imu = deserialize_message(data, imu_type)
            roll, pitch = _euler(imu.orientation)
            have_imu = True
            continue
        if topic != lidar_topic or not have_imu:
            continue
        scene = matching_scene(scenes, timestamp_ns)
        if scene is None or counts[str(scene["id"])] >= max_frames_per_scene:
            continue
        cloud = deserialize_message(data, lidar_type)
        points = point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
        frames.append(
            {
                "id": f"{scene['id']}:{timestamp_ns}",
                "points_xyz": _xyz(points),
                "roll_rad": roll,
                "pitch_rad": pitch,
                "expected": scene["expected"],
            }
        )
        counts[str(scene["id"])] += 1

    missing_frames = [scene_id for scene_id, count in counts.items() if count == 0]
    report = evaluate_scenes(frames)
    report.update(
        {
            "source_bag": str(bag),
            "read_only_replay": True,
            "publishers_created": 0,
            "frame_counts": counts,
            "missing_annotated_scenes": missing_frames,
        }
    )
    if missing_frames:
        report["passed"] = False
        report["failures"].append("no synchronized frames for: " + ", ".join(missing_frames))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay annotated X2 sensor bags without publishers")
    parser.add_argument("bag", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames-per-scene", type=int, default=20)
    args = parser.parse_args(argv)
    document = json.loads(args.annotations.read_text(encoding="utf-8"))
    report = replay_bag(args.bag, document, max_frames_per_scene=args.max_frames_per_scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
