from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock
import json
import time

from .models import SafetySnapshot, StairEstimate, StairProfile
from .release_gate import HARDWARE_STAGES, evaluate_release_gate
from .supervisor import MissionSupervisor


class RuntimeState:
    def __init__(self, root: Path):
        self.root = root
        self.lock = RLock()
        self.profile = self._load_profile(root / "config" / "stair_profiles.json")
        self.stairs = StairEstimate()
        self.telemetry: dict = {
            "robot": {"host": "10.0.1.41", "connected": False, "mode": None, "input_source": None},
            "power": {"available": False},
            "joints": {"available": False, "count": 0, "violations": []},
            "sensors": {},
            "host": {"available": False},
            "recording": {"active": False, "bag": None},
        }
        self.checklist = {"hardware_estop": False, "gantry": False, "spotter": False}
        self.approvals = self._load_approvals(root / "config" / "vendor_approval.json")
        self.supervisor = MissionSupervisor(commands_enabled=False)
        self.training_runs = self._load_training_runs()
        self.refresh_safety()

    @staticmethod
    def _load_profile(path: Path) -> StairProfile:
        data = json.loads(path.read_text(encoding="utf-8"))["profiles"][0]
        return StairProfile(**data)

    @staticmethod
    def _load_approvals(path: Path) -> dict:
        if not path.exists():
            return {"approved": False, "command_rate_hz": None, "joint_limits_reviewed": False}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_training_runs(self) -> list[dict]:
        path = self.root / "training" / "runs.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def compatibility_summary(self) -> dict:
        path = self.root / "validation" / "pc2_compatibility.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"available": False, "passed_shadow": False, "passed_hardware": False, "roles": {}}
        return {
            "available": True,
            "created_at": report.get("created_at"),
            "passed_shadow": report.get("passed_shadow") is True,
            "passed_hardware": report.get("passed_hardware") is True,
            "roles": report.get("roles", {}),
            "required_missing": report.get("required_missing", []),
            "power_source_missing": report.get("power_source_missing") is True,
            "topic_count": len(report.get("topics", {})),
        }

    def refresh_safety(self) -> SafetySnapshot:
        with self.lock:
            robot = self.telemetry["robot"]
            sensors = self.telemetry["sensors"]
            now = time.time()
            release_gate = evaluate_release_gate(self.root)
            required = ("depth", "lidar", "imu_torso", "joints")
            fresh = all(now - float(sensors.get(name, {}).get("stamp", 0)) <= 0.25 for name in required)
            safety = SafetySnapshot(
                connected=bool(robot.get("connected")),
                sensors_fresh=fresh,
                robot_mode_ok=robot.get("mode") in {"STAND_DEFAULT", "LOCOMOTION_DEFAULT"},
                input_source_owned=robot.get("input_source") == "x2_stair_supervisor_v1",
                checklist_complete=all(self.checklist.values()),
                hardware_estop_confirmed=self.checklist["hardware_estop"],
                gantry_confirmed=self.checklist["gantry"],
                spotter_confirmed=self.checklist["spotter"],
                vendor_approval=bool(self.approvals.get("approved")),
                profile_approved=bool(self.profile.approved),
                perception_ok=self.stairs.supported and self.stairs.confidence >= self.profile.min_confidence,
                tilt_ok=not bool(self.telemetry.get("tilt_fault")),
                joint_limits_ok=not bool(self.telemetry["joints"].get("violations")),
                command_rate_approved=bool(self.approvals.get("command_rate_hz"))
                    and bool(self.approvals.get("joint_limits_reviewed")),
                reasons=[f"release gate: {reason}" for reason in release_gate.blockers],
            )
            self.supervisor.update_safety(safety)
            return safety

    def set_checklist(self, values: dict) -> None:
        with self.lock:
            for key in self.checklist:
                if key in values:
                    self.checklist[key] = values[key] is True
            self.refresh_safety()

    def snapshot(self) -> dict:
        with self.lock:
            safety = self.refresh_safety()
            release_gate = evaluate_release_gate(self.root)
            return {
                "stairs": self.stairs.to_dict(),
                "mission": self.supervisor.snapshot(),
                "safety": safety.to_dict(),
                "telemetry": deepcopy(self.telemetry),
                "checklist": dict(self.checklist),
                "profile": {
                    "id": self.profile.profile_id,
                    "approved": self.profile.approved,
                    "riser_m": [self.profile.min_riser_m, self.profile.max_riser_m],
                    "tread_m": [self.profile.min_tread_m, self.profile.max_tread_m],
                    "min_width_m": self.profile.min_width_m,
                },
                "commands_compiled": False,
                "release_gate": release_gate.to_dict(),
                "hardware_stages": list(HARDWARE_STAGES),
                "compatibility": self.compatibility_summary(),
            }
