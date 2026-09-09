from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any


class MissionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    MONITORING = "MONITORING"
    READY = "READY"
    ARMED = "ARMED"
    APPROACHING = "APPROACHING"
    CLIMBING = "CLIMBING"
    DESCENDING = "DESCENDING"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    STOPPING = "STOPPING"
    FAULT = "FAULT"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


@dataclass(slots=True)
class StairProfile:
    profile_id: str = "standard-straight-v1"
    min_riser_m: float = 0.150
    max_riser_m: float = 0.190
    min_tread_m: float = 0.250
    max_tread_m: float = 0.320
    min_width_m: float = 0.900
    min_landing_m: float = 0.900
    min_confidence: float = 0.80
    approved: bool = False


@dataclass(slots=True)
class StairEstimate:
    available: bool = False
    supported: bool = False
    direction: str | None = None
    distance_m: float | None = None
    yaw_deg: float | None = None
    riser_m: float | None = None
    tread_m: float | None = None
    width_m: float | None = None
    step_count: int = 0
    landing_detected: bool = False
    confidence: float = 0.0
    rejection_reason: str | None = "no estimate"
    stamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SafetySnapshot:
    connected: bool = False
    sensors_fresh: bool = False
    robot_mode_ok: bool = False
    input_source_owned: bool = False
    checklist_complete: bool = False
    hardware_estop_confirmed: bool = False
    gantry_confirmed: bool = False
    spotter_confirmed: bool = False
    vendor_approval: bool = False
    profile_approved: bool = False
    perception_ok: bool = False
    tilt_ok: bool = True
    joint_limits_ok: bool = True
    command_rate_approved: bool = False
    reasons: list[str] = field(default_factory=list)
    stamp: float = field(default_factory=time.time)

    def blockers(self) -> list[str]:
        checks = {
            "robot disconnected": self.connected,
            "sensor data is stale": self.sensors_fresh,
            "robot mode is not approved": self.robot_mode_ok,
            "control input source is not owned": self.input_source_owned,
            "operator checklist is incomplete": self.checklist_complete,
            "hardware E-stop was not confirmed": self.hardware_estop_confirmed,
            "gantry was not confirmed": self.gantry_confirmed,
            "spotter was not confirmed": self.spotter_confirmed,
            "AgiBot vendor approval is missing": self.vendor_approval,
            "stair profile is not approved": self.profile_approved,
            "stair estimate is unsafe or low confidence": self.perception_ok,
            "robot tilt exceeds limit": self.tilt_ok,
            "joint state exceeds limit": self.joint_limits_ok,
            "command frequency/limits are not vendor-approved": self.command_rate_approved,
        }
        return [reason for reason, passed in checks.items() if not passed] + list(self.reasons)

    @property
    def ready(self) -> bool:
        return not self.blockers()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blockers"] = self.blockers()
        value["ready"] = self.ready
        return value

