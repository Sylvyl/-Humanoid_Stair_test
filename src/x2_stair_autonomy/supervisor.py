from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
import time

from .models import Direction, MissionState, SafetySnapshot


ACTIVE_STATES = {
    MissionState.ARMED,
    MissionState.APPROACHING,
    MissionState.CLIMBING,
    MissionState.DESCENDING,
    MissionState.LANDING,
}


@dataclass(slots=True)
class MissionSnapshot:
    state: MissionState = MissionState.DISCONNECTED
    direction: Direction | None = None
    mission_id: str | None = None
    fault: str | None = None
    command_output_enabled: bool = False
    shadow_mode: bool = True
    last_heartbeat_age_ms: float | None = None
    updated_at: float = 0.0

    def to_dict(self):
        value = asdict(self)
        value["state"] = self.state.value
        value["direction"] = self.direction.value if self.direction else None
        return value


class MissionSupervisor:
    """Fail-closed mission state machine; it does not publish robot commands."""

    def __init__(self, *, commands_enabled: bool = False, heartbeat_timeout_s: float = 0.35):
        self.lock = RLock()
        self.commands_enabled = commands_enabled
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.safety = SafetySnapshot()
        self.mission = MissionSnapshot(command_output_enabled=False, shadow_mode=True, updated_at=time.time())
        self._heartbeat_at: float | None = None

    def update_safety(self, safety: SafetySnapshot) -> None:
        with self.lock:
            self.safety = safety
            if not safety.connected and self.mission.state != MissionState.DISCONNECTED:
                self._fault("robot connection lost")
            elif safety.connected and self.mission.state == MissionState.DISCONNECTED:
                self.mission.state = MissionState.MONITORING
            if safety.ready and self.mission.state == MissionState.MONITORING:
                self.mission.state = MissionState.READY
            elif not safety.ready and self.mission.state == MissionState.READY:
                self.mission.state = MissionState.MONITORING
            elif not safety.ready and self.mission.state in ACTIVE_STATES:
                self._fault("; ".join(safety.blockers()))
            self.mission.updated_at = time.time()

    def arm(self) -> None:
        with self.lock:
            if self.mission.state == MissionState.FAULT:
                raise RuntimeError("fault must be cleared by a fresh completed checklist")
            if self.mission.state != MissionState.READY:
                raise RuntimeError(f"cannot arm from {self.mission.state.value}")
            if not self.safety.ready:
                raise RuntimeError("cannot arm: " + "; ".join(self.safety.blockers()))
            self.mission.state = MissionState.ARMED
            self._heartbeat_at = time.monotonic()
            self.mission.updated_at = time.time()

    def start(self, mission_id: str, direction: Direction) -> None:
        with self.lock:
            if self.mission.state != MissionState.ARMED:
                raise RuntimeError("mission must be armed before start")
            if not self.safety.ready:
                raise RuntimeError("safety gate changed before start")
            self.mission.mission_id = mission_id
            self.mission.direction = direction
            self.mission.state = MissionState.APPROACHING
            # Real output cannot be enabled by configuration alone.  The build
            # intentionally has no command adapter until vendor approval.
            self.mission.command_output_enabled = False
            self.mission.shadow_mode = True
            self._heartbeat_at = time.monotonic()
            self.mission.updated_at = time.time()

    def heartbeat(self) -> None:
        with self.lock:
            if self.mission.state not in ACTIVE_STATES:
                raise RuntimeError("heartbeat accepted only for an armed or active mission")
            self._heartbeat_at = time.monotonic()

    def advance(self, state: MissionState) -> None:
        allowed = {
            MissionState.APPROACHING: {MissionState.CLIMBING, MissionState.DESCENDING},
            MissionState.CLIMBING: {MissionState.LANDING},
            MissionState.DESCENDING: {MissionState.LANDING},
            MissionState.LANDING: {MissionState.COMPLETE},
        }
        with self.lock:
            if state not in allowed.get(self.mission.state, set()):
                raise RuntimeError(f"invalid transition {self.mission.state.value} -> {state.value}")
            self.mission.state = state
            self.mission.updated_at = time.time()

    def abort(self, reason: str = "operator software stop") -> None:
        with self.lock:
            self.mission.state = MissionState.STOPPING
            self.mission.command_output_enabled = False
            self.mission.fault = reason
            self.mission.updated_at = time.time()
            self.mission.state = MissionState.FAULT

    def disarm(self) -> None:
        with self.lock:
            if self.mission.state in {MissionState.APPROACHING, MissionState.CLIMBING,
                                      MissionState.DESCENDING, MissionState.LANDING}:
                raise RuntimeError("abort an active mission before disarming")
            self.mission = MissionSnapshot(
                state=MissionState.READY if self.safety.ready else MissionState.MONITORING,
                command_output_enabled=False,
                shadow_mode=True,
                updated_at=time.time(),
            )
            self._heartbeat_at = None

    def clear_fault(self) -> None:
        with self.lock:
            if self.mission.state != MissionState.FAULT:
                return
            if not self.safety.ready:
                raise RuntimeError("cannot clear fault: " + "; ".join(self.safety.blockers()))
            self.mission = MissionSnapshot(state=MissionState.READY, updated_at=time.time())
            self._heartbeat_at = None

    def tick(self) -> None:
        with self.lock:
            if self.mission.state in ACTIVE_STATES and self._heartbeat_at is not None:
                age = time.monotonic() - self._heartbeat_at
                self.mission.last_heartbeat_age_ms = round(age * 1000, 1)
                if age > self.heartbeat_timeout_s:
                    self._fault("operator heartbeat lost")

    def _fault(self, reason: str) -> None:
        self.mission.state = MissionState.FAULT
        self.mission.fault = reason
        self.mission.command_output_enabled = False
        self.mission.shadow_mode = True
        self.mission.updated_at = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return self.mission.to_dict()

