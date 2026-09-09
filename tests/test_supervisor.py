import time
import unittest

from x2_stair_autonomy.models import Direction, MissionState, SafetySnapshot
from x2_stair_autonomy.supervisor import MissionSupervisor


def ready_safety(**overrides):
    values = dict(
        connected=True,
        sensors_fresh=True,
        robot_mode_ok=True,
        input_source_owned=True,
        checklist_complete=True,
        hardware_estop_confirmed=True,
        gantry_confirmed=True,
        spotter_confirmed=True,
        vendor_approval=True,
        profile_approved=True,
        perception_ok=True,
        tilt_ok=True,
        joint_limits_ok=True,
        command_rate_approved=True,
    )
    values.update(overrides)
    return SafetySnapshot(**values)


class SupervisorTests(unittest.TestCase):
    def test_happy_path_stays_shadow_only(self):
        supervisor = MissionSupervisor(commands_enabled=True)
        supervisor.update_safety(ready_safety())
        self.assertEqual(supervisor.snapshot()["state"], "READY")
        supervisor.arm()
        supervisor.start("test-up", Direction.UP)
        result = supervisor.snapshot()
        self.assertEqual(result["state"], "APPROACHING")
        self.assertTrue(result["shadow_mode"])
        self.assertFalse(result["command_output_enabled"])

    def test_arm_fails_with_blocker(self):
        supervisor = MissionSupervisor()
        supervisor.update_safety(ready_safety(vendor_approval=False))
        with self.assertRaises(RuntimeError):
            supervisor.arm()

    def test_watchdog_latches_fault(self):
        supervisor = MissionSupervisor(heartbeat_timeout_s=0.01)
        supervisor.update_safety(ready_safety())
        supervisor.arm()
        time.sleep(0.02)
        supervisor.tick()
        self.assertEqual(supervisor.snapshot()["state"], "FAULT")
        self.assertIn("heartbeat", supervisor.snapshot()["fault"])

    def test_invalid_transition_rejected(self):
        supervisor = MissionSupervisor()
        supervisor.update_safety(ready_safety())
        supervisor.arm()
        with self.assertRaises(RuntimeError):
            supervisor.advance(MissionState.LANDING)

    def test_abort_is_always_fail_closed(self):
        supervisor = MissionSupervisor()
        supervisor.abort()
        self.assertEqual(supervisor.snapshot()["state"], "FAULT")
        self.assertFalse(supervisor.snapshot()["command_output_enabled"])
