import os
import unittest

from training.x2_stairs import robot_reference as ref


class RobotReferenceTests(unittest.TestCase):
    def test_joint_order_is_unique_and_complete(self):
        self.assertEqual(len(ref.JOINT_ORDER), 29)
        self.assertEqual(len(set(ref.JOINT_ORDER)), 29)

    def test_every_joint_has_defaults_gains_and_limits(self):
        for name in ref.JOINT_ORDER:
            self.assertIn(name, ref.DEFAULT_DOF_POS)
            self.assertIn(name, ref.KP)
            self.assertIn(name, ref.KD)
            self.assertIn(name, ref.JOINT_LIMITS_RAD)
            self.assertIn(name, ref.JOINT_TORQUE_LIMITS_NM)
            lo, hi = ref.JOINT_LIMITS_RAD[name]
            self.assertLess(lo, hi)
            self.assertGreater(ref.JOINT_TORQUE_LIMITS_NM[name], 0)
            self.assertGreater(ref.KP[name], 0)
            self.assertGreater(ref.KD[name], 0)

    def test_resolve_sdk_path_fails_closed_without_env(self):
        old = os.environ.pop("AIMDK_SDK_ROOT", None)
        try:
            with self.assertRaises(RuntimeError):
                ref.resolve_sdk_path("does-not-matter")
        finally:
            if old is not None:
                os.environ["AIMDK_SDK_ROOT"] = old

    def test_resolve_sdk_path_fails_closed_on_missing_path(self):
        old = os.environ.get("AIMDK_SDK_ROOT")
        os.environ["AIMDK_SDK_ROOT"] = "Z:/definitely-does-not-exist"
        try:
            with self.assertRaises(RuntimeError):
                ref.resolve_sdk_path("also-missing")
        finally:
            if old is not None:
                os.environ["AIMDK_SDK_ROOT"] = old
            else:
                os.environ.pop("AIMDK_SDK_ROOT", None)


if __name__ == "__main__":
    unittest.main()
