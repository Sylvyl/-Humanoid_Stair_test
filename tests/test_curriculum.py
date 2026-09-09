import unittest

from training.x2_stairs.task_spec import StairDomain, direction_for_stage, domain_for_stage


class CurriculumStageTests(unittest.TestCase):
    def test_single_step_stages_have_one_step_and_forced_direction(self):
        up = domain_for_stage("single_step_up")
        self.assertEqual(up.steps, (1, 1))
        self.assertEqual(direction_for_stage("single_step_up"), "up")

        down = domain_for_stage("single_step_down")
        self.assertEqual(down.steps, (1, 1))
        self.assertEqual(direction_for_stage("single_step_down"), "down")

    def test_three_steps_stage_has_no_forced_direction(self):
        domain = domain_for_stage("three_steps")
        self.assertEqual(domain.steps, (3, 3))
        self.assertIsNone(direction_for_stage("three_steps"))

    def test_random_straight_flight_is_full_default_domain(self):
        self.assertEqual(domain_for_stage("random_straight_flight"), StairDomain())

    def test_unimplemented_stages_raise_clearly(self):
        with self.assertRaises(NotImplementedError):
            domain_for_stage("flat_balance")
        with self.assertRaises(NotImplementedError):
            domain_for_stage("noisy_sensor_student")

    def test_unknown_stage_raises_value_error(self):
        with self.assertRaises(ValueError):
            domain_for_stage("not_a_real_stage")


if __name__ == "__main__":
    unittest.main()
