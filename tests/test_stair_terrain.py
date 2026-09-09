import unittest

from training.x2_stairs import terrain
from training.x2_stairs.task_spec import TaskSpec
from x2_stair_autonomy.models import StairProfile


class StairTerrainTests(unittest.TestCase):
    def test_layout_matches_sampled_domain(self):
        spec = TaskSpec()
        for seed in range(10):
            domain = spec.sample_domain(seed)
            layout = terrain.build_layout(domain)
            self.assertEqual(len(layout.steps), domain["steps"])
            self.assertAlmostEqual(layout.riser_m, domain["riser_m"])
            self.assertAlmostEqual(layout.tread_m, domain["tread_m"])
            self.assertAlmostEqual(layout.width_m, domain["width_m"])

    def test_height_increases_monotonically_with_x(self):
        spec = TaskSpec()
        domain = spec.sample_domain(3)
        layout = terrain.build_layout(domain)
        heights = [terrain.height_at(layout, step.x_center) for step in layout.steps]
        self.assertEqual(heights, sorted(heights))
        self.assertEqual(heights[-1], layout.top_z)
        self.assertEqual(terrain.height_at(layout, -1.0), 0.0)

    def test_landing_meets_profile_minimum(self):
        profile = StairProfile()
        spec = TaskSpec()
        domain = spec.sample_domain(5)
        layout = terrain.build_layout(domain, profile)
        landing_depth = layout.landing.half_depth * 2
        self.assertGreaterEqual(landing_depth, profile.min_landing_m)
        self.assertEqual(layout.landing.z_top, layout.top_z)

    def test_rejects_non_positive_geometry(self):
        with self.assertRaises(ValueError):
            terrain.build_layout({"riser_m": 0, "tread_m": 0.3, "width_m": 1.0, "steps": 3})

    def test_to_mjcf_emits_one_geom_per_step_plus_landing(self):
        spec = TaskSpec()
        domain = spec.sample_domain(7)
        layout = terrain.build_layout(domain)
        xml = terrain.to_mjcf(layout)
        self.assertEqual(xml.count("<geom"), len(layout.steps) + 1)


if __name__ == "__main__":
    unittest.main()
