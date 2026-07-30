import unittest

from scripts.build_skillcorner_phase_maps import canonical_xy, oriented_xy


class SkillCornerPhaseMapTest(unittest.TestCase):
    def test_left_to_right_coordinates_are_only_shifted(self):
        self.assertEqual(canonical_xy(-52.5, -34.0, "left_to_right"), [0.0, 0.0])

    def test_right_to_left_coordinates_are_rotated(self):
        self.assertEqual(
            canonical_xy(-52.5, -34.0, "right_to_left"),
            [105.0, 68.0],
        )

    def test_oriented_coordinates_match_video_projector_scale(self):
        self.assertEqual(
            oriented_xy(-40.0, 20.0, "left_to_right"),
            [-40.0, 20.0],
        )
        self.assertEqual(
            oriented_xy(-40.0, 20.0, "right_to_left"),
            [40.0, -20.0],
        )


if __name__ == "__main__":
    unittest.main()
