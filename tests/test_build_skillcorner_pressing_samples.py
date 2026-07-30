import unittest

import numpy as np

from scripts.build_skillcorner_pressing_samples import (
    graph_frame,
    opposite_attacking_side,
    sample_frame_ids,
    truth,
)


class SkillCornerPressingSamplesTest(unittest.TestCase):
    def test_sample_frame_ids_include_context_and_event(self):
        self.assertEqual(sample_frame_ids(100, 120), [95, 102, 110, 118, 125])

    def test_pressing_direction_is_inverted_for_possession_canonical_view(self):
        self.assertEqual(
            opposite_attacking_side("left_to_right"), "right_to_left"
        )
        self.assertEqual(
            opposite_attacking_side("right_to_left"), "left_to_right"
        )

    def test_truth_accepts_csv_and_json_values(self):
        self.assertTrue(truth("True"))
        self.assertTrue(truth(True))
        self.assertFalse(truth("False"))
        self.assertFalse(truth(None))

    def test_graph_frame_matches_video_graph_schema(self):
        frame = {
            "frame": 10,
            "ball_xy": [0.0, 0.0],
            "players": [
                {
                    "track_id": 1,
                    "team": "team_a",
                    "goalkeeper": False,
                    "xy": [0.5, 0.0],
                },
                {
                    "track_id": 2,
                    "team": "team_b",
                    "goalkeeper": True,
                    "xy": [10.0, 0.0],
                },
            ],
        }
        features, mask, metadata = graph_frame(frame, {})
        self.assertEqual(features.shape, (23, 13))
        self.assertEqual(mask.shape, (23,))
        self.assertEqual(int(mask.sum()), 3)
        np.testing.assert_allclose(features[0, :2], [53.0 / 105.0, 0.5])
        self.assertEqual(features[0, 4], 1)
        self.assertEqual(features[0, 8], 1)
        self.assertEqual(features[0, 12], 1)
        self.assertEqual(features[1, 5], 1)
        self.assertEqual(features[1, 7], 1)
        self.assertEqual(features[2, 6], 1)
        self.assertEqual(features[2, 11], 1)
        self.assertEqual(metadata["visible_nodes"], 3)


if __name__ == "__main__":
    unittest.main()
