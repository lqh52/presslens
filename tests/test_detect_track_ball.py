import unittest

from scripts.detect_track_ball import bridge_short_gaps, select_trajectory


class BallTrackingTemporalTest(unittest.TestCase):
    def test_bridge_short_gap_interpolates_ball_location(self):
        trajectory = {
            0: {
                "image_xy": [10.0, 20.0],
                "ground_xy": [10.0, 22.0],
                "confidence": 0.8,
            },
            2: {
                "image_xy": [14.0, 24.0],
                "ground_xy": [14.0, 26.0],
                "confidence": 0.6,
            },
        }

        bridged = bridge_short_gaps(trajectory, maximum_gap=2)

        self.assertEqual(bridged[1]["method"], "interpolated")
        self.assertEqual(bridged[1]["image_xy"], [12.0, 22.0])
        self.assertEqual(bridged[1]["ground_xy"], [12.0, 24.0])
        self.assertAlmostEqual(bridged[1]["confidence"], 0.39)

    def test_select_trajectory_prefers_continuous_candidates(self):
        candidates = [
            [
                {"image_xy": [10.0, 10.0], "confidence": 0.75},
                {"image_xy": [90.0, 90.0], "confidence": 0.95},
            ],
            [{"image_xy": [12.0, 11.0], "confidence": 0.75}],
            [{"image_xy": [14.0, 12.0], "confidence": 0.75}],
        ]

        selected = select_trajectory(candidates, width=120, height=90)

        self.assertEqual(
            [selected[index]["image_xy"] for index in range(3)],
            [[10.0, 10.0], [12.0, 11.0], [14.0, 12.0]],
        )


if __name__ == "__main__":
    unittest.main()
