import unittest

from scripts.infer_tactical_patterns import frame_pattern, temporally_stabilize


def player(track_id, x, y):
    return {"track_id": track_id, "x": x, "y": y}


class TacticalPatternInferenceTest(unittest.TestCase):
    def test_detects_touchline_trap_with_two_nearby_defenders(self):
        objects = [
            player(1, 0, 30),
            player(2, -15, 20),
            player(3, 4, 29),
            player(4, 7, 27),
            player(5, 25, 0),
        ]
        identities = {
            1: "team_a",
            2: "team_a",
            3: "team_b",
            4: "team_b",
            5: "team_b",
        }

        result = frame_pattern(
            objects, {"pitch_xy": [0.5, 30.0]}, identities
        )

        self.assertEqual(result["label"], "touchline_trap")
        self.assertEqual(result["possession_team"], "team_a")
        self.assertGreaterEqual(result["pressure"]["within_12m"], 2)

    def test_abstains_when_ball_is_missing(self):
        result = frame_pattern(
            [player(1, 0, 0), player(2, 1, 0), player(3, 2, 0), player(4, 3, 0)],
            None,
            {1: "team_a", 2: "team_a", 3: "team_b", 4: "team_b"},
        )
        self.assertEqual(result["label"], "abstain")

    def test_temporal_vote_removes_single_frame_flicker(self):
        rows = [
            {"pattern": {"label": "compact_block", "display": "Compact block"}},
            {"pattern": {"label": "low_pressure", "display": "Low pressure"}},
            {"pattern": {"label": "compact_block", "display": "Compact block"}},
        ]

        temporally_stabilize(rows, radius=1)

        self.assertEqual(rows[1]["pattern"]["label"], "compact_block")
        self.assertEqual(rows[1]["pattern"]["raw_label"], "low_pressure")


if __name__ == "__main__":
    unittest.main()
