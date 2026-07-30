from __future__ import annotations

import unittest

from scripts.attack_direction import (
    aggregate_clip_calibrations,
    calibrate_detections,
)
from scripts.calibrate_attack_directions import registry_payload


def row(
    *,
    team: str,
    role: str,
    track_id: int,
    frame: int,
    x: float,
) -> dict:
    return {
        "team": team,
        "role": role,
        "track_id": track_id,
        "image_id": frame,
        "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": 0.0},
    }


def strong_clip(*, reversed_internal_identity: bool = False) -> list[dict]:
    rows = []
    left_player_x = (18.0, 27.0, 36.0) if reversed_internal_identity else (
        -36.0,
        -27.0,
        -18.0,
    )
    right_player_x = tuple(-x for x in left_player_x)
    left_goalkeeper_x = 47.0 if reversed_internal_identity else -47.0
    right_goalkeeper_x = -left_goalkeeper_x
    for frame in range(5):
        for offset, x in enumerate(left_player_x):
            rows.append(
                row(
                    team="left",
                    role="player",
                    track_id=offset + 1,
                    frame=frame,
                    x=x + frame * 0.1,
                )
            )
        for offset, x in enumerate(right_player_x):
            rows.append(
                row(
                    team="right",
                    role="player",
                    track_id=offset + 11,
                    frame=frame,
                    x=x - frame * 0.1,
                )
            )
        rows.append(
            row(
                team="left",
                role="goalkeeper",
                track_id=100,
                frame=frame,
                x=left_goalkeeper_x,
            )
        )
        rows.append(
            row(
                team="right",
                role="goalkeeper",
                track_id=200,
                frame=frame,
                x=right_goalkeeper_x,
            )
        )
    return rows


class AttackDirectionTest(unittest.TestCase):
    def test_goalkeepers_and_formation_calibrate_raw_orientation(self):
        evidence = calibrate_detections(strong_clip())
        self.assertTrue(evidence["confident"])
        self.assertEqual(evidence["directions"], {"left": 1, "right": -1})
        methods = {item["method"] for item in evidence["methods"]}
        self.assertIn("paired_goalkeeper_anchors", methods)
        self.assertIn("formation_track_median_order", methods)

    def test_internal_identity_can_have_the_reverse_raw_direction(self):
        evidence = calibrate_detections(
            strong_clip(reversed_internal_identity=True)
        )
        self.assertTrue(evidence["confident"])
        self.assertEqual(evidence["directions"], {"left": -1, "right": 1})

    def test_ambiguous_spatial_evidence_abstains(self):
        detections = []
        for team, base_track in (("left", 0), ("right", 10)):
            for track in range(3):
                detections.append(
                    row(
                        team=team,
                        role="player",
                        track_id=base_track + track,
                        frame=0,
                        x=0.2 * track,
                    )
                )
        evidence = calibrate_detections(detections)
        self.assertFalse(evidence["confident"])
        self.assertIsNone(evidence["directions"])
        self.assertIn("abstained", evidence["status"])

    def test_same_goalkeeper_track_cannot_vote_for_both_teams(self):
        detections = []
        for frame in range(4):
            detections.extend(
                [
                    row(
                        team="left",
                        role="goalkeeper",
                        track_id=99,
                        frame=frame,
                        x=-47.0,
                    ),
                    row(
                        team="right",
                        role="goalkeeper",
                        track_id=99,
                        frame=frame,
                        x=47.0,
                    ),
                ]
            )
        evidence = calibrate_detections(detections)
        self.assertFalse(evidence["confident"])
        self.assertIsNone(evidence["directions"])

    def test_match_half_needs_multiple_agreeing_clips(self):
        positive = {
            "confident": True,
            "confidence": 0.9,
            "directions": {"left": 1, "right": -1},
        }
        negative = {
            "confident": True,
            "confidence": 0.9,
            "directions": {"left": -1, "right": 1},
        }
        agreed = aggregate_clip_calibrations([positive, positive])
        self.assertTrue(agreed["confident"])
        split = aggregate_clip_calibrations([positive, negative])
        self.assertFalse(split["confident"])
        self.assertEqual(split["status"], "abstained_conflicting_clips")

    def test_registry_omits_direction_when_half_abstains(self):
        payload = registry_payload(
            {("match", 1): []},
            {("match", 1)},
            minimum_clips=2,
            minimum_vote_confidence=0.75,
        )
        half = payload["matches"]["match"]["halves"]["1"]
        self.assertFalse(half["direction_confident"])
        self.assertIsNone(half["attacking_direction"])


if __name__ == "__main__":
    unittest.main()
