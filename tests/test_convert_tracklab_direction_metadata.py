from __future__ import annotations

import unittest

try:
    from scripts.convert_tracklab_state import (
        apply_resolved_team_assignments,
        explicit_direction_metadata,
    )
except ModuleNotFoundError as error:
    if error.name not in {"cv2", "ultralytics"}:
        raise
    apply_resolved_team_assignments = None
    explicit_direction_metadata = None


@unittest.skipIf(
    explicit_direction_metadata is None,
    "requires the sn-gamestate conversion environment",
)
class ConvertTrackLabDirectionMetadataTest(unittest.TestCase):
    def test_manual_cli_direction_retains_unit_confidence(self):
        metadata = explicit_direction_metadata(
            {"left": -1, "right": 1}
        )
        self.assertEqual(metadata["source"], "match_half_metadata")
        self.assertEqual(metadata["status"], "calibrated")
        self.assertTrue(metadata["confident"])
        self.assertEqual(metadata["confidence"], 1.0)
        self.assertEqual(metadata["evidence"]["confidence"], 1.0)

    def test_registry_aggregate_provenance_is_preserved(self):
        metadata = explicit_direction_metadata(
            {"left": 1, "right": -1},
            {
                "source": "direction_registry_match_half_override",
                "status": "calibrated",
                "confident": True,
                "confidence": 0.8125,
                "evidence": {
                    "aggregate": {
                        "clips_available": 7,
                        "clips_usable": 5,
                        "orientation_clip_counts": {"-1": 1, "1": 4},
                    }
                },
            },
        )
        self.assertEqual(
            metadata["source"],
            "direction_registry_match_half_override",
        )
        self.assertEqual(metadata["confidence"], 0.8125)
        self.assertEqual(metadata["evidence"]["confidence"], 0.8125)
        aggregate = metadata["evidence"]["provenance"]["aggregate"]
        self.assertEqual(aggregate["clips_usable"], 5)
        self.assertEqual(
            aggregate["orientation_clip_counts"],
            {"-1": 1, "1": 4},
        )

    def test_neutral_assignment_map_is_fail_closed(self):
        import pandas as pd

        detections = pd.DataFrame(
            [
                {
                    "role": "player",
                    "track_id": 7,
                    "team": "right",
                    "team_cluster": 0,
                },
                {
                    "role": "goalkeeper",
                    "track_id": 8,
                    "team": "right",
                    "team_cluster": 1,
                },
                {
                    "role": "referee",
                    "track_id": 9,
                    "team": "left",
                    "team_cluster": 0,
                },
            ]
        )
        evidence = apply_resolved_team_assignments(
            detections,
            {7: "left"},
            neutral_fail_closed=True,
        )
        self.assertEqual(detections.loc[0, "team"], "left")
        self.assertEqual(detections.loc[1, "team"], "ignore")
        # Non-athletes are outside the graph-team assignment path.
        self.assertEqual(detections.loc[2, "team"], "left")
        self.assertEqual(
            evidence,
            {
                "mode": "persisted_map_fail_closed",
                "persisted_side_tracks": 1,
                "graph_side_tracks": 1,
                "unmapped_athlete_tracks_excluded": [8],
            },
        )


if __name__ == "__main__":
    unittest.main()
