from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.agent_track_labeling import (
    assign_seed_splits,
    canonical_clip_id,
    detections_by_track,
    evaluate_predictions,
    fixture_id,
    reconcile_manifest,
    select_representatives,
    validate_agent_label,
)


class AgentTrackLabelingTest(unittest.TestCase):
    def test_track_filter_requires_confident_minimum_duration(self):
        frames = [
            {
                "frame": index,
                "detections": [
                    {
                        "track_id": 1,
                        "bbox": [0, 0, 10, 20],
                        "confidence": confidence,
                    },
                    {
                        "track_id": 2,
                        "bbox": [0, 0, 10, 20],
                        "confidence": 0.9,
                    },
                ],
            }
            for index, confidence in enumerate((0.45, 0.46, 0.8))
        ]
        tracks = detections_by_track(
            frames, minimum_confidence=0.45, minimum_detections=3
        )
        self.assertNotIn(1, tracks)
        self.assertEqual(len(tracks[2]), 3)

    def test_fixture_and_published_aliases_are_stable(self):
        self.assertEqual(
            fixture_id("lei-ars-20150926-h1-0093-published"),
            "lei-ars-20150926",
        )
        self.assertEqual(
            canonical_clip_id("lei-ars-20150926-h1-0093-published"),
            "lei-ars-20150926-h1-0093",
        )

    def test_representatives_cover_complete_track(self):
        rows = [
            {
                "frame": index,
                "bbox": [0, 0, 10 + index, 20 + index],
                "confidence": 0.8,
            }
            for index in range(30)
        ]
        selected = select_representatives(rows, count=6)
        self.assertEqual(len(selected), 6)
        self.assertLess(selected[0]["frame"], 6)
        self.assertGreaterEqual(selected[-1]["frame"], 25)

    def test_seed_split_retains_evaluation_tracks(self):
        tracks = [
            {
                "key": f"fixture-h1-0001:{index}",
                "fixture_id": "fixture",
                "reviewed_label": "team_a",
                "quality": float(10 - index),
            }
            for index in range(4)
        ]
        assign_seed_splits(tracks, seed_per_label=2)
        self.assertEqual(
            [row["split"] for row in tracks].count("seed"), 2
        )
        self.assertEqual(
            [row["split"] for row in tracks].count("evaluation"), 2
        )

    def test_agent_abstention_must_use_unknown(self):
        valid = {
            "participant_type": "unknown",
            "role": "unknown",
            "label": "unknown",
            "abstain": True,
            "kit_visible": False,
            "identity_visible": True,
            "official_evidence_visible": False,
            "goalkeeper_seed_available": False,
            "goalkeeper_kit_match": False,
            "matched_seed_images": [],
            "consistent_crop_count": 0,
            "primary_visual_cues": [],
            "contradicting_evidence": [],
            "reason": "Kit is occluded",
        }
        self.assertEqual(validate_agent_label(valid), valid)
        invalid = {**valid, "label": "team_a"}
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_agent_label(invalid)

    def test_goalkeeper_requires_same_side_seed(self):
        payload = {
            "participant_type": "team_b",
            "role": "goalkeeper",
            "label": "team_b_goalkeeper",
            "abstain": False,
            "kit_visible": True,
            "identity_visible": True,
            "official_evidence_visible": False,
            "goalkeeper_seed_available": True,
            "goalkeeper_kit_match": True,
            "matched_seed_images": [1],
            "consistent_crop_count": 4,
            "primary_visual_cues": ["green kit"],
            "contradicting_evidence": [],
            "reason": "Matches goalkeeper",
        }
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_agent_label(payload, {"team_a", "team_b"})

    def test_reconciliation_propagates_identity_to_every_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "clip_id": "bur-ars-20150411-h1-0128",
                        "frames": [
                            {
                                "frame": 0,
                                "detections": [
                                    {
                                        "track_id": 7,
                                        "bbox": [1, 2, 3, 4],
                                        "confidence": 0.9,
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "key": "bur-ars-20150411-h1-0128:7",
                                "clip_id": "bur-ars-20150411-h1-0128",
                                "track_id": 7,
                                "source_result": str(result),
                                "split": "evaluation",
                                "reviewed_label": "team_a",
                            }
                        ]
                    }
                )
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "key": "bur-ars-20150411-h1-0128:7",
                        "response": {"label": "team_a"},
                    }
                )
                + "\n"
            )
            output = root / "output"
            reconcile_manifest(manifest, predictions, output)
            payload = json.loads(
                (output / "bur-ars-20150411-h1-0128.json").read_text()
            )
            identity = payload["frames"][0]["detections"][0]["identity"]
            self.assertEqual(identity["label"], "team_a")
            self.assertEqual(identity["status"], "agent_proposal")

    def test_evaluation_reports_coverage_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "key": "clip:1",
                                "split": "evaluation",
                                "reviewed_label": "team_a",
                            },
                            {
                                "key": "clip:2",
                                "split": "evaluation",
                                "reviewed_label": "team_b",
                            },
                            {
                                "key": "clip:3",
                                "split": "seed",
                                "reviewed_label": "team_a",
                            },
                        ]
                    }
                )
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {"key": "clip:1", "response": {"label": "team_a"}}
                )
                + "\n"
            )
            report = evaluate_predictions(manifest, predictions)
            self.assertEqual(report["evaluation_tracks"], 2)
            self.assertEqual(report["coverage"], 0.5)
            self.assertEqual(report["accuracy_at_coverage"], 1.0)
            self.assertEqual(report["overall_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
