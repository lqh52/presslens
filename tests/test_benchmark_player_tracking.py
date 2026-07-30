from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_player_tracking import (
    box_iou,
    continuity_metrics,
    labelled_metrics,
    load_soccernet_truth,
)


class PlayerTrackingBenchmarkTest(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(
            box_iou([0, 0, 10, 10], [5, 0, 15, 10]),
            1 / 3,
        )

    def test_continuity_metrics_detect_fragmentation(self):
        frames = [
            {"frame": 0, "detections": [{"track_id": 1}, {"track_id": 2}]},
            {"frame": 1, "detections": [{"track_id": 1}, {"track_id": 3}]},
            {"frame": 2, "detections": [{"track_id": 1}]},
        ]
        metrics = continuity_metrics(frames)
        self.assertEqual(metrics["unique_tracks"], 3)
        self.assertEqual(metrics["median_players_per_frame"], 2)
        self.assertAlmostEqual(metrics["adjacent_id_retention"], 2 / 4)

    def test_untracked_detections_count_as_boxes_not_tracks(self):
        metrics = continuity_metrics(
            [
                {
                    "frame": 0,
                    "detections": [
                        {"track_id": 7},
                        {"track_id": None, "tracking_status": "untracked"},
                    ],
                }
            ]
        )
        self.assertEqual(metrics["detections"], 2)
        self.assertEqual(metrics["mean_players_per_frame"], 2)
        self.assertEqual(metrics["unique_tracks"], 1)

    def test_load_truth_and_labelled_metrics(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Labels-GameState.json"
            path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image_id": "1001",
                                "file_name": "000001.jpg",
                                "is_labeled": True,
                            }
                        ],
                        "annotations": [
                            {
                                "image_id": "1001",
                                "track_id": 7,
                                "category_id": 1,
                                "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 40},
                            },
                            {
                                "image_id": "1001",
                                "track_id": 8,
                                "category_id": 3,
                                "bbox_image": {"x": 0, "y": 0, "w": 5, "h": 5},
                            },
                        ],
                    }
                )
            )
            truth = load_soccernet_truth(path)
        self.assertEqual(truth[0][0]["bbox"], [10.0, 20.0, 40.0, 60.0])
        metrics = labelled_metrics(
            [
                {
                    "frame": 0,
                    "detections": [
                        {"track_id": 4, "bbox": [10, 20, 40, 60]},
                        {"track_id": 5, "bbox": [100, 100, 110, 110]},
                    ],
                }
            ],
            truth,
            0.5,
        )
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
