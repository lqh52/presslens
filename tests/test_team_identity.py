from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.team_identity import (
    aggregate_track_embedding,
    explicit_match_team_config,
    infer_cluster_to_internal_from_tracklab_team,
    infer_nonparticipant_track_ids,
    load_match_team_config,
    load_linear_model,
    neutral_match_team_config,
    predict_club,
    resolve_team_assignments,
)


class TeamIdentityTest(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(
            [
                {
                    "track_id": 7,
                    "role": "player",
                    "embeddings": np.asarray([[3.0, 0.0]], dtype=np.float32),
                },
                {
                    "track_id": 7,
                    "role": "player",
                    "embeddings": np.asarray([[2.0, 0.0]], dtype=np.float32),
                },
                {
                    "track_id": 9,
                    "role": "goalkeeper",
                    "embeddings": np.asarray([[0.0, 3.0]], dtype=np.float32),
                },
            ]
        )

    def artifact(self, directory: Path) -> Path:
        path = directory / "team.npz"
        np.savez_compressed(
            path,
            coef=np.asarray([[4.0, 0.0]], dtype=np.float32),
            intercept=np.asarray([0.0], dtype=np.float32),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            classes=np.asarray(["arsenal", "burnley"]),
            threshold=np.asarray([0.8], dtype=np.float32),
        )
        return path

    def test_embedding_is_robustly_normalized(self):
        result = aggregate_track_embedding(self.rows().query("track_id == 7"))
        np.testing.assert_allclose(result, np.asarray([1.0, 0.0]), atol=1e-6)

    def test_binary_linear_model_respects_class_order(self):
        with tempfile.TemporaryDirectory() as raw:
            model = load_linear_model(self.artifact(Path(raw)))
            label, score = predict_club(model, np.asarray([1.0, 0.0]))
            self.assertEqual(label, "burnley")
            self.assertGreater(score, 0.98)

    def test_manual_label_wins_and_goalkeeper_keeps_fallback(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            labels = directory / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": {
                            "h1-128:7": {
                                "key": "h1-128:7",
                                "label": "arsenal",
                            }
                        }
                    }
                )
            )
            resolved, evidence = resolve_team_assignments(
                {7: "right", 9: "left"},
                self.rows(),
                sequence_id="h1-128",
                club_to_internal={"arsenal": "left", "burnley": "right"},
                labels_path=labels,
                model_path=self.artifact(directory),
            )
            self.assertEqual(resolved[7], "left")
            self.assertEqual(resolved[9], "left")
            self.assertEqual(evidence["manual_tracks"], 1)
            self.assertEqual(evidence["model_tracks"], 0)

    def test_sequence_key_prevents_cross_clip_override(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            labels = directory / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": {
                            "h2-128:7": {
                                "key": "h2-128:7",
                                "label": "arsenal",
                            }
                        }
                    }
                )
            )
            resolved, _ = resolve_team_assignments(
                {7: "right", 9: "left"},
                self.rows(),
                sequence_id="h1-128",
                club_to_internal={"arsenal": "left", "burnley": "right"},
                labels_path=labels,
                model_path=None,
            )
            self.assertEqual(resolved[7], "right")

    def test_manual_ignore_excludes_track(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            labels = directory / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": {
                            "h1-128:7": {
                                "key": "h1-128:7",
                                "label": "ignore",
                            }
                        }
                    }
                )
            )
            resolved, evidence = resolve_team_assignments(
                {7: "right", 9: "left"},
                self.rows(),
                sequence_id="h1-128",
                club_to_internal={"arsenal": "left", "burnley": "right"},
                labels_path=labels,
                model_path=None,
            )
            self.assertEqual(resolved[7], "ignore")
            self.assertEqual(evidence["manual_excluded_tracks"], 1)

    def test_unreviewed_registry_stays_neutral(self):
        match_id = (
            "england_epl/2016-2017/"
            "2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea"
        )
        config = load_match_team_config(
            Path("data/annotations/team_identity_registry.example.json"),
            match_id,
        )
        self.assertEqual(config["mapping_status"], "unreviewed")
        self.assertEqual(config["team_names"], {"left": "Team A", "right": "Team B"})
        self.assertEqual(config["club_to_internal"], {})
        self.assertIsNone(config["cluster_to_internal"])

    def test_unreviewed_mapping_uses_each_sequence_tracklab_team_mode(self):
        detections = pd.DataFrame(
            [
                {"role": "player", "team_cluster": 0, "team": "right"},
                {"role": "player", "team_cluster": 0, "team": "right"},
                {"role": "player", "team_cluster": 1, "team": "left"},
                {"role": "player", "team_cluster": 1, "team": "left"},
            ]
        )
        mapping, evidence = infer_cluster_to_internal_from_tracklab_team(
            detections
        )
        self.assertEqual(mapping, {0: "right", 1: "left"})
        self.assertEqual(
            evidence["source"], "tracklab_team_mode_per_sequence"
        )

    def test_unreviewed_registry_ignores_unsafe_global_cluster_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = Path(raw) / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "matches": {
                            "fixture": {
                                "mapping_status": "unreviewed",
                                "cluster_to_internal": {
                                    "0": "right",
                                    "1": "left",
                                },
                            }
                        },
                    }
                )
            )
            config = load_match_team_config(
                registry,
                "fixture",
                sequence_id="fixture-h1-001",
            )
            self.assertIsNone(config["cluster_to_internal"])
            self.assertEqual(
                config["cluster_mapping_scope"],
                "tracklab_team_per_sequence",
            )

    def test_reviewed_registry_requires_clip_scoped_cluster_alignment(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = Path(raw) / "registry.json"
            payload = {
                "version": 1,
                "matches": {
                    "fixture": {
                        "mapping_status": "reviewed",
                        "cluster_to_internal": {
                            "0": "right",
                            "1": "left",
                        },
                        "cluster_to_club": {
                            "0": "Arsenal",
                            "1": "Chelsea",
                        },
                    }
                },
            }
            registry.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                ValueError, "sequence_cluster_to_internal"
            ):
                load_match_team_config(
                    registry, "fixture", sequence_id="fixture-h1-001"
                )
            payload["matches"]["fixture"][
                "sequence_cluster_to_internal"
            ] = {
                "fixture-h1-001": {"0": "left", "1": "right"}
            }
            registry.write_text(json.dumps(payload))
            config = load_match_team_config(
                registry, "fixture", sequence_id="fixture-h1-001"
            )
            self.assertEqual(
                config["team_names"],
                {"left": "Chelsea", "right": "Arsenal"},
            )
            self.assertEqual(
                config["club_to_internal"],
                {"arsenal": "right", "chelsea": "left"},
            )
            self.assertEqual(
                config["cluster_to_internal"],
                {0: "left", 1: "right"},
            )

    def test_explicit_and_neutral_configs_are_distinct(self):
        neutral = neutral_match_team_config()
        reviewed = explicit_match_team_config("Arsenal", "Chelsea")
        self.assertEqual(neutral["mapping_status"], "unreviewed")
        self.assertEqual(reviewed["mapping_status"], "reviewed")
        self.assertEqual(
            reviewed["club_to_internal"],
            {"arsenal": "left", "chelsea": "right"},
        )

    def test_strict_model_rejects_clubs_from_another_match(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(
                ValueError, "clubs outside this match.*burnley"
            ):
                resolve_team_assignments(
                    {7: "left", 9: "right"},
                    self.rows(),
                    sequence_id="new-match-h1-001",
                    club_to_internal={"arsenal": "left", "chelsea": "right"},
                    labels_path=None,
                    model_path=self.artifact(directory),
                    strict_model_classes=True,
                )

    def test_nonparticipant_filter_uses_ignore_role_and_pitch_without_club_mapping(self):
        with tempfile.TemporaryDirectory() as raw:
            model_path = Path(raw) / "validity.npz"
            np.savez_compressed(
                model_path,
                coef=np.asarray(
                    [[-2.0, 0.0], [0.0, 1.0], [4.0, 0.0]],
                    dtype=np.float32,
                ),
                intercept=np.zeros(3, dtype=np.float32),
                mean=np.zeros(2, dtype=np.float32),
                scale=np.ones(2, dtype=np.float32),
                classes=np.asarray(["arsenal", "burnley", "ignore"]),
                threshold=np.asarray([0.8], dtype=np.float32),
            )
            rows = []
            for track_id, role, embedding, pitch, count in (
                (1, "player", [3.0, 0.0], [0.0, 0.0], 3),
                (2, "player", [0.0, 3.0], [0.0, 35.0], 3),
                (3, "goalkeeper", [0.0, 3.0], [0.0, 0.0], 3),
                (4, "player", [0.0, 3.0], [0.0, 0.0], 2),
                (5, "player", [-3.0, 0.0], [0.0, 0.0], 3),
                (6, "player", [-3.0, 0.0], [0.0, 0.0], 3),
            ):
                rows.extend(
                    {
                        "track_id": track_id,
                        "role": role,
                        "embeddings": np.asarray([embedding], dtype=np.float32),
                        "bbox_pitch": {
                            "x_bottom_middle": pitch[0],
                            "y_bottom_middle": pitch[1],
                        },
                        "role_detection": (
                            "referee" if track_id == 6 else "player"
                        ),
                    }
                    for _ in range(count)
                )
            excluded, evidence = infer_nonparticipant_track_ids(
                pd.DataFrame(rows),
                model_path=model_path,
            )
            self.assertEqual(excluded, {1, 2, 3, 4, 6})
            self.assertEqual(
                evidence["reasons"]["expert_ignore_model"], [1]
            )
            self.assertEqual(
                evidence["reasons"]["role_detection_nonplayer"], [6]
            )
            self.assertNotIn(5, excluded)


if __name__ == "__main__":
    unittest.main()
