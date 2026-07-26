from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.process_gsr_outputs import (
    DirectionSelection,
    artifacts_current,
    conversion_signature,
    converter_command,
    direction_override,
    graph_count,
    graph_count_with_provenance,
    load_manifest_rows,
    prediction_count,
    resolve_direction_selection,
    stage_plan,
    status_payload,
    validate_final_direction_registry,
    validate_neutral_registry,
    weak_count,
)


class ProcessGSROutputsTest(unittest.TestCase):
    def direction_registry(
        self,
        *,
        waiting: int = 0,
        confident: bool = True,
    ) -> tuple[dict, dict[str, dict]]:
        rows = {
            "clip": {
                "id": "clip",
                "match_id": "match",
                "half": 1,
            }
        }
        half = {
            "status": (
                "calibrated"
                if confident
                else "abstained_insufficient_clips"
            ),
            "direction_confident": confident,
            "confidence": 0.8 if confident else 0.0,
            "attacking_direction": (
                {
                    "left": {"raw": 1, "confidence": 0.8},
                    "right": {"raw": -1, "confidence": 0.8},
                }
                if confident
                else None
            ),
            "evidence": {
                "status": (
                    "calibrated"
                    if confident
                    else "abstained_insufficient_clips"
                ),
                "clips_available": 1,
                "clips_usable": int(confident),
            },
        }
        recorded = 1 - waiting
        registry = {
            "schema_version": 1,
            "matches": {
                "match": {"halves": {"1": half}},
            },
            "summary": {
                "clips_selected": 1,
                "clips_recorded": recorded,
                "clips_waiting_for_state": waiting,
                "clips_accounted_for": 1,
                "match_halves": 1,
            },
        }
        return registry, rows

    def test_converter_is_neutral_and_explicitly_disables_legacy_identity(self):
        command = converter_command(
            tracklab_python=Path("/runtime/python"),
            state=Path("/states/clip.pklz"),
            video=Path("/clips/clip.mp4"),
            yolo=Path("/models/yolo.pt"),
            graph=Path("/graphs/clip.npz"),
            sequence_id="clip",
            team_registry=Path("/registry/teams.json"),
            match_id="league/season/match",
            directions=(-1, 1),
            ball_confidence=0.03,
        )
        self.assertIn("--match-registry", command)
        self.assertIn("--match-id", command)
        self.assertIn("--disable-team-labels", command)
        self.assertIn("--disable-team-model", command)
        self.assertIn("--left-team-direction", command)
        self.assertNotIn("team_identity_burnley_arsenal.npz", " ".join(command))
        self.assertNotIn("Burnley", " ".join(command))

    def test_manifest_rows_are_normalized_and_reject_fractional_half(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = Path(raw) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "id": " clip ",
                                "match_id": " match ",
                                "half": "2",
                            }
                        ]
                    }
                )
            )
            rows = load_manifest_rows(manifest)
            self.assertEqual(
                rows["clip"],
                {"id": "clip", "match_id": "match", "half": 2},
            )
            manifest.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "id": "clip",
                                "match_id": "match",
                                "half": 1.5,
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "invalid half"):
                load_manifest_rows(manifest)

    def test_direction_manifest_wins_then_registry_then_inference(self):
        row = {
            "id": "clip",
            "match_id": "match",
            "half": 1,
            "left_team_direction": -1,
            "right_team_direction": 1,
        }
        registry = {
            "matches": {
                "match": {
                    "halves": {
                        "1": {
                            "attacking_direction": {
                                "left": {"raw": 1},
                                "right": {"raw": -1},
                            }
                        }
                    }
                }
            }
        }
        self.assertEqual(
            direction_override(row, registry),
            ((-1, 1), "manifest_match_half_override"),
        )
        row.pop("left_team_direction")
        row.pop("right_team_direction")
        self.assertEqual(
            direction_override(row, registry),
            ((1, -1), "direction_registry_match_half_override"),
        )
        self.assertEqual(
            direction_override(row, None),
            (None, "converter_goalkeeper_or_team_median"),
        )

    def test_registry_direction_provenance_reaches_converter_and_status(self):
        row = {"id": "clip", "match_id": "match", "half": 2}
        registry = {
            "schema_version": 1,
            "generated_at": "2026-07-25T06:02:39+00:00",
            "coordinate_system": "raw pitch x",
            "method": "aggregate clip votes",
            "matches": {
                "match": {
                    "halves": {
                        "2": {
                            "status": "calibrated",
                            "direction_confident": True,
                            "confidence": 0.8125,
                            "attacking_direction": {
                                "left": {"raw": 1, "confidence": 0.8125},
                                "right": {"raw": -1, "confidence": 0.8125},
                            },
                            "evidence": {
                                "status": "calibrated",
                                "confident": True,
                                "confidence": 0.8125,
                                "clips_available": 7,
                                "clips_usable": 5,
                                "orientation_clip_counts": {
                                    "-1": 1,
                                    "1": 4,
                                },
                                "orientation_weights": {
                                    "-1": 0.9,
                                    "1": 3.9,
                                },
                                "clips": [
                                    {
                                        "sequence_id": "large-record-not-copied",
                                        "alignment": {"details": "..."},
                                    }
                                ],
                            },
                        }
                    }
                }
            },
        }
        selection = resolve_direction_selection(row, registry)
        self.assertEqual(selection.directions, (1, -1))
        self.assertEqual(
            selection.source, "direction_registry_match_half_override"
        )
        self.assertEqual(selection.status, "calibrated")
        self.assertTrue(selection.confident)
        self.assertEqual(selection.confidence, 0.8125)
        self.assertEqual(selection.evidence["aggregate"]["clips_usable"], 5)
        self.assertNotIn("clips", selection.evidence["aggregate"])

        provenance = selection.converter_provenance()
        command = converter_command(
            tracklab_python=Path("/runtime/python"),
            state=Path("/states/clip.pklz"),
            video=Path("/clips/clip.mp4"),
            yolo=Path("/models/yolo.pt"),
            graph=Path("/graphs/clip.npz"),
            sequence_id="clip",
            team_registry=Path("/registry/teams.json"),
            match_id="match",
            directions=selection.directions,
            ball_confidence=0.03,
            direction_provenance=provenance,
        )
        provenance_index = command.index("--direction-provenance-json") + 1
        passed = json.loads(command[provenance_index])
        self.assertEqual(passed["confidence"], 0.8125)
        self.assertEqual(passed["status"], "calibrated")
        self.assertEqual(
            passed["evidence"]["aggregate"]["orientation_clip_counts"],
            {"-1": 1, "1": 4},
        )

        with tempfile.TemporaryDirectory() as raw:
            direction_registry = Path(raw) / "directions.json"
            status = status_payload(
                None,
                manifest=Path("/manifest.json"),
                state_root=Path("/states"),
                output_dir=Path("/graphs"),
                team_registry=Path("/teams.json"),
                direction_registry=direction_registry,
            )
        self.assertEqual(
            status["direction_registry"], str(direction_registry)
        )

    def test_registry_rejects_directions_marked_non_confident(self):
        row = {"id": "clip", "match_id": "match", "half": 1}
        registry = {
            "matches": {
                "match": {
                    "halves": {
                        "1": {
                            "status": "abstained_conflicting_clips",
                            "direction_confident": False,
                            "confidence": 0.6,
                            "attacking_direction": {
                                "left": 1,
                                "right": -1,
                            },
                        }
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            resolve_direction_selection(row, registry)

    def test_final_registry_requires_terminal_manifest_coverage(self):
        registry, rows = self.direction_registry()
        validate_final_direction_registry(registry, rows)

        registry, rows = self.direction_registry(waiting=1)
        with self.assertRaisesRegex(ValueError, "not final"):
            validate_final_direction_registry(registry, rows)

        registry, rows = self.direction_registry()
        registry["matches"]["match"]["halves"].clear()
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_final_direction_registry(registry, rows)

    def test_registry_abstention_is_preserved_for_converter_inference(self):
        registry, rows = self.direction_registry(confident=False)
        validate_final_direction_registry(registry, rows)
        selection = resolve_direction_selection(rows["clip"], registry)
        self.assertIsNone(selection.directions)
        self.assertEqual(
            selection.source,
            "direction_registry_abstained_converter_inference",
        )
        self.assertEqual(
            selection.status,
            "abstained_insufficient_clips",
        )
        self.assertFalse(selection.confident)
        self.assertEqual(selection.confidence, 0.0)
        self.assertEqual(selection.evidence["match_id"], "match")
        del registry["matches"]["match"]["halves"]["1"]["confidence"]
        self.assertEqual(
            resolve_direction_selection(
                rows["clip"], registry
            ).confidence,
            0.0,
        )

    def test_conversion_signature_binds_command_and_direction_selection(self):
        selection = DirectionSelection(
            directions=(1, -1),
            source="direction_registry_match_half_override",
            status="calibrated",
            confident=True,
            confidence=0.8,
            evidence={"aggregate": {"clips_usable": 3}},
        )
        command = ["python", "convert.py", "--ball-confidence", "0.03"]
        baseline = conversion_signature(command, selection)
        self.assertEqual(baseline, conversion_signature(command, selection))
        self.assertNotEqual(
            baseline,
            conversion_signature(
                ["python", "convert.py", "--ball-confidence", "0.05"],
                selection,
            ),
        )
        self.assertNotEqual(
            baseline,
            conversion_signature(
                command,
                DirectionSelection(
                    directions=(-1, 1),
                    source=selection.source,
                    status=selection.status,
                    confident=True,
                    confidence=selection.confidence,
                    evidence=selection.evidence,
                ),
            ),
        )

    def test_same_direction_is_rejected(self):
        row = {
            "id": "clip",
            "match_id": "match",
            "half": 1,
            "attacking_direction": {"left": 1, "right": 1},
        }
        with self.assertRaises(ValueError):
            direction_override(row, None)

    def test_unreviewed_registry_produces_only_neutral_names(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = Path(raw) / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "matches": {
                            "match": {
                                "mapping_status": "unreviewed",
                                "cluster_to_internal": {
                                    "0": "left",
                                    "1": "right",
                                },
                            }
                        },
                    }
                )
            )
            validate_neutral_registry(registry, ["match"])

            payload = json.loads(registry.read_text())
            payload["matches"]["match"].update(
                mapping_status="reviewed",
                cluster_to_club={"0": "Arsenal", "1": "Chelsea"},
            )
            registry.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                validate_neutral_registry(registry, ["match"])

    def test_artifact_validators_and_freshness(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            graph = root / "clip.npz"
            features = np.zeros((2, 23, 13), dtype=np.float32)
            masks = np.zeros((2, 23), dtype=bool)
            np.savez_compressed(graph, features=features, masks=masks)
            graph.with_suffix(".jsonl").write_text(
                json.dumps({"frame": 1}) + "\n"
                + json.dumps({"frame": 2})
                + "\n"
            )
            weak = root / "clip-weak.npz"
            np.savez_compressed(
                weak,
                labels=np.array([0, -1]),
                confidence=np.array([0.8, 0.0], dtype=np.float32),
                label_names=np.array(["unstructured"]),
            )
            weak.with_suffix(".jsonl").write_text(
                json.dumps({"weak_label": "unstructured"}) + "\n"
                + json.dumps({"weak_label": "abstain"})
                + "\n"
            )
            prediction = root / "clip-predictions.jsonl"
            prediction.write_text(
                "".join(
                    json.dumps(
                        {
                            "predicted_situation": "high_press",
                            "confidence": 0.8,
                            "probabilities": {"high_press": 0.8},
                        }
                    )
                    + "\n"
                    for _ in range(2)
                )
            )
            dependency = root / "input"
            dependency.touch()
            older = min(
                graph.stat().st_mtime_ns,
                graph.with_suffix(".jsonl").stat().st_mtime_ns,
            ) - 1_000_000
            os.utime(dependency, ns=(older, older))

            self.assertEqual(graph_count(graph), 2)
            self.assertEqual(weak_count(weak, 2), 2)
            self.assertEqual(prediction_count(prediction, 2), 2)
            current, reason = artifacts_current(
                [graph, graph.with_suffix(".jsonl")],
                [dependency],
                lambda: graph_count(graph),
            )
            self.assertTrue(current, reason)

            newer = max(
                graph.stat().st_mtime_ns,
                graph.with_suffix(".jsonl").stat().st_mtime_ns,
            ) + 1_000_000
            os.utime(dependency, ns=(newer, newer))
            current, reason = artifacts_current(
                [graph, graph.with_suffix(".jsonl")],
                [dependency],
                lambda: graph_count(graph),
            )
            self.assertFalse(current)
            self.assertIn("older", reason)

    def test_graph_provenance_and_signature_control_resumability(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            graph = root / "clip.npz"
            features = np.zeros((1, 23, 13), dtype=np.float32)
            masks = np.zeros((1, 23), dtype=bool)
            np.savez_compressed(graph, features=features, masks=masks)
            metadata = {
                "sequence": "clip",
                "team_identity_match_id": "match",
                "team_identity_status": "unreviewed",
                "possession_team": "left",
                "attacking_direction_raw": 1,
                "attacking_direction_label": "left_to_right",
                "direction_source": (
                    "direction_registry_match_half_override"
                ),
                "direction_status": "calibrated",
                "direction_confident": True,
                "direction_confidence": 0.8,
            }
            graph.with_suffix(".jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            weak = root / "clip-weak.npz"
            np.savez_compressed(
                weak,
                labels=np.array([4]),
                confidence=np.array([0.8], dtype=np.float32),
                label_names=np.array(["high_press"]),
            )
            weak.with_suffix(".jsonl").write_text(
                json.dumps({"weak_label": "high_press"}) + "\n"
            )
            prediction = root / "clip-predictions.jsonl"
            prediction.write_text(
                json.dumps(
                    {
                        "predicted_situation": "high_press",
                        "confidence": 0.8,
                        "probabilities": {"high_press": 0.8},
                    }
                )
                + "\n"
            )
            dependency = root / "dependency"
            dependency.touch()
            older = graph.stat().st_mtime_ns - 1_000_000
            os.utime(dependency, ns=(older, older))
            selection = DirectionSelection(
                directions=(1, -1),
                source="direction_registry_match_half_override",
                status="calibrated",
                confident=True,
                confidence=0.8,
            )
            validator = lambda: graph_count_with_provenance(
                graph,
                sequence_id="clip",
                match_id="match",
                selection=selection,
            )
            self.assertEqual(validator(), 1)
            plan, count = stage_plan(
                graph=graph,
                weak=weak,
                predictions=prediction,
                graph_dependencies=[dependency],
                weak_dependencies=[graph, graph.with_suffix(".jsonl")],
                prediction_dependencies=[
                    graph,
                    graph.with_suffix(".jsonl"),
                ],
                graph_validator=validator,
                conversion_signature_matches=False,
            )
            self.assertIsNone(count)
            self.assertTrue(plan["convert"].startswith("run:"))
            self.assertTrue(plan["weak_labels"].startswith("run:"))
            self.assertTrue(plan["classify"].startswith("run:"))

            metadata["attacking_direction_raw"] = -1
            metadata["attacking_direction_label"] = "right_to_left"
            graph.with_suffix(".jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            with self.assertRaisesRegex(
                ValueError, "selected match-half direction"
            ):
                validator()


if __name__ == "__main__":
    unittest.main()
