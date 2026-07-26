from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.build_expanded_review_pool import (
    ArtifactError,
    ClipArtifacts,
    LABEL_TITLES,
    PredictionSummary,
    _neutral_team_slot,
    _select_target_proposals,
    completed_downstream_rows,
    neutral_corrected_track_assignments,
    review_item,
    resolve_artifacts,
    select_balanced_windows,
    summarize_predictions,
    temporal_window_proposals,
)
from scripts.expanded_review_server import (
    HTML,
    ReviewServer,
    parse_range_header,
    validate_annotation,
)


LABELS = (
    "high_press",
    "trap_left",
    "trap_right",
    "central_screen",
    "unstructured",
)


def prediction(
    frame: int,
    label: str,
    probability: float,
    *,
    reliable: bool = True,
    direction_confidence: float = 1.0,
) -> dict:
    remainder = (1 - probability) / (len(LABELS) - 1)
    probabilities = {candidate: remainder for candidate in LABELS}
    probabilities[label] = probability
    return {
        "frame": frame,
        "predicted_situation": label,
        "possession_confident": reliable,
        "attacking_direction_raw": 1,
        "attacking_direction_label": "left_to_right",
        "direction_confident": True,
        "direction_confidence": direction_confidence,
        "direction_source": "match_half_metadata",
        "direction_status": "calibrated",
        "direction_evidence": {
            "status": "calibrated",
            "confident": True,
            "confidence": direction_confidence,
        },
        "confidence": probability,
        "probabilities": probabilities,
    }


def artifact(clip_id: str, match: str) -> ClipArtifacts:
    root = Path("/tmp/not-read-by-selection")
    return ClipArtifacts(
        item={
            "id": clip_id,
            "game": match,
            "match_id": match,
            "half": 1,
            "nframes": 200,
            "clip_path": f"{clip_id}.mp4",
        },
        clip_path=root / f"{clip_id}.mp4",
        graph_path=root / f"{clip_id}.npz",
        predictions_path=root / f"{clip_id}-predictions.jsonl",
        state_path=root / f"{clip_id}.pklz",
    )


def proposal_summary(
    label: str,
    confidence: float,
    agreement: float,
    *,
    frame: int = 50,
) -> PredictionSummary:
    return PredictionSummary(
        model_label=label,
        classification_confidence=confidence,
        temporal_agreement=agreement,
        majority_frames=max(1, round(agreement * 10)),
        valid_graph_frames=10,
        representative_index=0,
        representative_frame=frame,
        window_start_frame=0,
        window_end_frame=100,
        direction_usable=True,
        direction_confidence=1.0,
        direction_sources=("test",),
        direction_statuses=("confident",),
        direction_evidence={},
    )


class ExpandedReviewPoolTest(unittest.TestCase):
    def test_trap_titles_are_explicitly_attacker_relative(self):
        self.assertIn("attacker-relative", LABEL_TITLES["trap_left"])
        self.assertIn("attacker-relative", LABEL_TITLES["trap_right"])

    def test_overlay_uses_converter_corrected_team_not_raw_cluster(self):
        rows = [prediction(25, "high_press", 0.8)]
        rows[0].update(
            team_identity_status="unreviewed",
            team_identity_map={"left": "Team A", "right": "Team B"},
            team_cluster_evidence={
                "neutral_corrected_track_assignments": {
                    "7": "right",
                    "9": "left",
                },
                "neutral_identity_inputs": {
                    "manual_labels_enabled": False,
                    "identity_model_enabled": False,
                },
            },
        )
        assignments = neutral_corrected_track_assignments(
            rows, context="clip"
        )
        # Track 7's raw cluster says 0, but the converter corrected it to the
        # anonymous right slot. The overlay must use the correction.
        row = SimpleNamespace(track_id=7, team_cluster=0, team="left")
        self.assertEqual(_neutral_team_slot(row, assignments), 1)
        self.assertEqual(assignments, {7: "right", 9: "left"})

    def test_overlay_refuses_raw_cluster_fallback_or_identity_inputs(self):
        rows = [prediction(25, "high_press", 0.8)]
        rows[0].update(
            team_identity_status="unreviewed",
            team_identity_map={"left": "Team A", "right": "Team B"},
            team_cluster_evidence={
                "neutral_corrected_track_assignments": {"7": "left"},
                "neutral_identity_inputs": {
                    "manual_labels_enabled": False,
                    "identity_model_enabled": True,
                },
            },
        )
        with self.assertRaisesRegex(
            ArtifactError, "identity model disabled"
        ):
            neutral_corrected_track_assignments(rows, context="clip")
        self.assertIsNone(
            _neutral_team_slot(
                SimpleNamespace(track_id=8, team_cluster=0, team="left"),
                {7: "left"},
            )
        )

    def test_classification_probability_is_not_temporal_agreement(self):
        rows = [
            prediction(0, "high_press", 0.8),
            prediction(25, "high_press", 0.7),
            prediction(50, "trap_left", 0.6),
            prediction(75, "trap_right", 0.9, reliable=False),
        ]
        # Give the majority class probabilities 0.8, 0.7, and 0.1 across the
        # three reliable frames.
        rows[2]["probabilities"]["high_press"] = 0.1
        summary = summarize_predictions(rows)
        self.assertEqual(summary.model_label, "high_press")
        self.assertAlmostEqual(summary.temporal_agreement, 2 / 3)
        self.assertAlmostEqual(summary.classification_confidence, (0.8 + 0.7 + 0.1) / 3)

    def test_direction_confidence_is_conservative_and_serialized(self):
        registry_rows = [
            prediction(
                0,
                "high_press",
                0.9,
                direction_confidence=0.9,
            ),
            prediction(
                25,
                "high_press",
                0.9,
                direction_confidence=0.82,
            ),
        ]
        # Exercise backward compatibility with graphs where the value is only
        # nested inside direction_evidence.
        for row in registry_rows:
            row.pop("direction_confidence")
        registry_summary = summarize_predictions(registry_rows)
        self.assertAlmostEqual(registry_summary.direction_confidence, 0.82)
        registry_item = review_item(
            artifact("registry-direction", "Match A"),
            registry_summary,
            Path("videos/registry-direction.mp4"),
            fps=25,
        )
        self.assertEqual(registry_item["direction_confidence"], 0.82)
        self.assertEqual(
            f"{registry_item['direction_confidence']:.0%}",
            "82%",
        )

        manual_summary = summarize_predictions(
            [
                prediction(0, "high_press", 0.9),
                prediction(25, "high_press", 0.9),
            ]
        )
        manual_item = review_item(
            artifact("manual-direction", "Match B"),
            manual_summary,
            Path("videos/manual-direction.mp4"),
            fps=25,
        )
        self.assertEqual(manual_item["direction_confidence"], 1.0)
        self.assertEqual(
            f"{manual_item['direction_confidence']:.0%}",
            "100%",
        )

    def test_invalid_direction_confidence_is_withheld(self):
        for untrusted in (None, "0.82", True, -0.1, 1.1):
            with self.subTest(direction_confidence=untrusted):
                rows = [
                    prediction(0, "high_press", 0.9),
                    prediction(25, "high_press", 0.9),
                ]
                for row in rows:
                    if untrusted is None:
                        row.pop("direction_confidence")
                        row["direction_evidence"].pop("confidence")
                    else:
                        row["direction_confidence"] = untrusted
                        row["direction_evidence"]["confidence"] = untrusted
                self.assertEqual(
                    temporal_window_proposals(
                        rows,
                        nframes=100,
                        span_frames=100,
                        stride_frames=25,
                        min_graph_frames=1,
                    ),
                    {},
                )

    def test_searches_four_second_windows_instead_of_whole_source(self):
        rows = [
            *[prediction(frame, "high_press", 0.8) for frame in (0, 25, 50, 75)],
            *[prediction(frame, "trap_left", 0.75) for frame in (100, 125, 150, 175)],
        ]
        proposals = temporal_window_proposals(
            rows,
            nframes=200,
            span_frames=100,
            stride_frames=25,
            min_graph_frames=2,
        )
        self.assertEqual(
            (proposals["high_press"].window_start_frame, proposals["high_press"].window_end_frame),
            (0, 100),
        )
        self.assertEqual(
            (proposals["trap_left"].window_start_frame, proposals["trap_left"].window_end_frame),
            (100, 200),
        )

    def test_unreliable_possession_is_withheld(self):
        unreliable = [
            prediction(0, "high_press", 0.9, reliable=False),
            prediction(25, "high_press", 0.9, reliable=False),
        ]
        self.assertEqual(
            temporal_window_proposals(
                unreliable,
                nframes=100,
                span_frames=100,
                stride_frames=25,
                min_graph_frames=1,
            ),
            {},
        )

    def test_all_direction_dependent_classes_are_withheld_on_abstention(self):
        for label in ("high_press", "trap_left", "trap_right", "central_screen"):
            with self.subTest(label=label):
                ambiguous = [
                    prediction(0, label, 0.9),
                    prediction(25, label, 0.9),
                ]
                for row in ambiguous:
                    row["direction_status"] = "abstained_ambiguous"
                proposals = temporal_window_proposals(
                    ambiguous,
                    nframes=100,
                    span_frames=100,
                    stride_frames=25,
                    min_graph_frames=1,
                )
                self.assertNotIn(label, proposals)

    def test_direction_invariant_fallback_survives_direction_abstention(self):
        ambiguous = [
            prediction(0, "unstructured", 0.9),
            prediction(25, "unstructured", 0.9),
        ]
        for row in ambiguous:
            row["direction_status"] = "abstained_ambiguous"
        proposals = temporal_window_proposals(
            ambiguous,
            nframes=100,
            span_frames=100,
            stride_frames=25,
            min_graph_frames=1,
        )
        self.assertIn("unstructured", proposals)
        self.assertFalse(proposals["unstructured"].direction_usable)

    def test_direction_metadata_must_be_internally_consistent(self):
        rows = [
            prediction(0, "high_press", 0.9),
            prediction(25, "high_press", 0.9),
        ]
        for row in rows:
            row["attacking_direction_raw"] = -1
            row["attacking_direction_label"] = "left_to_right"
        self.assertEqual(
            temporal_window_proposals(
                rows,
                nframes=100,
                span_frames=100,
                stride_frames=25,
                min_graph_frames=1,
            ),
            {},
        )

    def test_direction_confidence_must_be_explicit_boolean_true(self):
        for untrusted in (None, "true", 1, False):
            with self.subTest(direction_confident=untrusted):
                rows = [
                    prediction(0, "high_press", 0.9),
                    prediction(25, "high_press", 0.9),
                ]
                for row in rows:
                    if untrusted is None:
                        row.pop("direction_confident")
                    else:
                        row["direction_confident"] = untrusted
                self.assertEqual(
                    temporal_window_proposals(
                        rows,
                        nframes=100,
                        span_frames=100,
                        stride_frames=25,
                        min_graph_frames=1,
                    ),
                    {},
                )

    def test_actual_class_balancing_prefers_match_diversity(self):
        clips = [
            artifact("a-high-1", "Match A"),
            artifact("a-high-2", "Match A"),
            artifact("b-high-1", "Match B"),
        ]
        prepared = {
            clips[0].id: [prediction(25, "high_press", 0.95), prediction(50, "high_press", 0.95)],
            clips[1].id: [prediction(25, "high_press", 0.94), prediction(50, "high_press", 0.94)],
            clips[2].id: [prediction(25, "high_press", 0.80), prediction(50, "high_press", 0.80)],
        }
        selected, counts, by_match = select_balanced_windows(
            clips,
            prepared,
            fps=25,
            duration=4,
            stride_seconds=1,
            min_graph_frames=2,
            per_target=2,
        )
        high_matches = {
            row.item["match_id"]
            for row, summary in selected
            if summary.model_label == "high_press"
        }
        self.assertEqual(high_matches, {"Match A", "Match B"})
        self.assertEqual(counts["high_press"], 2)
        self.assertEqual(by_match["Match A"]["high_press"], 1)
        self.assertEqual(by_match["Match B"]["high_press"], 1)

    def test_global_assignment_keeps_strong_trap_over_weak_high_press(self):
        clip = artifact("shared", "Match A")
        proposals = {
            clip.id: {
                "high_press": proposal_summary("high_press", 0.391, 0.4),
                "trap_right": proposal_summary("trap_right", 0.92, 0.923),
            }
        }
        selected = _select_target_proposals(
            [clip],
            proposals,
            per_target=1,
        )
        self.assertEqual(
            [(row.id, summary.model_label) for row, summary in selected],
            [("shared", "trap_right")],
        )

    def test_global_assignment_does_not_starve_target_classes(self):
        clips = [
            artifact("shared-left", "Match A"),
            artifact("shared-right", "Match A"),
            artifact("high-only", "Match A"),
        ]
        proposals = {
            "shared-left": {
                "high_press": proposal_summary("high_press", 0.99, 1.0),
                "trap_left": proposal_summary("trap_left", 0.85, 0.9),
            },
            "shared-right": {
                "high_press": proposal_summary("high_press", 0.98, 1.0),
                "trap_right": proposal_summary("trap_right", 0.84, 0.9),
            },
            "high-only": {
                "high_press": proposal_summary("high_press", 0.4, 0.4),
            },
        }
        selected = _select_target_proposals(
            clips,
            proposals,
            per_target=1,
        )
        assignments = {
            row.id: summary.model_label for row, summary in selected
        }
        self.assertEqual(
            assignments,
            {
                "high-only": "high_press",
                "shared-left": "trap_left",
                "shared-right": "trap_right",
            },
        )

    def test_global_assignment_is_deterministic(self):
        clips = [
            artifact("clip-d", "Match B"),
            artifact("clip-c", "Match B"),
            artifact("clip-b", "Match A"),
            artifact("clip-a", "Match A"),
        ]
        proposals = {
            clip.id: {
                "high_press": proposal_summary("high_press", 0.8, 0.8),
                "trap_left": proposal_summary("trap_left", 0.8, 0.8),
                "trap_right": proposal_summary("trap_right", 0.8, 0.8),
            }
            for clip in clips
        }
        forward = _select_target_proposals(
            clips,
            proposals,
            per_target=1,
        )
        reverse = _select_target_proposals(
            list(reversed(clips)),
            dict(reversed(list(proposals.items()))),
            per_target=1,
        )
        self.assertEqual(
            [(row.id, summary.model_label) for row, summary in forward],
            [(row.id, summary.model_label) for row, summary in reverse],
        )
        self.assertEqual(len({row.id for row, _ in forward}), len(forward))
        self.assertEqual(
            {summary.model_label for _, summary in forward},
            {"high_press", "trap_left", "trap_right"},
        )

    def test_lower_classes_only_fill_missing_target_capacity(self):
        clips = [
            artifact("high", "Match A"),
            artifact("central", "Match B"),
        ]
        prepared = {
            "high": [prediction(25, "high_press", 0.8), prediction(50, "high_press", 0.8)],
            "central": [
                prediction(25, "central_screen", 0.8),
                prediction(50, "central_screen", 0.8),
            ],
        }
        selected, counts, _ = select_balanced_windows(
            clips,
            prepared,
            fps=25,
            duration=4,
            stride_seconds=1,
            min_graph_frames=2,
            per_target=1,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(counts["high_press"], 1)
        self.assertEqual(counts["central_screen"], 1)

    def test_missing_inputs_stop_before_any_rendering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "clips.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "id": "missing-clip",
                                "clip_path": "missing.mp4",
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(
                ArtifactError, "no videos were rendered"
            ):
                resolve_artifacts(
                    manifest,
                    root / "graphs",
                    root / "states",
                    root,
                )

    def test_terminal_status_selects_completed_and_reports_exclusions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "clips.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": [
                            {"id": "good"},
                            {"id": "no-ball"},
                            {"id": "excluded"},
                        ]
                    }
                )
            )
            status = root / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "manifest": str(manifest.resolve()),
                        "clips": {
                            "good": {
                                "status": "completed",
                                "graph_path": "/graphs/good.npz",
                            },
                            "no-ball": {
                                "status": "failed",
                                "failed_stage": "convert",
                                "error": "No classifiable frames with a detected ball",
                            },
                            "excluded": {
                                "status": "excluded",
                                "reason": "shot rejected",
                            },
                        },
                    }
                )
            )
            completed, report = completed_downstream_rows(manifest, status)
            self.assertEqual(set(completed), {"good"})
            self.assertEqual(report["completed_count"], 1)
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual(report["excluded_count"], 1)
            self.assertIn(
                "No classifiable frames",
                report["terminal_exclusions"][0]["reason"],
            )

            payload = json.loads(status.read_text())
            payload["clips"]["excluded"] = {"status": "waiting_for_state"}
            status.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                ArtifactError, "not terminal; no videos were rendered"
            ):
                completed_downstream_rows(manifest, status)


class ExpandedReviewServerTest(unittest.TestCase):
    def test_ui_separates_confidence_and_retrieval_and_guards_overflow(self):
        self.assertIn("Classification confidence", HTML)
        self.assertIn("Candidate retrieval metadata", HTML)
        self.assertIn("It is not classification confidence", HTML)
        self.assertIn("overflow-wrap:anywhere", HTML)
        self.assertIn('id="previous"', HTML)
        self.assertIn('id="next"', HTML)
        self.assertIn('"First ↻":"Next →"', HTML)
        self.assertIn("state.items.length-1?0:index+1", HTML)
        self.assertIn("Direction verified", HTML)
        self.assertIn("Direction not required", HTML)
        self.assertIn("Required by this canonical tactical class", HTML)
        self.assertIn('id="direction-confidence"', HTML)
        self.assertIn("pct(item.direction_confidence)", HTML)
        self.assertIn("Direction confidence:", HTML)
        self.assertNotIn("Model evidence", HTML)

    def test_annotation_validation(self):
        row = validate_annotation(
            {
                "id": "clip-1",
                "decision": "accept",
                "label_override": "trap_right",
                "notes": "Direction checked.",
            },
            valid_ids={"clip-1"},
            labels=set(LABELS),
        )
        self.assertEqual(row["label_override"], "trap_right")
        with self.assertRaisesRegex(ValueError, "Invalid tactical label"):
            validate_annotation(
                {
                    "id": "clip-1",
                    "decision": "accept",
                    "label_override": "right_touchline",
                },
                valid_ids={"clip-1"},
                labels=set(LABELS),
            )

    def test_server_persists_effective_override_locally(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "videos").mkdir()
            (root / "videos" / "clip-1.mp4").write_bytes(b"video")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "labels": list(LABELS),
                        "label_titles": {label: label for label in LABELS},
                        "items": [
                            {
                                "id": "clip-1",
                                "video": "videos/clip-1.mp4",
                                "model_label": "high_press",
                            }
                        ],
                    }
                )
            )
            # Exercise persistence without binding a socket in the test
            # sandbox.
            server = ReviewServer.__new__(ReviewServer)
            server.annotation_path = root / "annotations.json"
            server.valid_ids = {"clip-1"}
            server.labels = set(LABELS)
            server.item_by_id = {
                "clip-1": {"model_label": "high_press"}
            }
            server.write_lock = threading.Lock()
            saved = server.save(
                {
                    "id": "clip-1",
                    "decision": "accept",
                    "label_override": "trap_left",
                    "notes": "",
                }
            )
            self.assertEqual(saved["model_label"], "high_press")
            self.assertEqual(saved["effective_label"], "trap_left")
            persisted = json.loads((root / "annotations.json").read_text())
            self.assertEqual(
                persisted["annotations"]["clip-1"]["effective_label"],
                "trap_left",
            )

    def test_video_byte_ranges(self):
        self.assertIsNone(parse_range_header(None, 100))
        self.assertEqual(parse_range_header("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_range_header("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_range_header("bytes=-10", 100), (90, 99))
        with self.assertRaises(ValueError):
            parse_range_header("bytes=100-101", 100)


if __name__ == "__main__":
    unittest.main()
