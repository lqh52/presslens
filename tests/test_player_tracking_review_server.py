from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.player_tracking_review_server import (
    HTML,
    Handler,
    LABEL_HTML,
    SYNTHETIC_LABEL_HTML,
    TRAP_EVENT_HTML,
    REVIEWED_EVENT_HTML,
    ReviewServer,
)


class PlayerTrackingReviewPageTest(unittest.TestCase):
    def test_real_feature_tuning_controls_are_present(self):
        for control in (
            'id="dino-weight"',
            'id="reid-weight"',
            'id="color-weight"',
            'id="validation"',
        ):
            self.assertIn(control, HTML)
        self.assertIn("tuning_features", HTML)
        self.assertIn("const raw=", HTML)
        self.assertIn("assignments changed", HTML)
        self.assertIn("re-clustered anchors", HTML)
        self.assertIn("for(let iteration=0;iteration<12;iteration++)", HTML)

    def test_tuning_is_saved_per_fixture(self):
        self.assertIn("state.review.tuning?.[x.fixture_id]", HTML)
        self.assertIn("fixture:clip().fixture_id", HTML)
        self.assertIn('id="submit-tuning"', HTML)
        self.assertIn("Weights saved for this fixture", HTML)
        self.assertNotIn('bits.push("DET")', HTML)

    def test_candidate_is_selected_before_first_tuning_render(self):
        self.assertIn("select(preferred);updateTuning()", HTML)

    def test_overlay_filters_low_confidence_boxes(self):
        self.assertIn("Number(d.confidence)>=.45", HTML)
        self.assertIn("tracked people ≥45%", HTML)

    def test_label_gallery_filters_low_confidence_tracks(self):
        source = (
            __import__("inspect")
            .getsource(
                __import__(
                    "scripts.player_tracking_review_server",
                    fromlist=["ReviewServer"],
                ).ReviewServer
            )
        )
        self.assertIn('float(detection.get("confidence", 0.0)) < 0.45', source)
        self.assertIn('"yolo26m-botsort-high-recall"', source)

    def test_track_labelling_page_supports_seed_classes(self):
        self.assertIn('href="/label"', HTML)
        self.assertIn("Label tracked objects", LABEL_HTML)
        for label in (
            "team_a",
            "team_b",
            "other",
            "team_a_goalkeeper",
            "team_b_goalkeeper",
        ):
            self.assertIn(label, LABEL_HTML)
        self.assertIn("/api/track-label", LABEL_HTML)
        self.assertIn("fixed match-level seeds", LABEL_HTML)

    def test_tactical_labelling_uses_synthetic_canonical_graphs(self):
        self.assertIn('href="/tactics"', HTML)
        self.assertIn("Validate synthetic canonical graphs", SYNTHETIC_LABEL_HTML)
        self.assertIn('id="pitch"', SYNTHETIC_LABEL_HTML)
        self.assertIn('id="previous"', SYNTHETIC_LABEL_HTML)
        self.assertIn('id="yes"', SYNTHETIC_LABEL_HTML)
        self.assertIn('id="no"', SYNTHETIC_LABEL_HTML)
        self.assertIn(
            "Rule assigned: ${row.proposed_name}. Is this correct?",
            SYNTHETIC_LABEL_HTML,
        )
        self.assertIn('id="corrections"', SYNTHETIC_LABEL_HTML)
        self.assertIn("corrected_label", SYNTHETIC_LABEL_HTML)
        self.assertIn("/api/synthetic-tactic-label", SYNTHETIC_LABEL_HTML)
        definitions = ReviewServer.synthetic_definitions()
        self.assertEqual(
            set(definitions),
            {
                "unstructured",
                "central_screen",
                "trap_left",
                "trap_right",
                "high_press",
            },
        )

    def test_synthetic_yes_and_corrected_no_are_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            label_path = Path(temporary) / "labels.json"
            fake_server = SimpleNamespace(
                synthetic_label_names=["unstructured", "high_press"],
                synthetic_indices=[4],
                synthetic_targets=np.asarray([0, 0, 0, 0, 1]),
                synthetic_tactic_labels_path=label_path,
                label_lock=threading.Lock(),
                synthetic_tactic_labels=lambda: (
                    json.loads(label_path.read_text()).get("labels", {})
                    if label_path.exists()
                    else {}
                ),
            )
            for payload in (
                {"sample_id": "synthetic:4", "answer": True},
                {
                    "sample_id": "synthetic:4",
                    "answer": False,
                    "corrected_label": "unstructured",
                },
            ):
                handler = Handler.__new__(Handler)
                body = json.dumps(payload).encode()
                handler.server = fake_server
                handler.rfile = io.BytesIO(body)
                handler.headers = {"Content-Length": str(len(body))}
                responses = []
                handler.send_bytes = lambda *args, **kwargs: responses.append(
                    (args, kwargs)
                )
                handler.save_synthetic_tactic_label()
                self.assertEqual(responses[0][0][1], "application/json")
            saved = fake_server.synthetic_tactic_labels()["synthetic:4"]
            self.assertFalse(saved["answer"])
            self.assertEqual(saved["training_label"], "unstructured")

    def test_image_space_graph_edges_are_toggleable_and_team_local(self):
        self.assertIn('id="tactic-filter"', HTML)
        self.assertIn('value="traps"', HTML)
        self.assertIn("function populateClipSelect", HTML)
        self.assertIn("findIndex", HTML)
        self.assertIn("video.pause()", HTML)
        self.assertIn("renderTactic(first)", HTML)
        self.assertIn('id="graph-edges"', HTML)
        self.assertIn('id="pitch"', HTML)
        self.assertIn('id="ball"', HTML)
        self.assertIn("function drawBall", HTML)
        self.assertIn("function renderTactic", HTML)
        self.assertIn("clip().tactical_model", HTML)
        self.assertIn("abstain_reasons", HTML)
        self.assertIn('id="tactic-name"', HTML)
        self.assertIn("function drawCanonicalPitch", HTML)
        self.assertIn("ball?.pitch_xy", HTML)
        self.assertIn("other.identity.label!==node.identity.label", HTML)
        self.assertIn("neighbour.distance<=22", HTML)
        self.assertIn("<=12", HTML)
        self.assertIn("while(remaining.size)", HTML)

    def test_trap_event_page_has_one_fixed_label_and_canonical_map(self):
        self.assertIn("exactly one fixed tactical label", TRAP_EVENT_HTML)
        self.assertIn('src="/trap-video?event=', TRAP_EVENT_HTML)
        self.assertIn("<canvas>", TRAP_EVENT_HTML)
        self.assertIn("event.display", TRAP_EVENT_HTML)

    def test_reviewed_events_pair_broadcast_and_canonical_videos(self):
        self.assertIn("one fixed label", REVIEWED_EVENT_HTML)
        self.assertIn("kind=broadcast", REVIEWED_EVENT_HTML)
        self.assertIn("kind=canonical", REVIEWED_EVENT_HTML)
        self.assertIn("boxes, IDs, team edges, ball", REVIEWED_EVENT_HTML)


if __name__ == "__main__":
    unittest.main()
