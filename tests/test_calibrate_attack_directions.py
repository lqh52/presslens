from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.calibrate_attack_directions import (
    CalibrationJobContext,
    CollectionStats,
    collect_clip_calibrations,
    preflight_job_contexts,
    summary_payload,
)


def fake_job(clip_id: str):
    return SimpleNamespace(id=clip_id, experiment_name=clip_id)


def team_config() -> dict:
    return {
        "mapping_status": "unreviewed",
        "cluster_mapping_scope": "tracklab_team_per_sequence",
        "cluster_to_internal": None,
    }


def calibrated() -> dict:
    return {
        "status": "calibrated",
        "confident": True,
        "confidence": 0.9,
        "directions": {"left": 1, "right": -1},
    }


class CalibrationResilienceTest(unittest.TestCase):
    def contexts(self) -> list[CalibrationJobContext]:
        return [
            CalibrationJobContext(
                job=fake_job("bad"),
                match_id="fixture",
                half=1,
                team_config=team_config(),
            ),
            CalibrationJobContext(
                job=fake_job("good"),
                match_id="fixture",
                half=1,
                team_config=team_config(),
            ),
        ]

    def test_corrupt_pickle_is_excluded_and_next_clip_continues(self):
        states = [Path("/states/bad.pklz"), Path("/states/good.pklz")]
        with (
            patch(
                "scripts.calibrate_attack_directions.find_completed_state",
                side_effect=states,
            ),
            patch(
                "scripts.calibrate_attack_directions.load_detections",
                side_effect=[
                    pickle.UnpicklingError("invalid load key"),
                    object(),
                ],
            ),
            patch(
                "scripts.calibrate_attack_directions.align_internal_teams",
                return_value=(object(), {"mapping_status": "unreviewed"}),
            ),
            patch(
                "scripts.calibrate_attack_directions.calibrate_detections",
                return_value=calibrated(),
            ),
        ):
            grouped, stats = collect_clip_calibrations(
                self.contexts(),
                state_root=Path("/states"),
                team_registry=Path("/registry.json"),
            )

        evidence = grouped[("fixture", 1)]
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["status"], "excluded_state_load_error")
        self.assertEqual(evidence[0]["error"]["stage"], "state_load")
        self.assertTrue(
            evidence[0]["error"]["type"].endswith(".UnpicklingError")
        )
        self.assertEqual(evidence[0]["error"]["message"], "invalid load key")
        self.assertEqual(evidence[0]["error"]["path"], "/states/bad.pklz")
        self.assertEqual(evidence[1]["status"], "calibrated")
        self.assertEqual(stats.clips_with_completed_state, 2)
        self.assertEqual(stats.clips_waiting_for_state, 0)

    def test_ambiguous_cluster_alignment_is_audited_and_continues(self):
        states = [Path("/states/bad.pklz"), Path("/states/good.pklz")]
        with (
            patch(
                "scripts.calibrate_attack_directions.find_completed_state",
                side_effect=states,
            ),
            patch(
                "scripts.calibrate_attack_directions.load_detections",
                side_effect=[object(), object()],
            ),
            patch(
                "scripts.calibrate_attack_directions.align_internal_teams",
                side_effect=[
                    ValueError("cluster-to-side votes are tied"),
                    (object(), {"mapping_status": "unreviewed"}),
                ],
            ),
            patch(
                "scripts.calibrate_attack_directions.calibrate_detections",
                return_value=calibrated(),
            ),
        ):
            grouped, _ = collect_clip_calibrations(
                self.contexts(),
                state_root=Path("/states"),
                team_registry=Path("/registry.json"),
            )

        evidence = grouped[("fixture", 1)]
        self.assertEqual(
            evidence[0]["status"], "excluded_team_alignment_error"
        )
        self.assertEqual(evidence[0]["error"]["type"], "builtins.ValueError")
        self.assertEqual(
            evidence[0]["error"]["message"],
            "cluster-to-side votes are tied",
        )
        self.assertEqual(evidence[1]["status"], "calibrated")

    def test_invalid_registry_is_fatal_during_preflight(self):
        jobs = [fake_job("clip")]
        rows = {"clip": {"id": "clip", "match_id": "fixture", "half": 1}}
        with patch(
            "scripts.calibrate_attack_directions.load_match_team_config",
            side_effect=ValueError("invalid registry"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid registry"):
                preflight_job_contexts(
                    jobs,
                    rows,
                    team_registry=Path("/registry.json"),
                )

    def test_invalid_manifest_metadata_is_fatal_during_preflight(self):
        jobs = [fake_job("clip")]
        rows = {"clip": {"id": "clip", "match_id": "", "half": 1}}
        with self.assertRaisesRegex(ValueError, "match_id"):
            preflight_job_contexts(
                jobs,
                rows,
                team_registry=Path("/registry.json"),
            )

    def test_non_integral_manifest_half_is_fatal_during_preflight(self):
        jobs = [fake_job("clip")]
        rows = {
            "clip": {"id": "clip", "match_id": "fixture", "half": 1.5}
        }
        with self.assertRaisesRegex(ValueError, "invalid half"):
            preflight_job_contexts(
                jobs,
                rows,
                team_registry=Path("/registry.json"),
            )

    def test_summary_counts_exclusions_separately_and_deterministically(self):
        grouped = {
            ("fixture", 1): [
                calibrated(),
                {
                    "status": "abstained_insufficient_spatial_evidence",
                    "confident": False,
                    "confidence": 0.0,
                    "directions": None,
                },
                {
                    "status": "excluded_team_alignment_error",
                    "confident": False,
                    "confidence": 0.0,
                    "directions": None,
                    "error": {"stage": "team_alignment"},
                },
                {
                    "status": "excluded_state_load_error",
                    "confident": False,
                    "confidence": 0.0,
                    "directions": None,
                    "error": {"stage": "state_load"},
                },
            ]
        }
        summary = summary_payload(
            grouped,
            clips_selected=5,
            collection_stats=CollectionStats(
                clips_waiting_for_state=1,
                clips_with_completed_state=4,
            ),
            match_halves=1,
            match_halves_confident=1,
        )
        self.assertEqual(summary["clips_calibrated"], 1)
        self.assertEqual(summary["clips_abstained"], 1)
        self.assertEqual(summary["clips_excluded"], 2)
        self.assertEqual(summary["clips_calibrated_or_abstained"], 2)
        self.assertEqual(summary["clips_accounted_for"], 5)
        self.assertEqual(
            list(summary["exclusions_by_stage"]),
            ["state_load", "team_alignment"],
        )


if __name__ == "__main__":
    unittest.main()
