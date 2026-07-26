from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.run_gsr_batch import (
    GSRJob,
    build_tracklab_command,
    find_completed_state,
    load_jobs,
    parse_gpu_ids,
    probe_nframes,
    safe_identifier,
    valid_completed_state,
)


class GSRBatchRunnerTest(unittest.TestCase):
    def test_manifest_clip_path_and_frame_count_are_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clip = root / "clip.mp4"
            clip.touch()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "id": "arsenal-v-chelsea-h1-0001",
                                "clip_path": "clip.mp4",
                                "nframes": 200,
                            }
                        ]
                    }
                )
            )
            jobs = load_jobs(manifest, root)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].id, "arsenal-v-chelsea-h1-0001")
        self.assertEqual(jobs[0].nframes, 200)
        self.assertEqual(jobs[0].state_filename, "arsenal-v-chelsea-h1-0001.pklz")

    def test_ffprobe_exact_frame_count_wins(self):
        response = json.dumps(
            {
                "streams": [
                    {
                        "nb_read_frames": "200",
                        "nb_frames": "N/A",
                        "duration": "8.0",
                        "avg_frame_rate": "25/1",
                    }
                ]
            }
        )
        with patch("scripts.run_gsr_batch.subprocess.run") as mocked:
            mocked.return_value.stdout = response
            self.assertEqual(probe_nframes(Path("clip.mp4")), 200)

    def test_command_uses_external_video_and_item_id(self):
        job = GSRJob(
            id="match-h2-0042",
            clip_path=Path("/tmp/match-h2-0042.mp4"),
            nframes=200,
            experiment_name="match-h2-0042",
            state_filename="match-h2-0042.pklz",
        )
        command = build_tracklab_command(
            Path("/opt/tracklab"),
            job,
            Path("/tmp/outputs/match-h2-0042/run"),
        )
        self.assertIn("dataset=video", command)
        self.assertIn("dataset.nframes=200", command)
        self.assertIn("experiment_name=match-h2-0042", command)
        self.assertIn(
            "state.save_file=states/match-h2-0042.pklz",
            command,
        )
        self.assertIn(
            "hydra.run.dir=/tmp/outputs/match-h2-0042/run",
            command,
        )

    def test_valid_state_requires_archive_log_and_matching_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clip = root / "clip.mp4"
            clip.touch()
            job = GSRJob(
                id="clip",
                clip_path=clip.resolve(),
                nframes=200,
                experiment_name="clip",
                state_filename="clip.pklz",
            )
            run = root / "outputs" / "clip" / "2026-01-01" / "00-00-00"
            state = run / "states" / "clip.pklz"
            state.parent.mkdir(parents=True)
            configs = run / "configs"
            configs.mkdir()
            (configs / "overrides.yaml").write_text(
                "\n".join(
                    [
                        f"- dataset.video_path={clip.resolve()}",
                        "- dataset.nframes=200",
                    ]
                )
                + "\n"
            )
            (run / "main.log").write_text(
                "Tracking ended, final TrackerState stats:\n"
                "Saved state at : /somewhere/clip.pklz\n"
            )
            with zipfile.ZipFile(state, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "summary.json",
                    json.dumps({"columns": {"detection": [], "image": []}}),
                )
                archive.writestr("0.pkl", b"detections")
                archive.writestr("0_image.pkl", b"images")
            valid, message = valid_completed_state(state, job)
            self.assertTrue(valid, message)
            self.assertEqual(
                find_completed_state(root / "outputs", job),
                state,
            )

    def test_incomplete_log_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clip = root / "clip.mp4"
            clip.touch()
            job = GSRJob(
                id="clip",
                clip_path=clip.resolve(),
                nframes=200,
                experiment_name="clip",
                state_filename="clip.pklz",
            )
            run = root / "run"
            state = run / "states" / "clip.pklz"
            state.parent.mkdir(parents=True)
            (run / "configs").mkdir()
            (run / "configs" / "overrides.yaml").write_text(
                f"- dataset.video_path={clip.resolve()}\n"
                "- dataset.nframes=200\n"
            )
            (run / "main.log").write_text("Tracking started\n")
            with zipfile.ZipFile(state, "w") as archive:
                archive.writestr(
                    "summary.json",
                    json.dumps({"columns": {"detection": [], "image": []}}),
                )
                archive.writestr("0.pkl", b"detections")
                archive.writestr("0_image.pkl", b"images")
            valid, _ = valid_completed_state(state, job)
            self.assertFalse(valid)

    def test_gpu_and_identifier_validation(self):
        self.assertEqual(parse_gpu_ids("1, 2,3"), ["1", "2", "3"])
        self.assertEqual(safe_identifier(" Arsenal / Chelsea "), "Arsenal-Chelsea")
        with self.assertRaises(ValueError):
            parse_gpu_ids("1,1")


if __name__ == "__main__":
    unittest.main()
