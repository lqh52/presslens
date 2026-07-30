#!/usr/bin/env python3
"""Extract short, low-FPS videos for contiguous accepted trap events."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


TRAPS = {"trap_left", "trap_right"}


def contiguous_events(frames: list[dict]) -> list[list[dict]]:
    events, current = [], []
    for row in frames:
        prediction = row.get("prediction")
        if not prediction or prediction.get("label") not in TRAPS:
            if current:
                events.append(current)
                current = []
            continue
        if (
            current
            and (
                prediction["label"] != current[-1]["prediction"]["label"]
                or int(row["frame"]) != int(current[-1]["frame"]) + 1
            )
        ):
            events.append(current)
            current = []
        current.append(row)
    if current:
        events.append(current)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=25)
    parser.add_argument("--output-fps", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--only-event")
    parser.add_argument("--override-label", choices=sorted(TRAPS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = []
    for prediction_path in sorted((args.directory / "tactical-model").glob("*.json")):
        payload = json.loads(prediction_path.read_text())
        clip_id = payload["clip_id"]
        result_path = (
            args.directory
            / "results"
            / "yolo26m-botsort-high-recall"
            / f"{clip_id}.json"
        )
        result = json.loads(result_path.read_text())
        projection = json.loads(
            (args.directory / "pitch-projections" / f"{clip_id}.json").read_text()
        )
        balls = json.loads(
            (args.directory / "ball-tracking" / f"{clip_id}.json").read_text()
        )
        identity = json.loads(
            (args.directory / "identities" / f"{clip_id}.json").read_text()
        )
        projection_frames = {
            int(row["frame"]): row for row in projection["frames"]
        }
        ball_frames = {int(row["frame"]): row for row in balls["frames"]}
        identities = {
            str(row["track_id"]): {
                "label": row["label"],
                "goalkeeper": row.get("goalkeeper", False),
            }
            for row in identity["tracks"]
        }
        for event_index, rows in enumerate(contiguous_events(payload["frames"]), 1):
            first, last = int(rows[0]["frame"]), int(rows[-1]["frame"])
            centre = (first + last) // 2
            start = max(0.0, first / args.source_fps - args.padding)
            duration = max(
                1.6, (last - first + 1) / args.source_fps + 2 * args.padding
            )
            event_id = f"{clip_id}-{rows[0]['prediction']['label']}-{event_index}"
            if args.only_event and event_id != args.only_event:
                continue
            reviewed_label = args.override_label or rows[0]["prediction"]["label"]
            output_path = args.output_dir / f"{event_id}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.4f}",
                    "-i",
                    str(Path(result["clip_path"])),
                    "-t",
                    f"{duration:.4f}",
                    "-vf",
                    f"fps={args.output_fps}",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            events.append(
                {
                    "id": event_id,
                    "clip_id": clip_id,
                    "label": reviewed_label,
                    "display": reviewed_label.replace("_", " ").title(),
                    "model_label": rows[0]["prediction"]["label"],
                    "human_corrected": reviewed_label
                    != rows[0]["prediction"]["label"],
                    "confidence": round(
                        sum(row["prediction"]["confidence"] for row in rows)
                        / len(rows),
                        4,
                    ),
                    "source_frames": [first, last],
                    "source_time_seconds": round(centre / args.source_fps, 3),
                    "video": output_path.name,
                    "fps": args.output_fps,
                    "duration_seconds": round(duration, 3),
                    "canonical": {
                        "objects": projection_frames.get(centre, {}).get(
                            "objects", []
                        ),
                        "ball": ball_frames.get(centre, {}).get("ball"),
                        "identities": identities,
                    },
                }
            )
    manifest = {
        "schema_version": 1,
        "description": "One accepted tactical label per low-FPS event excerpt",
        "events": events,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote {len(events)} trap event excerpts to {args.output_dir}")


if __name__ == "__main__":
    main()
