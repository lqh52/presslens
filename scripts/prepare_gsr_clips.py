#!/usr/bin/env python3
"""Extract short 720p clips and write a manifest for the GSR batch runner."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def high_resolution_source(source: Path) -> Path:
    if source.name.endswith("_224p.mkv"):
        source = source.with_name(source.name.replace("_224p.mkv", "_720p.mkv"))
    if not source.is_file():
        raise FileNotFoundError(f"Missing 720p source: {source}")
    return source


def game_id(source: Path) -> str:
    parts = source.parent.parts
    if len(parts) < 3:
        return source.parent.name
    return "/".join(parts[-3:])


def valid_clip(path: Path, expected_frames: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        value = subprocess.check_output(command, text=True).strip()
        return int(value) == expected_frames
    except (subprocess.CalledProcessError, ValueError):
        return False


def extract(
    row: dict,
    output_dir: Path,
    duration: float,
    fps: int,
    force: bool,
) -> dict:
    source = high_resolution_source(Path(row["source_video"]))
    window_duration = float(row["end_seconds"] - row["start_seconds"])
    if duration > window_duration:
        raise ValueError(
            f"Requested {duration}s from {row['id']}, only {window_duration}s available"
        )
    start = float(row["start_seconds"]) + (window_duration - duration) / 2
    destination = output_dir / f"{row['id']}.mp4"
    expected_frames = int(round(duration * fps))
    if force or not valid_clip(destination, expected_frames):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps},scale=1280:720:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)
    if not valid_clip(destination, expected_frames):
        raise RuntimeError(
            f"Extracted clip failed validation ({expected_frames} frames): "
            f"{destination}"
        )
    return {
        **row,
        "match_id": game_id(source),
        "source_video_720p": str(source),
        "source_start_seconds": round(start, 3),
        "duration_seconds": duration,
        "fps": fps,
        "nframes": expected_frames,
        "clip_path": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/gsr-arsenal-expansion"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/arsenal_expansion/gsr-clips.json"),
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    rows = payload["clips"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                extract,
                row,
                args.output_dir,
                args.duration,
                args.fps,
                args.force,
            ): row
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            print(f"Prepared {index}/{len(rows)}")
    order = {row["id"]: index for index, row in enumerate(rows)}
    completed.sort(key=lambda row: order[row["id"]])
    output = {
        "schema_version": 1,
        "source": "Local NDA-authorised SoccerNet 720p excerpts",
        "duration_seconds": args.duration,
        "fps": args.fps,
        "clips": completed,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(completed)} GSR clips to {args.output_manifest}")


if __name__ == "__main__":
    main()
