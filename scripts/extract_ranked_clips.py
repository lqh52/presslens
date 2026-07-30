#!/usr/bin/env python3
"""Extract ranked candidate windows for annotation without re-encoding."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ranked_candidates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clips/candidates"),
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(payload["candidates"][: args.limit], start=1):
        destination = args.output_dir / f"{index:03d}_{item['id']}.mp4"
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(item["start_seconds"]),
            "-i",
            item["source_video"],
            "-t",
            str(item["end_seconds"] - item["start_seconds"]),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(destination),
        ]
        subprocess.run(command, check=True)
        print(f"[{index:03d}] {destination}")


if __name__ == "__main__":
    main()
