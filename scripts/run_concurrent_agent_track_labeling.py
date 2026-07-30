#!/usr/bin/env python3
"""Run independent Gemini labeling shards concurrently and merge their outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def split_round_robin(items: list[str], workers: int) -> list[list[str]]:
    shards = [[] for _ in range(min(workers, len(items)))]
    for index, item in enumerate(items):
        shards[index % len(shards)].append(item)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--key", action="append", required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    shards = split_round_robin(args.key, args.workers)
    shard_dir = args.output.parent / f"{args.output.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, keys in enumerate(shards):
        shard_output = shard_dir / f"part-{index:02d}.jsonl"
        command = [
            sys.executable,
            "scripts/run_agent_track_labeling.py",
            "label",
            "--evidence",
            str(args.evidence),
            "--output",
            str(shard_output),
            "--env",
            str(args.env),
            "--model",
            args.model,
            "--include-unreviewed",
            "--retries",
            str(args.retries),
            "--retry-delay",
            str(args.retry_delay),
            "--request-delay",
            str(args.request_delay),
        ]
        for key in keys:
            command.extend(["--key", key])
        print(f"Starting Gemini shard {index + 1}/{len(shards)}: {len(keys)} tracks")
        processes.append((index, subprocess.Popen(command)))

    failed = []
    for index, process in processes:
        return_code = process.wait()
        if return_code:
            failed.append((index, return_code))
    if failed:
        raise RuntimeError(f"Gemini shards failed: {failed}")

    rows = {}
    for path in sorted(shard_dir.glob("part-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for key in args.key:
            if key in rows:
                handle.write(json.dumps(rows[key]) + "\n")
    temporary.replace(args.output)
    print(
        f"Merged {len(rows)}/{len(args.key)} Gemini predictions into {args.output}"
    )


if __name__ == "__main__":
    main()
