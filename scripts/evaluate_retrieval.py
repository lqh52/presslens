#!/usr/bin/env python3
"""Evaluate the ranked build-up candidates after human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ranked_candidates.json"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/reviewed.json"),
    )
    parser.add_argument("--k", type=int, action="append", default=[5, 10, 20, 50])
    args = parser.parse_args()

    ranked = json.loads(args.manifest.read_text())["candidates"]
    reviewed = json.loads(args.annotations.read_text())["annotations"]
    sequence = [item for item in ranked if item["id"] in reviewed]
    positives = sum(bool(reviewed[item["id"]]["is_build_up"]) for item in sequence)
    print(f"Reviewed: {len(sequence)}/{len(ranked)}")
    print(f"Build-up positives: {positives}/{len(sequence)}")
    for k in dict.fromkeys(args.k):
        subset = sequence[:k]
        if not subset:
            print(f"Precision@{k}: n/a")
            continue
        hits = sum(bool(reviewed[item["id"]]["is_build_up"]) for item in subset)
        print(f"Precision@{k}: {hits / len(subset):.3f} ({hits}/{len(subset)})")


if __name__ == "__main__":
    main()
