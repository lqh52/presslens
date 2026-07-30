#!/usr/bin/env python3
"""Append explicitly selected clips from a source manifest without duplicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    source = {
        row["id"]: row for row in json.loads(args.source.read_text())["clips"]
    }
    existing = {row["id"] for row in payload["clips"]}
    for clip_id in args.include:
        if clip_id not in source:
            raise ValueError(f"Unknown clip: {clip_id}")
        if clip_id not in existing:
            payload["clips"].append(
                {
                    **source[clip_id],
                    "triage": {
                        "predicted_class": "trap_candidate_geometry_rule",
                        "selection_source": "legacy_auditable_geometry_rule",
                    },
                }
            )
            existing.add(clip_id)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Manifest now contains {len(payload['clips'])} clips")


if __name__ == "__main__":
    main()
