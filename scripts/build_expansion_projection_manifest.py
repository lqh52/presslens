#!/usr/bin/env python3
"""Build projection configuration for a selected Arsenal expansion subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {
        row["id"] for row in json.loads(args.selection.read_text())["clips"]
    }
    status = json.loads(args.status.read_text())["clips"]
    missing = selected - set(status)
    if missing:
        raise ValueError(f"Missing downstream states: {sorted(missing)}")
    output = {
        "clips": {
            clip_id: {
                "state": status[clip_id]["state_path"],
                "frame_offset": 0,
            }
            for clip_id in sorted(selected)
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(selected)} projection states to {args.output}")


if __name__ == "__main__":
    main()
