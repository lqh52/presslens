#!/usr/bin/env python3
"""Gate, temporally smooth, and package tactical predictions for review."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DISPLAY = {
    "unstructured": "Unstructured",
    "central_screen": "Central screen",
    "trap_left": "Trap left",
    "trap_right": "Trap right",
    "high_press": "High press",
    "abstain": "Insufficient evidence",
}


def package(rows: list[dict], output_dir: Path, frame_count: int = 100) -> None:
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip_id, clip_rows in sorted(by_clip.items()):
        clip_rows.sort(key=lambda row: int(row["frame"]))
        frames = [{"frame": index, "prediction": None} for index in range(frame_count)]
        accepted = []
        for position, row in enumerate(clip_rows):
            neighbours = clip_rows[max(0, position - 3) : position + 4]
            probabilities = {
                label: float(np.median([item["probabilities"][label] for item in neighbours]))
                for label in row["probabilities"]
            }
            label = max(probabilities, key=probabilities.get)
            confidence = probabilities[label]
            reasons = []
            if not row.get("possession_confident"):
                reasons.append("possession")
            if not row.get("direction_confident"):
                reasons.append("attack direction")
            if int(row.get("visible_nodes", 0)) < 7:
                reasons.append("visible players")
            if confidence < 0.65:
                reasons.append("model confidence")
            accepted_label = label if not reasons else "abstain"
            prediction = {
                "label": accepted_label,
                "display": DISPLAY[accepted_label],
                "confidence": round(confidence, 4),
                "raw_label": label,
                "raw_display": DISPLAY[label],
                "probabilities": {
                    key: round(value, 4) for key, value in probabilities.items()
                },
                "possession_team": row.get("possession_team"),
                "possession_confidence": row.get("possession_confidence"),
                "direction_confidence": row.get("direction_confidence"),
                "visible_nodes": row.get("visible_nodes"),
                "abstain_reasons": reasons,
            }
            frame = int(row["frame"])
            if 0 <= frame < frame_count:
                frames[frame]["prediction"] = prediction
            if accepted_label != "abstain":
                accepted.append(prediction)
        counts = Counter(row["label"] for row in accepted)
        dominant = counts.most_common(1)[0][0] if counts else "abstain"
        dominant_rows = [row for row in accepted if row["label"] == dominant]
        result = {
            "schema_version": 1,
            "clip_id": clip_id,
            "model": "tactical_graph_realistic_synthetic.pt",
            "summary": {
                "dominant_pattern": dominant,
                "display": DISPLAY[dominant],
                "confidence": (
                    round(float(np.median([row["confidence"] for row in dominant_rows])), 4)
                    if dominant_rows
                    else 0.0
                ),
                "accepted_frames": len(accepted),
                "graph_frames": len(clip_rows),
                "frame_count": frame_count,
                "coverage": round(len(accepted) / frame_count, 4),
                "counts": counts,
            },
            "frames": frames,
        }
        (output_dir / f"{clip_id}.json").write_text(
            json.dumps(result, separators=(",", ":")) + "\n"
        )
        print(
            f"{clip_id}: {DISPLAY[dominant]} "
            f"({result['summary']['confidence']:.0%}, "
            f"coverage {result['summary']['coverage']:.0%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=100)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text().splitlines()
        if line.strip()
    ]
    package(rows, args.output_dir, args.frame_count)


if __name__ == "__main__":
    main()
