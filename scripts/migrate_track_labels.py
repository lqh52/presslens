#!/usr/bin/env python3
"""Transfer reviewed track labels between aligned tracking runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_player_tracking import box_iou


def track_boxes(payload: dict[str, Any]) -> dict[int, dict[int, list[float]]]:
    tracks: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for frame in payload["frames"]:
        for detection in frame["detections"]:
            if detection.get("track_id") is not None:
                tracks[int(detection["track_id"])][int(frame["frame"])] = detection[
                    "bbox"
                ]
    return dict(tracks)


def similarity(
    old: dict[int, list[float]], new: dict[int, list[float]]
) -> tuple[float, float, float]:
    shared = sorted(set(old) & set(new))
    if not shared:
        return 0.0, 0.0, 0.0
    overlaps = [box_iou(old[frame], new[frame]) for frame in shared]
    mean_iou = sum(overlaps) / len(overlaps)
    new_coverage = len(shared) / max(len(new), 1)
    old_coverage = len(shared) / max(len(old), 1)
    return mean_iou * new_coverage, mean_iou, old_coverage


def migrate(
    old_results: Path,
    new_results: Path,
    old_labels_path: Path,
    output: Path,
    *,
    minimum_iou: float,
    minimum_new_coverage: float,
) -> dict[str, Any]:
    old_labels = json.loads(old_labels_path.read_text()).get("labels", {})
    migrated: dict[str, Any] = {}
    audit = []
    for new_path in sorted(new_results.glob("*.json")):
        clip_id = json.loads(new_path.read_text())["clip_id"]
        old_path = old_results / new_path.name
        if not old_path.exists():
            continue
        old_tracks = track_boxes(json.loads(old_path.read_text()))
        new_tracks = track_boxes(json.loads(new_path.read_text()))
        labelled_old = {
            int(key.rsplit(":", 1)[1]): value
            for key, value in old_labels.items()
            if key.startswith(f"{clip_id}:")
        }
        for new_id, new_boxes in new_tracks.items():
            candidates = []
            for old_id, label in labelled_old.items():
                if old_id not in old_tracks:
                    continue
                score, mean_iou, old_coverage = similarity(
                    old_tracks[old_id], new_boxes
                )
                candidates.append(
                    (score, mean_iou, old_coverage, old_id, label)
                )
            if not candidates:
                continue
            score, mean_iou, old_coverage, old_id, label = max(candidates)
            if mean_iou < minimum_iou or score / max(mean_iou, 1e-9) < minimum_new_coverage:
                continue
            key = f"{clip_id}:{new_id}"
            migrated[key] = {
                "clip_id": clip_id,
                "track_id": new_id,
                "label": label["label"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "migrated_from_track_id": old_id,
                "migration_mean_iou": round(mean_iou, 6),
                "migration_old_coverage": round(old_coverage, 6),
            }
            audit.append(
                {
                    "clip_id": clip_id,
                    "old_track_id": old_id,
                    "new_track_id": new_id,
                    "label": label["label"],
                    "mean_iou": round(mean_iou, 6),
                }
            )
    result = {
        "schema_version": 1,
        "labels": migrated,
        "migration": {
            "source_labels": len(old_labels),
            "migrated_labels": len(migrated),
            "minimum_iou": minimum_iou,
            "minimum_new_coverage": minimum_new_coverage,
            "audit": audit,
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--old-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.65)
    parser.add_argument("--minimum-new-coverage", type=float, default=0.6)
    args = parser.parse_args()
    result = migrate(
        args.old_results,
        args.new_results,
        args.old_labels,
        args.output,
        minimum_iou=args.minimum_iou,
        minimum_new_coverage=args.minimum_new_coverage,
    )
    print(
        json.dumps(
            {
                "source_labels": result["migration"]["source_labels"],
                "migrated_labels": result["migration"]["migrated_labels"],
            }
        )
    )


if __name__ == "__main__":
    main()
