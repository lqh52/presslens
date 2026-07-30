#!/usr/bin/env python3
"""Infer auditable tactical patterns from projected players and the tracked ball."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


TEAM_LABELS = {"team_a", "team_b"}
DISPLAY_NAMES = {
    "touchline_trap": "Touchline trap",
    "intense_pressure": "Intense pressure",
    "central_block": "Central block",
    "compact_block": "Compact block",
    "low_pressure": "Low pressure",
    "unstructured": "Unstructured",
    "abstain": "Insufficient evidence",
}


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def frame_pattern(
    objects: list[dict[str, Any]],
    ball: dict[str, Any] | None,
    identities: dict[int, str],
) -> dict[str, Any]:
    players = [
        {
            "track_id": int(item["track_id"]),
            "team": identities.get(int(item["track_id"])),
            "xy": (float(item["x"]), float(item["y"])),
        }
        for item in objects
        if identities.get(int(item["track_id"])) in TEAM_LABELS
    ]
    if not ball or ball.get("pitch_xy") is None or len(players) < 4:
        return {"label": "abstain", "confidence": 0.0, "reason": "missing_geometry"}
    ball_xy = tuple(map(float, ball["pitch_xy"]))
    nearest = sorted(
        ((distance(player["xy"], ball_xy), player) for player in players),
        key=lambda item: item[0],
    )
    holder_distance, holder = nearest[0]
    opposing_nearest = next(
        (value for value in nearest if value[1]["team"] != holder["team"]), None
    )
    margin = (
        opposing_nearest[0] - holder_distance if opposing_nearest is not None else 0.0
    )
    possession_confidence = max(
        0.0, min(1.0, 0.72 - holder_distance / 12.0 + margin / 12.0)
    )
    if holder_distance > 7.0 or possession_confidence < 0.32:
        return {
            "label": "abstain",
            "confidence": round(possession_confidence, 4),
            "reason": "possession_ambiguous",
            "nearest_player_m": round(holder_distance, 3),
        }

    possession_team = holder["team"]
    defending_team = "team_b" if possession_team == "team_a" else "team_a"
    defenders = [player for player in players if player["team"] == defending_team]
    defenders_to_ball = sorted(distance(player["xy"], ball_xy) for player in defenders)
    within_8 = sum(value <= 8.0 for value in defenders_to_ball)
    within_12 = sum(value <= 12.0 for value in defenders_to_ball)
    within_16 = sum(value <= 16.0 for value in defenders_to_ball)
    if len(defenders) >= 3:
        xs = [player["xy"][0] for player in defenders]
        ys = [player["xy"][1] for player in defenders]
        width, depth = max(ys) - min(ys), max(xs) - min(xs)
    else:
        width = depth = 99.0

    label, confidence, reason = "unstructured", 0.48, "no_stable_shape_rule"
    if abs(ball_xy[1]) >= 24.0 and within_12 >= 2:
        label, confidence, reason = (
            "touchline_trap",
            min(0.94, 0.58 + 0.09 * within_12 + 0.08 * (abs(ball_xy[1]) - 24) / 10),
            "ball_near_touchline+multi_player_pressure",
        )
    elif within_16 >= 3 and within_8 >= 2:
        label, confidence, reason = (
            "intense_pressure",
            min(0.92, 0.56 + 0.07 * within_16 + 0.06 * within_8),
            "dense_pressure_around_ball",
        )
    elif abs(ball_xy[1]) <= 11.5 and within_16 >= 2:
        label, confidence, reason = (
            "central_block",
            min(0.88, 0.55 + 0.07 * within_16),
            "central_ball+screening_density",
        )
    elif len(defenders) >= 4 and width <= 36.0 and depth <= 30.0:
        label, confidence, reason = (
            "compact_block",
            min(0.84, 0.55 + (36 - width) / 100 + (30 - depth) / 100),
            "compact_defensive_team_shape",
        )
    elif not defenders_to_ball or defenders_to_ball[0] > 14.0:
        label, confidence, reason = (
            "low_pressure",
            min(0.88, 0.58 + (defenders_to_ball[0] - 14) / 40)
            if defenders_to_ball
            else 0.58,
            "no_nearby_defender",
        )

    return {
        "label": label,
        "display": DISPLAY_NAMES[label],
        "confidence": round(confidence * possession_confidence, 4),
        "reason": reason,
        "possession_team": possession_team,
        "possession_confidence": round(possession_confidence, 4),
        "ball_xy": [round(value, 3) for value in ball_xy],
        "holder_track_id": holder["track_id"],
        "nearest_player_m": round(holder_distance, 3),
        "pressure": {
            "within_8m": within_8,
            "within_12m": within_12,
            "within_16m": within_16,
            "nearest_defender_m": (
                round(defenders_to_ball[0], 3) if defenders_to_ball else None
            ),
        },
        "defending_shape": {
            "team": defending_team,
            "visible_players": len(defenders),
            "width_m": round(width, 3),
            "depth_m": round(depth, 3),
        },
    }


def temporally_stabilize(rows: list[dict[str, Any]], radius: int = 6) -> None:
    raw_labels = [row["pattern"]["label"] for row in rows]
    for index, row in enumerate(rows):
        if raw_labels[index] == "abstain":
            continue
        window = raw_labels[max(0, index - radius) : index + radius + 1]
        usable = [label for label in window if label != "abstain"]
        if not usable:
            continue
        label, votes = Counter(usable).most_common(1)[0]
        if votes / len(usable) >= 0.54:
            row["pattern"]["raw_label"] = row["pattern"]["label"]
            row["pattern"]["label"] = label
            row["pattern"]["display"] = DISPLAY_NAMES[label]
            row["pattern"]["temporal_support"] = round(votes / len(usable), 4)


def process_clip(
    projection_path: Path,
    ball_path: Path,
    identity_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    projection = json.loads(projection_path.read_text())
    ball = json.loads(ball_path.read_text())
    identity = json.loads(identity_path.read_text())
    identities = {
        int(track["track_id"]): track["label"]
        for track in identity["tracks"]
        if track.get("label") in TEAM_LABELS
    }
    ball_frames = {int(row["frame"]): row.get("ball") for row in ball["frames"]}
    rows = []
    for frame in projection["frames"]:
        index = int(frame["frame"])
        rows.append(
            {
                "frame": index,
                "pattern": frame_pattern(
                    frame.get("objects", []), ball_frames.get(index), identities
                ),
            }
        )
    temporally_stabilize(rows)
    reliable = [
        row["pattern"]
        for row in rows
        if row["pattern"]["label"] != "abstain"
        and row["pattern"].get("confidence", 0.0) >= 0.35
    ]
    counts = Counter(row["label"] for row in reliable)
    dominant = counts.most_common(1)[0][0] if counts else "abstain"
    dominant_rows = [row for row in reliable if row["label"] == dominant]
    result = {
        "schema_version": 1,
        "clip_id": projection["clip_id"],
        "method": "auditable_geometry_rules_v1",
        "taxonomy": DISPLAY_NAMES,
        "summary": {
            "dominant_pattern": dominant,
            "display": DISPLAY_NAMES[dominant],
            "confidence": (
                round(median(row["confidence"] for row in dominant_rows), 4)
                if dominant_rows
                else 0.0
            ),
            "reliable_frames": len(reliable),
            "frame_count": len(rows),
            "coverage": round(len(reliable) / len(rows), 4) if rows else 0.0,
            "counts": counts,
        },
        "frames": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    for projection_path in sorted((directory / "pitch-projections").glob("*.json")):
        clip_id = projection_path.stem
        result = process_clip(
            projection_path,
            directory / "ball-tracking" / f"{clip_id}.json",
            directory / "identities" / f"{clip_id}.json",
            directory / "tactical-patterns" / f"{clip_id}.json",
        )
        summary = result["summary"]
        print(
            f"{clip_id}: {summary['display']} "
            f"({summary['confidence']:.0%}, coverage {summary['coverage']:.0%})"
        )


if __name__ == "__main__":
    main()
