#!/usr/bin/env python3
"""Build canonical team-shape maps from SkillCorner phase and tracking data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BLOCK_LABELS = ("high_block", "medium_block", "low_block")


def canonical_xy(
    x: float, y: float, attacking_side: str
) -> list[float]:
    """Orient the possession team left-to-right on a 105 x 68 metre pitch."""
    x, y = oriented_xy(x, y, attacking_side)
    return [round(x + 52.5, 3), round(y + 34.0, 3)]


def oriented_xy(
    x: float, y: float, attacking_side: str
) -> list[float]:
    """Return centred metric coordinates with possession attacking right."""
    if attacking_side == "right_to_left":
        x, y = -x, -y
    return [round(x, 3), round(y, 3)]


def read_phases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_target_frames(
    path: Path, target_frames: set[int]
) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            frame = int(row["frame"])
            if frame in target_frames:
                frames[frame] = row
                if len(frames) == len(target_frames):
                    break
    return frames


def build_maps(
    match_path: Path,
    phases_path: Path,
    tracking_path: Path,
    per_label: int = 3,
) -> dict[str, Any]:
    match = json.loads(match_path.read_text())
    player_teams = {
        int(player["id"]): int(player["team_id"])
        for player in match["players"]
        if player.get("team_id") is not None
    }
    candidates = [
        row
        for row in read_phases(phases_path)
        if row["team_out_of_possession_phase_type"] in BLOCK_LABELS
    ]
    candidates.sort(key=lambda row: float(row["duration"]), reverse=True)
    for row in candidates:
        row["_sample_frame"] = str(
            (int(row["frame_start"]) + int(row["frame_end"])) // 2
        )
    frames = load_target_frames(
        tracking_path, {int(row["_sample_frame"]) for row in candidates}
    )

    maps_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phase in candidates:
        label = phase["team_out_of_possession_phase_type"]
        if len(maps_by_label[label]) >= per_label:
            continue
        frame_id = int(phase["_sample_frame"])
        frame = frames.get(frame_id)
        if not frame or frame.get("period") is None:
            continue
        possession_team_id = int(phase["team_in_possession_id"])
        players = []
        for player in frame.get("player_data", []):
            player_id = int(player["player_id"])
            team_id = player_teams.get(player_id)
            if team_id is None:
                continue
            players.append(
                {
                    "player_id": player_id,
                    "xy": canonical_xy(
                        float(player["x"]),
                        float(player["y"]),
                        phase["attacking_side"],
                    ),
                    "side": (
                        "possession_team"
                        if team_id == possession_team_id
                        else "out_of_possession_team"
                    ),
                    "detected": bool(player.get("is_detected")),
                }
            )
        ball_data = frame.get("ball_data", {})
        ball_xy = (
            canonical_xy(
                float(ball_data["x"]),
                float(ball_data["y"]),
                phase["attacking_side"],
            )
            if ball_data.get("x") is not None and ball_data.get("y") is not None
            else None
        )
        if (
            len(players) < 20
            or ball_xy is None
            or not bool(ball_data.get("is_detected"))
        ):
            continue
        maps_by_label[label].append(
            {
                "match_id": int(match["id"]),
                "frame": frame_id,
                "timestamp": frame["timestamp"],
                "period": int(frame["period"]),
                "duration": float(phase["duration"]),
                "attacking_side_source": phase["attacking_side"],
                "possession_team": phase["team_in_possession_shortname"],
                "out_of_possession_phase": label,
                "in_possession_phase": phase["team_in_possession_phase_type"],
                "players": players,
                "ball_xy": ball_xy,
                "ball_detected": bool(ball_data.get("is_detected")),
                "team_in_possession_width": float(
                    phase["team_in_possession_width_start"]
                ),
                "team_in_possession_length": float(
                    phase["team_in_possession_length_start"]
                ),
                "team_out_of_possession_width": float(
                    phase["team_out_of_possession_width_start"]
                ),
                "team_out_of_possession_length": float(
                    phase["team_out_of_possession_length_start"]
                ),
            }
        )

    maps = [
        row
        for label in BLOCK_LABELS
        for row in maps_by_label.get(label, [])
    ]
    return {
        "schema_version": 1,
        "source": "SkillCorner Open Data",
        "source_url": "https://github.com/SkillCorner/opendata",
        "label_status": "source",
        "coordinate_note": (
            "Possession team is normalized to attack left-to-right. "
            "Player and ball coordinates come from SkillCorner tracking."
        ),
        "match": {
            "id": int(match["id"]),
            "home_team": match["home_team"]["name"],
            "away_team": match["away_team"]["name"],
        },
        "maps": maps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=3)
    args = parser.parse_args()
    result = build_maps(
        args.match,
        args.phases,
        args.tracking,
        per_label=args.per_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    counts = {
        label: sum(
            row["out_of_possession_phase"] == label for row in result["maps"]
        )
        for label in BLOCK_LABELS
    }
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
