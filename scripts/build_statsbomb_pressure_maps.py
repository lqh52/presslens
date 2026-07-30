#!/usr/bin/env python3
"""Build canonical weak pressing maps from StatsBomb Open Data events and 360."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def fetch_json(relative_path: str) -> Any:
    with urllib.request.urlopen(f"{BASE_URL}/{relative_path}") as response:
        return json.load(response)


def point(location: list[float]) -> list[float]:
    return [
        round(float(location[0]) * 105.0 / 120.0, 4),
        round(float(location[1]) * 68.0 / 80.0, 4),
    ]


def build_match(match_id: int) -> dict[str, Any]:
    events = fetch_json(f"events/{match_id}.json")
    frames = {
        row["event_uuid"]: row
        for row in fetch_json(f"three-sixty/{match_id}.json")
    }
    maps = []
    for event in events:
        if event.get("type", {}).get("name") != "Pressure":
            continue
        frame = frames.get(event["id"])
        if not frame or not event.get("location"):
            continue
        raw_x = float(event["location"][0])
        counterpress = bool(event.get("counterpress", False))
        high_press_candidate = raw_x >= 80.0
        if counterpress:
            weak_label = "counterpress"
        elif high_press_candidate:
            weak_label = "high_press_candidate"
        else:
            weak_label = "individual_pressure"
        players = []
        for player in frame.get("freeze_frame", []):
            players.append(
                {
                    "xy": point(player["location"]),
                    "side": (
                        "pressing_team"
                        if player.get("teammate")
                        else "possession_team"
                    ),
                    "actor": bool(player.get("actor")),
                    "keeper": bool(player.get("keeper")),
                }
            )
        event_xy = point(event["location"])
        possession_players = [
            player for player in players if player["side"] == "possession_team"
        ]
        ball_proxy = (
            min(
                possession_players,
                key=lambda player: math.dist(player["xy"], event_xy),
            )["xy"]
            if possession_players
            else None
        )
        visible = frame.get("visible_area", [])
        visible_polygon = [
            point(visible[index : index + 2])
            for index in range(0, len(visible), 2)
        ]
        maps.append(
            {
                "event_id": event["id"],
                "match_id": match_id,
                "period": int(event["period"]),
                "timestamp": event["timestamp"],
                "minute": int(event["minute"]),
                "second": int(event["second"]),
                "pressing_team": event["team"]["name"],
                "possession_team": event.get("possession_team", {}).get("name"),
                "event_xy": event_xy,
                "ball_xy": ball_proxy,
                "ball_method": (
                    "nearest_possession_player_to_pressure_actor"
                    if ball_proxy
                    else "unavailable"
                ),
                "weak_label": weak_label,
                "counterpress": counterpress,
                "high_press_candidate": high_press_candidate,
                "players": players,
                "visible_polygon": visible_polygon,
            }
        )
    return {
        "schema_version": 1,
        "source": "StatsBomb Open Data",
        "source_url": "https://github.com/hudl/open-data",
        "source_coordinate_system": "120x80",
        "canonical_pitch_m": [105, 68],
        "label_status": "weak",
        "label_note": (
            "Pressure is a StatsBomb event. counterpress is a source event flag. "
            "high_press_candidate is derived solely from event x >= 80/120 and "
            "does not establish a coordinated team press."
        ),
        "match_id": match_id,
        "maps": maps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_match(args.match_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    counts: dict[str, int] = {}
    for row in result["maps"]:
        counts[row["weak_label"]] = counts.get(row["weak_label"], 0) + 1
    print(
        json.dumps(
            {
                "match_id": args.match_id,
                "maps": len(result["maps"]),
                "labels": counts,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
