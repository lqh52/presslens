#!/usr/bin/env python3
"""Convert reconstructed review clips to the synthetic classifier graph schema."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def manual_labels(directory: Path) -> dict[tuple[str, int], dict]:
    path = directory / "track-labels.json"
    if not path.exists():
        return {}
    output = {}
    for row in json.loads(path.read_text()).get("labels", {}).values():
        raw = row["label"]
        output[(row["clip_id"], int(row["track_id"]))] = {
            "label": (
                "team_a"
                if raw.startswith("team_a")
                else "team_b"
                if raw.startswith("team_b")
                else "other"
            ),
            "goalkeeper": raw.endswith("_goalkeeper"),
        }
    return output


def fixture_directions(directory: Path) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    manual = manual_labels(directory)
    positions: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    goalkeepers: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for identity_path in sorted((directory / "identities").glob("*.json")):
        if identity_path.name == "match-anchors.json":
            continue
        identity = json.loads(identity_path.read_text())
        clip_id, fixture = identity["clip_id"], identity["fixture_id"]
        teams = {}
        keepers = set()
        for row in identity["tracks"]:
            track_id = int(row["track_id"])
            resolved = manual.get((clip_id, track_id), row)
            if resolved.get("label") in {"team_a", "team_b"}:
                teams[track_id] = resolved["label"]
                if resolved.get("goalkeeper"):
                    keepers.add(track_id)
        projection_path = directory / "pitch-projections" / f"{clip_id}.json"
        if not projection_path.exists():
            continue
        projection = json.loads(projection_path.read_text())
        for frame in projection["frames"]:
            for item in frame.get("objects", []):
                team = teams.get(int(item["track_id"]))
                if team:
                    positions[fixture][team].append(float(item["x"]))
                    if int(item["track_id"]) in keepers:
                        goalkeepers[fixture][team].append(float(item["x"]))
    directions, confidence = {}, {}
    for fixture, teams in positions.items():
        if not teams["team_a"] or not teams["team_b"]:
            continue
        keeper_anchors = {
            team: float(np.median(values))
            for team, values in goalkeepers[fixture].items()
            if values
        }
        if keeper_anchors:
            anchored_team = max(
                keeper_anchors, key=lambda team: abs(keeper_anchors[team])
            )
            own_goal_x = keeper_anchors[anchored_team]
            anchored_direction = -1 if own_goal_x > 0 else 1
            directions[fixture] = {
                anchored_team: anchored_direction,
                ("team_b" if anchored_team == "team_a" else "team_a"): -anchored_direction,
            }
            confidence[fixture] = min(1.0, 0.65 + abs(own_goal_x) / 80.0)
            continue
        medians = {
            team: float(np.median(values)) for team, values in teams.items()
        }
        lower = min(medians, key=medians.get)
        directions[fixture] = {
            lower: 1,
            ("team_b" if lower == "team_a" else "team_a"): -1,
        }
        confidence[fixture] = min(
            1.0, abs(medians["team_a"] - medians["team_b"]) / 12.0
        )
    return directions, confidence


def canonical(point: tuple[float, float], direction: int) -> np.ndarray:
    return np.asarray(
        [
            (direction * point[0] + PITCH_LENGTH / 2) / PITCH_LENGTH,
            (direction * point[1] + PITCH_WIDTH / 2) / PITCH_WIDTH,
        ],
        dtype=np.float32,
    )


def convert(directory: Path, stride: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    directions, direction_confidence = fixture_directions(directory)
    manual = manual_labels(directory)
    all_features, all_masks, metadata = [], [], []
    for projection_path in sorted((directory / "pitch-projections").glob("*.json")):
        clip_id = projection_path.stem
        identity = json.loads((directory / "identities" / f"{clip_id}.json").read_text())
        fixture = identity["fixture_id"]
        if fixture not in directions:
            continue
        identities = {}
        for row in identity["tracks"]:
            track_id = int(row["track_id"])
            resolved = {**row, **manual.get((clip_id, track_id), {})}
            if resolved.get("label") in {"team_a", "team_b"}:
                identities[track_id] = resolved
        projection = json.loads(projection_path.read_text())
        balls = {
            int(row["frame"]): row.get("ball")
            for row in json.loads(
                (directory / "ball-tracking" / f"{clip_id}.json").read_text()
            )["frames"]
        }
        previous: dict[int, tuple[int, tuple[float, float]]] = {}
        for frame in projection["frames"]:
            frame_index = int(frame["frame"])
            if frame_index % stride:
                continue
            ball = balls.get(frame_index)
            if not ball or ball.get("pitch_xy") is None:
                continue
            ball_xy = tuple(map(float, ball["pitch_xy"]))
            players = [
                (
                    int(item["track_id"]),
                    identities[int(item["track_id"])],
                    (float(item["x"]), float(item["y"])),
                )
                for item in frame.get("objects", [])
                if int(item["track_id"]) in identities
            ]
            if len(players) < 6:
                continue
            nearest = sorted(
                (
                    (float(np.linalg.norm(np.asarray(point) - ball_xy)), track, row, point)
                    for track, row, point in players
                ),
                key=lambda item: item[0],
            )
            holder_distance, holder_track, holder, _ = nearest[0]
            possession_team = holder["label"]
            opposing_distance = next(
                (item[0] for item in nearest if item[2]["label"] != possession_team),
                holder_distance,
            )
            possession_confidence = max(
                0.0,
                min(1.0, 0.75 - holder_distance / 9 + (opposing_distance - holder_distance) / 10),
            )
            direction = directions[fixture][possession_team]
            selected = sorted(
                players,
                key=lambda item: (
                    0 if item[1]["label"] == possession_team else 1,
                    item[0],
                ),
            )[:22]
            features = np.zeros((23, 13), dtype=np.float32)
            mask = np.zeros(23, dtype=bool)
            for node, (track_id, row, point) in enumerate(selected):
                features[node, :2] = canonical(point, direction)
                old = previous.get(track_id)
                if old and frame_index > old[0]:
                    seconds = (frame_index - old[0]) / 25.0
                    velocity = (
                        (np.asarray(point) - np.asarray(old[1])) * direction / seconds
                    )
                    features[node, 2:4] = np.clip(
                        velocity / [PITCH_LENGTH, PITCH_WIDTH], -0.2, 0.2
                    )
                previous[track_id] = (frame_index, point)
                features[node, 4 if row["label"] == possession_team else 5] = 1
                features[node, 7 + (0 if row.get("goalkeeper") else 1)] = 1
                if track_id == holder_track and holder_distance <= 5:
                    features[node, 12] = 1
                mask[node] = True
            ball_node = min(len(selected), 22)
            features[ball_node, :2] = canonical(ball_xy, direction)
            features[ball_node, 6] = 1
            features[ball_node, 11] = 1
            mask[ball_node] = True
            all_features.append(features)
            all_masks.append(mask)
            metadata.append(
                {
                    "clip_id": clip_id,
                    "fixture_id": fixture,
                    "frame": frame_index,
                    "possession_team": possession_team,
                    "possession_confidence": round(possession_confidence, 4),
                    "possession_confident": possession_confidence >= 0.45
                    and holder_distance <= 5,
                    "ball_holder_distance_m": round(holder_distance, 3),
                    "attacking_direction_raw": direction,
                    "direction_confidence": round(direction_confidence[fixture], 4),
                    "direction_confident": direction_confidence[fixture] >= 0.35,
                    "visible_nodes": int(mask.sum()),
                }
            )
    return np.stack(all_features), np.stack(all_masks), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    features, masks, metadata = convert(args.directory, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, masks=masks)
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata)
    )
    print(
        f"Wrote {len(features)} real-video graphs from "
        f"{len(set(row['clip_id'] for row in metadata))} clips"
    )


if __name__ == "__main__":
    main()
