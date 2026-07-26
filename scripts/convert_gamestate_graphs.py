#!/usr/bin/env python3
"""Convert SoccerNet-GSR v1.3 labels into canonical pitch-space graphs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
ROLE_INDEX = {"goalkeeper": 0, "player": 1, "ball": 4}


def pitch_point(annotation: dict) -> np.ndarray | None:
    pitch = annotation.get("bbox_pitch") or {}
    x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
    if x is None or y is None or not np.isfinite([x, y]).all():
        return None
    # Retain a small tolerance for calibration noise at the boundary.
    if abs(x) > PITCH_LENGTH / 2 + 2 or abs(y) > PITCH_WIDTH / 2 + 2:
        return None
    return np.array([x, y], dtype=np.float32)


def sequence_directions(annotations: list[dict]) -> dict[str, int]:
    """Return +1 for a team attacking +pitch-x, -1 for the opposite direction."""
    goalkeeper_x: dict[str, list[float]] = defaultdict(list)
    all_x: dict[str, list[float]] = defaultdict(list)
    for annotation in annotations:
        attributes = annotation.get("attributes") or {}
        team = attributes.get("team")
        point = pitch_point(annotation)
        if team not in ("left", "right") or point is None:
            continue
        all_x[team].append(float(point[0]))
        if attributes.get("role") == "goalkeeper":
            goalkeeper_x[team].append(float(point[0]))
    anchors = {}
    for team in ("left", "right"):
        values = goalkeeper_x[team] or all_x[team]
        if values:
            anchors[team] = float(np.median(values))
    if len(anchors) != 2:
        return {"left": 1, "right": -1}
    low_team = min(anchors, key=anchors.get)
    return {low_team: 1, ("right" if low_team == "left" else "left"): -1}


def canonicalize(point: np.ndarray, direction: int) -> np.ndarray:
    rotated = point * direction
    return np.array(
        [(rotated[0] + PITCH_LENGTH / 2) / PITCH_LENGTH, (rotated[1] + PITCH_WIDTH / 2) / PITCH_WIDTH],
        dtype=np.float32,
    )


def convert_sequence(path: Path, stride: int) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    payload = json.loads(path.read_text())
    if payload["info"].get("version") != "1.3":
        raise ValueError(f"{path}: expected label version 1.3")
    by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation.get("category_id") in (1, 2, 4):
            by_image[str(annotation["image_id"])].append(annotation)
    directions = sequence_directions(payload["annotations"])
    frames, masks, metadata = [], [], []
    previous: dict[int, tuple[int, np.ndarray]] = {}
    fps = float(payload["info"]["frame_rate"])

    for frame_index, image in enumerate(payload["images"]):
        if frame_index % stride:
            continue
        objects = by_image.get(str(image["image_id"]), [])
        ball_annotations = [a for a in objects if a["category_id"] == 4 and pitch_point(a) is not None]
        players = [
            a
            for a in objects
            if a["category_id"] in (1, 2)
            and (a.get("attributes") or {}).get("team") in ("left", "right")
            and pitch_point(a) is not None
        ]
        if not ball_annotations or len(players) < 6:
            continue
        ball_raw = pitch_point(ball_annotations[0])
        assert ball_raw is not None
        distances = np.array([np.linalg.norm(pitch_point(a) - ball_raw) for a in players])
        holder_index = int(distances.argmin())
        holder_distance = float(distances[holder_index])
        holder_track_id = int(players[holder_index]["track_id"])
        possession_team = (players[holder_index].get("attributes") or {})["team"]
        pressing_team = "right" if possession_team == "left" else "left"
        direction = directions[possession_team]
        players.sort(
            key=lambda a: (
                0 if (a.get("attributes") or {}).get("team") == possession_team else 1,
                int(a["track_id"]),
            )
        )
        selected = players[:22] + ball_annotations[:1]
        features = np.zeros((23, 13), dtype=np.float32)
        mask = np.zeros(23, dtype=bool)
        for node_index, annotation in enumerate(selected):
            point = pitch_point(annotation)
            assert point is not None
            features[node_index, :2] = canonicalize(point, direction)
            track_id = int(annotation["track_id"])
            old = previous.get(track_id)
            if old is not None and frame_index > old[0]:
                velocity = (point - old[1]) / ((frame_index - old[0]) / fps)
                velocity *= direction
                features[node_index, 2:4] = np.clip(
                    velocity / [PITCH_LENGTH, PITCH_WIDTH], -0.2, 0.2
                )
            previous[track_id] = (frame_index, point)
            attributes = annotation.get("attributes") or {}
            if annotation["category_id"] == 4:
                features[node_index, 6] = 1  # neutral/ball team
                role = "ball"
            else:
                team = attributes["team"]
                features[node_index, 4 if team == possession_team else 5] = 1
                role = attributes.get("role", "player")
                if track_id == holder_track_id and holder_distance <= 3.0:
                    features[node_index, 12] = 1
            features[node_index, 7 + ROLE_INDEX.get(role, 1)] = 1
            mask[node_index] = True
        frames.append(features)
        masks.append(mask)
        metadata.append(
            {
                "sequence": payload["info"]["name"],
                "frame": int(Path(image["file_name"]).stem),
                "time_seconds": frame_index / fps,
                "possession_team": possession_team,
                "attacking_direction_raw": direction,
                "ball_holder_distance_m": round(holder_distance, 3),
                "possession_confident": holder_distance <= 3.0,
                "visible_nodes": int(mask.sum()),
            }
        )
    return frames, masks, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5, help="Keep every Nth 25-fps frame")
    args = parser.parse_args()
    all_frames, all_masks, all_metadata = [], [], []
    paths = sorted(args.labels_dir.glob("*/Labels-GameState.json"))
    for index, path in enumerate(paths, 1):
        frames, masks, metadata = convert_sequence(path, args.stride)
        all_frames.extend(frames)
        all_masks.extend(masks)
        all_metadata.extend(metadata)
        print(f"[{index:03d}/{len(paths):03d}] {path.parent.name}: {len(frames)} graphs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(all_frames),
        masks=np.stack(all_masks),
    )
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_metadata)
    )
    print(f"Wrote {len(all_frames)} graphs to {args.output}")


if __name__ == "__main__":
    main()
