#!/usr/bin/env python3
"""Export broadcast-visible SkillCorner pressing sequences for weak supervision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.build_skillcorner_phase_maps import oriented_xy
    from scripts.render_presslens_media import (
        BALL_COLOR,
        PITCH_LINE,
        TEAM_COLORS,
        graph_edges,
        pitch_background,
        to_canvas,
    )
except ModuleNotFoundError:
    from build_skillcorner_phase_maps import oriented_xy
    from render_presslens_media import (
        BALL_COLOR,
        PITCH_LINE,
        TEAM_COLORS,
        graph_edges,
        pitch_background,
        to_canvas,
    )

WIDE_CHANNELS = {"wide_left", "wide_right"}


def truth(value: str | bool | None) -> bool:
    return str(value).lower() == "true"


def opposite_attacking_side(attacking_side: str) -> str:
    """Convert the pressing actor's direction to the possession direction."""
    if attacking_side == "left_to_right":
        return "right_to_left"
    if attacking_side == "right_to_left":
        return "left_to_right"
    raise ValueError(f"Unsupported attacking side: {attacking_side}")


def sample_frame_ids(start: int, end: int, count: int = 5) -> list[int]:
    if count == 1:
        return [(start + end) // 2]
    extended_start = max(0, start - 5)
    extended_end = end + 5
    return [
        round(extended_start + index * (extended_end - extended_start) / (count - 1))
        for index in range(count)
    ]


def load_frames(path: Path, targets: set[int]) -> dict[int, dict[str, Any]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            frame_id = int(row["frame"])
            if frame_id in targets:
                result[frame_id] = row
                if len(result) == len(targets):
                    break
    return result


def canonical_polygon(frame: dict[str, Any], attacking_side: str) -> list[list[float]]:
    corners = frame.get("image_corners_projection", {})
    result = []
    for name in ("top_left", "bottom_left", "bottom_right", "top_right"):
        x, y = corners.get(f"x_{name}"), corners.get(f"y_{name}")
        if x is not None and y is not None:
            result.append(oriented_xy(float(x), float(y), attacking_side))
    return result


def frame_payload(
    frame: dict[str, Any],
    attacking_side: str,
    pressing_team_id: int,
    player_teams: dict[int, int],
    goalkeepers: set[int],
) -> dict[str, Any] | None:
    ball = frame.get("ball_data", {})
    if (
        not truth(ball.get("is_detected"))
        or ball.get("x") is None
        or ball.get("y") is None
    ):
        return None
    players = []
    for player in frame.get("player_data", []):
        if not truth(player.get("is_detected")):
            continue
        player_id = int(player["player_id"])
        team_id = player_teams.get(player_id)
        if team_id is None:
            continue
        players.append(
            {
                "track_id": player_id,
                "player_id": player_id,
                "side": (
                    "pressing_team"
                    if team_id == pressing_team_id
                    else "possession_team"
                ),
                "team": (
                    "team_b"
                    if team_id == pressing_team_id
                    else "team_a"
                ),
                "goalkeeper": player_id in goalkeepers,
                "xy": oriented_xy(
                    float(player["x"]), float(player["y"]), attacking_side
                ),
                "estimated": False,
                "projection_method": "observed",
            }
        )
    if len(players) < 8:
        return None
    return {
        "frame": int(frame["frame"]),
        "timestamp": frame["timestamp"],
        "players": players,
        "ball_xy": oriented_xy(
            float(ball["x"]), float(ball["y"]), attacking_side
        ),
        "visible_polygon": canonical_polygon(frame, attacking_side),
    }


def classify_chains(dynamic_events_path: Path) -> dict[str, list[dict[str, Any]]]:
    with dynamic_events_path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["event_type"] == "on_ball_engagement"
            and row["event_subtype"] == "pressing"
            and truth(row["pressing_chain"])
        ]
    chains: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        chains[int(row["pressing_chain_index"])].append(row)

    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chain_id, chain in chains.items():
        high_rows = [
            row
            for row in chain
            if row["team_out_of_possession_phase_type"] == "high_block"
        ]
        medium_rows = [
            row
            for row in chain
            if row["team_out_of_possession_phase_type"] == "medium_block"
        ]
        wide_high_rows = [
            row
            for row in high_rows
            if row["channel_start"] in WIDE_CHANNELS
            and int(row["pressing_chain_length"] or 0) >= 2
        ]
        if wide_high_rows:
            category, relevant = "high_press_wing", wide_high_rows
        elif high_rows:
            category, relevant = "high_press_central", high_rows
        elif medium_rows:
            category, relevant = "medium_press", medium_rows
        else:
            continue
        representative = max(
            relevant,
            key=lambda row: (
                row["pressing_chain_end_type"] in {"regain", "disruption"},
                truth(row["force_backward"]),
                int(row["frame_end"]) - int(row["frame_start"]),
            ),
        )
        classes[category].append(
            {
                "chain_id": chain_id,
                "category": category,
                "representative": representative,
                "source_event": "pressing_chain",
                "pressing_team_id": int(representative["team_id"]),
                "possession_attacking_side": opposite_attacking_side(
                    representative["attacking_side"]
                ),
                "chain_length": int(representative["pressing_chain_length"]),
                "chain_outcome": representative["pressing_chain_end_type"] or None,
                "force_backward": truth(representative["force_backward"]),
            }
        )
    for values in classes.values():
        values.sort(
            key=lambda item: (
                item["chain_outcome"] in {"regain", "disruption"},
                item["force_backward"],
                item["chain_length"],
            ),
            reverse=True,
        )
    return classes


def classify_low_blocks(phases_path: Path) -> list[dict[str, Any]]:
    """Select settled source-labelled low-block phases as temporal examples."""
    with phases_path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["team_out_of_possession_phase_type"] == "low_block"
            and float(row["duration"]) >= 4.0
        ]
    rows.sort(key=lambda row: float(row["duration"]), reverse=True)
    return [
        {
            "chain_id": int(row["index"]),
            "category": "low_block",
            "representative": row,
            "source_event": "phase_of_play",
            "possession_team_id": int(row["team_in_possession_id"]),
            "possession_attacking_side": row["attacking_side"],
            "chain_length": 0,
            "chain_outcome": None,
            "force_backward": False,
        }
        for row in rows
    ]


def render_frame(frame: dict[str, Any], sample: dict[str, Any]) -> np.ndarray:
    image = pitch_background()
    nodes = [
        {
            "track": player["track_id"],
            "team": player["team"],
            "goalkeeper": player["goalkeeper"],
            "pitch": tuple(player["xy"]),
            "estimated": False,
            "canonical": to_canvas(*player["xy"]),
        }
        for player in frame["players"]
    ]
    same_team, pressure = graph_edges(nodes)
    for left, right in same_team:
        cv2.line(
            image,
            nodes[left]["canonical"],
            nodes[right]["canonical"],
            TEAM_COLORS[nodes[left]["team"]],
            2,
            cv2.LINE_AA,
        )
    for left, right in pressure:
        cv2.line(
            image,
            nodes[left]["canonical"],
            nodes[right]["canonical"],
            (60, 205, 235),
            2,
            cv2.LINE_AA,
        )
    for player in frame["players"]:
        centre = to_canvas(*player["xy"])
        colour = TEAM_COLORS[player["team"]]
        radius = 10 if player["goalkeeper"] else 8
        cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
        cv2.circle(image, centre, radius, PITCH_LINE, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(player["track_id"]),
            (centre[0] + 10, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            PITCH_LINE,
            1,
        )
    ball = to_canvas(*frame["ball_xy"])
    cv2.circle(image, ball, 7, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, ball, 7, (52, 47, 5), 2, cv2.LINE_AA)
    return image


def graph_frame(
    frame: dict[str, Any],
    previous: dict[int, tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    players = sorted(
        frame["players"],
        key=lambda player: (
            0 if player["team"] == "team_a" else 1,
            player["track_id"],
        ),
    )[:22]
    ball_xy = np.asarray(frame["ball_xy"], dtype=np.float32)
    nearest = sorted(
        (
            (
                float(np.linalg.norm(np.asarray(player["xy"]) - ball_xy)),
                player,
            )
            for player in players
        ),
        key=lambda item: item[0],
    )
    holder_distance, holder = nearest[0]
    features = np.zeros((23, 13), dtype=np.float32)
    mask = np.zeros(23, dtype=bool)
    for index, player in enumerate(players):
        point = np.asarray(player["xy"], dtype=np.float32)
        features[index, :2] = [
            (point[0] + 52.5) / 105.0,
            (point[1] + 34.0) / 68.0,
        ]
        track_id = int(player["track_id"])
        old = previous.get(track_id)
        if old and int(frame["frame"]) > old[0]:
            seconds = (int(frame["frame"]) - old[0]) / 10.0
            velocity = (point - old[1]) / seconds
            features[index, 2:4] = np.clip(
                velocity / [105.0, 68.0], -0.2, 0.2
            )
        previous[track_id] = (int(frame["frame"]), point)
        features[index, 4 if player["team"] == "team_a" else 5] = 1
        features[index, 7 if player["goalkeeper"] else 8] = 1
        if (
            player["track_id"] == holder["track_id"]
            and holder_distance <= 5.0
        ):
            features[index, 12] = 1
        mask[index] = True
    ball_index = len(players)
    features[ball_index, :2] = [
        (ball_xy[0] + 52.5) / 105.0,
        (ball_xy[1] + 34.0) / 68.0,
    ]
    features[ball_index, 6] = 1
    features[ball_index, 11] = 1
    mask[ball_index] = True
    metadata = {
        "frame": int(frame["frame"]),
        "possession_team": "team_a",
        "attacking_direction_raw": 1,
        "attacking_direction_label": "left_to_right",
        "direction_source": "skillcorner_attacking_side",
        "direction_status": "calibrated",
        "direction_confident": True,
        "direction_confidence": 1.0,
        "ball_holder_distance_m": round(holder_distance, 3),
        "possession_confident": holder_distance <= 5.0,
        "ball_detection_confidence": 1.0,
        "visible_nodes": int(mask.sum()),
        "source": "skillcorner_open_broadcast_tracking",
    }
    return features, mask, metadata


def preview_with_label(image: np.ndarray, sample: dict[str, Any]) -> np.ndarray:
    header = np.full((54, image.shape[1], 3), (13, 27, 20), np.uint8)
    label = sample["category"].replace("_", " ").upper()
    cv2.putText(
        header,
        label,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        BALL_COLOR,
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def build_dataset(
    match_path: Path,
    dynamic_events_path: Path,
    phases_path: Path,
    tracking_path: Path,
    output_manifest: Path,
    output_graphs: Path,
    output_dir: Path,
    per_class: int,
) -> dict[str, Any]:
    match = json.loads(match_path.read_text())
    player_teams = {
        int(player["id"]): int(player["team_id"])
        for player in match["players"]
        if player.get("team_id") is not None
    }
    goalkeepers = {
        int(player["id"])
        for player in match["players"]
        if player.get("player_role", {}).get("name") == "Goalkeeper"
    }
    candidates = classify_chains(dynamic_events_path)
    low_blocks = classify_low_blocks(phases_path)
    team_ids = set(player_teams.values())
    for candidate in low_blocks:
        opponents = team_ids - {candidate["possession_team_id"]}
        if len(opponents) != 1:
            continue
        candidate["pressing_team_id"] = opponents.pop()
        candidates["low_block"].append(candidate)
    all_candidates = [
        candidate
        for category in (
            "high_press_wing",
            "high_press_central",
            "medium_press",
            "low_block",
        )
        for candidate in candidates.get(category, [])
    ]
    for candidate in all_candidates:
        row = candidate["representative"]
        candidate["frame_ids"] = sample_frame_ids(
            int(row["frame_start"]), int(row["frame_end"])
        )
    raw_frames = load_frames(
        tracking_path,
        {
            frame_id
            for candidate in all_candidates
            for frame_id in candidate["frame_ids"]
        },
    )

    samples = []
    previews = []
    graph_features = []
    graph_masks = []
    graph_labels = []
    graph_sequence_indices = []
    graph_frame_indices = []
    graph_metadata = []
    class_names = (
        "medium_press",
        "high_press_central",
        "high_press_wing",
        "low_block",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for candidate in all_candidates:
        category = candidate["category"]
        if counts[category] >= per_class:
            continue
        row = candidate["representative"]
        possession_attacking_side = candidate["possession_attacking_side"]
        frames = []
        for frame_id in candidate["frame_ids"]:
            raw = raw_frames.get(frame_id)
            if raw is None:
                break
            payload = frame_payload(
                raw,
                possession_attacking_side,
                candidate["pressing_team_id"],
                player_teams,
                goalkeepers,
            )
            if payload is None:
                break
            frames.append(payload)
        if len(frames) != 5:
            continue
        sample_id = (
            f"{int(match['id'])}-{candidate['source_event']}-"
            f"{candidate['chain_id']:03d}-{category}"
        )
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample = {
            "sample_id": sample_id,
            "split": "train",
            "match_id": int(match["id"]),
            "chain_id": candidate["chain_id"],
            "category": category,
            "labels": {
                "high_press": category.startswith("high_press"),
                "wing_trap_candidate": category == "high_press_wing",
            },
            "source_phase": row["team_out_of_possession_phase_type"],
            "source_event": candidate["source_event"],
            "chain_length": candidate["chain_length"],
            "chain_outcome": candidate["chain_outcome"],
            "force_backward": candidate["force_backward"],
            "channel": row["channel_start"],
            "pressing_team": row.get("team_shortname"),
            "possession_player": row.get("player_in_possession_name"),
            "coordinate_system": {
                "units": "metres",
                "origin": "pitch_centre",
                "x_range": [-52.5, 52.5],
                "y_range": [-34.0, 34.0],
                "possession_attack": "left_to_right",
                "source_attacking_side_owner": (
                    "pressing_team"
                    if candidate["source_event"] == "pressing_chain"
                    else "possession_team"
                ),
                "source_attacking_side": row["attacking_side"],
            },
            "frames": frames,
        }
        previous: dict[int, tuple[int, np.ndarray]] = {}
        for index, frame in enumerate(frames):
            frame["calibration_frame"] = frame["frame"]
            frame["objects"] = [
                {
                    "track_id": player["track_id"],
                    "x": player["xy"][0],
                    "y": player["xy"][1],
                    "estimated": False,
                    "projection_method": "observed",
                }
                for player in frame["players"]
            ]
            frame["ball"] = {
                "confidence": 1.0,
                "method": "detected",
                "pitch_xy": frame["ball_xy"],
            }
            image = render_frame(frame, sample)
            image_path = sample_dir / f"frame-{index}.png"
            cv2.imwrite(str(image_path), image)
            if index == len(frames) // 2:
                previews.append(preview_with_label(image, sample))
            frame["image"] = str(image_path)
            features, mask, metadata = graph_frame(frame, previous)
            metadata.update(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "labels": sample["labels"],
                }
            )
            graph_features.append(features)
            graph_masks.append(mask)
            graph_labels.append(class_names.index(category))
            graph_sequence_indices.append(len(samples))
            graph_frame_indices.append(index)
            graph_metadata.append(metadata)
        samples.append(sample)
        counts[category] += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "source": "SkillCorner Open Data",
        "label_note": (
            "high_press combines source high_block and pressing_chain fields; "
            "wing_trap_candidate additionally requires a wide channel and chain "
            "length >= 2. medium_press is a hard negative for high_press. "
            "low_block uses settled source phase_of_play low_block intervals "
            "lasting at least four seconds."
        ),
        "visibility_note": (
            "Only directly detected broadcast-visible players and directly "
            "detected ball frames are retained; extrapolated players are excluded."
        ),
        "projection_schema": {
            "compatible_with": "scripts/project_tracking_to_pitch.py schema_version 1",
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
            "object_fields": [
                "track_id",
                "x",
                "y",
                "estimated",
                "projection_method",
            ],
            "ball_fields": ["confidence", "method", "pitch_xy"],
        },
        "graph_schema": {
            "compatible_with": "scripts/convert_review_video_graphs.py",
            "shape": [23, 13],
            "feature_order": [
                "x_normalized",
                "y_normalized",
                "vx_normalized",
                "vy_normalized",
                "possession_team",
                "pressing_team",
                "ball_team",
                "goalkeeper_role",
                "player_role",
                "referee_role",
                "other_role",
                "ball_role",
                "ball_control",
            ],
            "npz": str(output_graphs),
        },
        "samples": samples,
    }
    output_manifest.write_text(json.dumps(result, indent=2) + "\n")
    output_graphs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_graphs,
        features=np.stack(graph_features),
        masks=np.stack(graph_masks),
        labels=np.asarray(graph_labels, dtype=np.int64),
        label_names=np.asarray(class_names),
        sequence_index=np.asarray(graph_sequence_indices, dtype=np.int64),
        frame_index=np.asarray(graph_frame_indices, dtype=np.int64),
    )
    output_graphs.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in graph_metadata)
    )

    cells = [cv2.resize(image, (630, 456)) for image in previews]
    if cells:
        if len(cells) % 2:
            cells.append(np.full_like(cells[0], (13, 27, 20)))
        contact = np.vstack(
            [
                np.hstack(cells[index : index + 2])
                for index in range(0, len(cells), 2)
            ]
        )
        cv2.imwrite(str(output_dir / "contact-sheet.png"), contact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--dynamic-events", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-graphs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=5)
    args = parser.parse_args()
    result = build_dataset(
        args.match,
        args.dynamic_events,
        args.phases,
        args.tracking,
        args.output_manifest,
        args.output_graphs,
        args.output_dir,
        args.per_class,
    )
    counts: dict[str, int] = defaultdict(int)
    for sample in result["samples"]:
        counts[sample["category"]] += 1
    print(
        json.dumps(
            {
                "manifest": str(args.output_manifest),
                "output_dir": str(args.output_dir),
                "graphs": str(args.output_graphs),
                "counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
