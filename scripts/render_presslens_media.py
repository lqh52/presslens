#!/usr/bin/env python3
"""Render the eight current PressLens broadcast and canonical-map videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


TEAM_COLORS = {"team_a": (255, 167, 89), "team_b": (104, 104, 255)}
OTHER_COLOR = (70, 215, 245)
BALL_COLOR = (70, 235, 255)
PITCH_LINE = (225, 238, 229)
PRESSURE_COLOR = (60, 205, 235)
OUTPUT_NAMES = {
    "lei-ars-20150926-h1-0093-published": (
        "video-lei-ars-20150926-h1-0093-annotated.mp4",
        "canonical-lei-ars-20150926-h1-0093.mp4",
    ),
    "hul-ars-20160917-h1-0009": (
        "video-hul-ars-20160917-h1-0009-annotated.mp4",
        "canonical-hul-ars-20160917-h1-0009.mp4",
    ),
    "ars-che-20160924-h2-0058": (
        "video-ars-che-20160924-h2-0058-annotated.mp4",
        "canonical-ars-che-20160924-h2-0058.mp4",
    ),
    "ars-che-20160924-h2-0067": (
        "video-ars-che-20160924-h2-0067-annotated.mp4",
        "canonical-ars-che-20160924-h2-0067.mp4",
    ),
    "bur-ars-20150411-h1-0128": (
        "video-h1-128-annotated.mp4",
        "canonical-h1-128.mp4",
    ),
    "bur-ars-20150411-h1-0203": (
        "video-h1-203-annotated.mp4",
        "canonical-h1-203.mp4",
    ),
    "bur-ars-20150411-h1-0833": (
        "video-h1-833-annotated.mp4",
        "canonical-h1-833.mp4",
    ),
    "bur-ars-20150411-h1-1673": (
        "video-h1-1673-annotated.mp4",
        "canonical-h1-1673.mp4",
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_label(label: str) -> tuple[str, bool]:
    goalkeeper = label.endswith("_goalkeeper")
    if goalkeeper:
        label = label.removesuffix("_goalkeeper")
    return label, goalkeeper


def identity_map(base: Path, clip_id: str, manual: dict) -> dict[int, dict]:
    predicted = read_json(base / "identities" / f"{clip_id}.json")
    output = {
        int(row["track_id"]): {
            "label": row["label"],
            "goalkeeper": bool(row.get("goalkeeper")),
        }
        for row in predicted["tracks"]
    }
    for key, row in manual.items():
        if row["clip_id"] != clip_id:
            continue
        label, goalkeeper = normalize_label(row["label"])
        output[int(row["track_id"])] = {
            "label": label,
            "goalkeeper": goalkeeper,
        }
    return output


def mst_edges(nodes: list[dict]) -> list[tuple[int, int]]:
    if len(nodes) < 2:
        return []
    connected, remaining, edges = [0], set(range(1, len(nodes))), []
    while remaining:
        _, left, right = min(
            (
                (math.dist(nodes[a]["pitch"], nodes[b]["pitch"]), a, b)
                for a in connected
                for b in remaining
            ),
            key=lambda item: item[0],
        )
        edges.append((left, right))
        connected.append(right)
        remaining.remove(right)
    return edges


def graph_edges(nodes: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    same_team: set[tuple[int, int]] = set()
    for team in TEAM_COLORS:
        members = [index for index, node in enumerate(nodes) if node["team"] == team]
        local = [nodes[index] for index in members]
        for left, right in mst_edges(local):
            same_team.add(tuple(sorted((members[left], members[right]))))
        for left in members:
            nearest = sorted(
                (
                    (math.dist(nodes[left]["pitch"], nodes[right]["pitch"]), right)
                    for right in members
                    if right != left
                )
            )[:2]
            for distance, right in nearest:
                if distance <= 22:
                    same_team.add(tuple(sorted((left, right))))
    pressure = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if (
                nodes[left]["team"] != nodes[right]["team"]
                and math.dist(nodes[left]["pitch"], nodes[right]["pitch"]) <= 12
            ):
                pressure.append((left, right))
    return sorted(same_team), pressure


def pitch_background() -> np.ndarray:
    width, height, pad = 1050, 590, 4
    image = np.full((height, width, 3), (52, 107, 40), dtype=np.uint8)
    cv2.rectangle(image, (pad, pad), (width - pad, height - pad), PITCH_LINE, 3)
    cv2.line(image, (width // 2, pad), (width // 2, height - pad), PITCH_LINE, 3)
    cv2.circle(image, (width // 2, height // 2), 79, PITCH_LINE, 3)
    penalty_width = int(40.32 / 68 * (height - 2 * pad))
    penalty_depth = int(16.5 / 105 * (width - 2 * pad))
    goal_width = int(18.32 / 68 * (height - 2 * pad))
    goal_depth = int(5.5 / 105 * (width - 2 * pad))
    cy = height // 2
    for side in (0, 1):
        x = pad if side == 0 else width - pad
        direction = 1 if side == 0 else -1
        cv2.rectangle(
            image,
            (min(x, x + direction * penalty_depth), cy - penalty_width // 2),
            (max(x, x + direction * penalty_depth), cy + penalty_width // 2),
            PITCH_LINE,
            3,
        )
        cv2.rectangle(
            image,
            (min(x, x + direction * goal_depth), cy - goal_width // 2),
            (max(x, x + direction * goal_depth), cy + goal_width // 2),
            PITCH_LINE,
            3,
        )
    return image


def to_canvas(x: float, y: float) -> tuple[int, int]:
    width, height, pad = 1050, 590, 4
    return (
        int(pad + (x + 52.5) / 105 * (width - 2 * pad)),
        int(pad + (y + 34) / 68 * (height - 2 * pad)),
    )


def frame_nodes(
    projected: list[dict],
    accepted_tracks: set[int],
    identities: dict[int, dict],
    points: dict[int, tuple[int, int]],
) -> list[dict]:
    nodes = []
    for item in projected:
        track = int(item["track_id"])
        identity = identities.get(track)
        if (
            track not in accepted_tracks
            or not identity
            or identity["label"] not in TEAM_COLORS
            or abs(float(item["x"])) > 52.5
            or abs(float(item["y"])) > 34
        ):
            continue
        nodes.append(
            {
                "track": track,
                "team": identity["label"],
                "goalkeeper": identity["goalkeeper"],
                "pitch": (float(item["x"]), float(item["y"])),
                "estimated": bool(item.get("estimated")),
                "broadcast": points[track],
                "canonical": to_canvas(float(item["x"]), float(item["y"])),
            }
        )
    return nodes


def draw_edges(image: np.ndarray, nodes: list[dict], point_key: str) -> None:
    same_team, pressure = graph_edges(nodes)
    for left, right in same_team:
        cv2.line(
            image,
            nodes[left][point_key],
            nodes[right][point_key],
            TEAM_COLORS[nodes[left]["team"]],
            2,
            cv2.LINE_AA,
        )
    for left, right in pressure:
        cv2.line(
            image,
            nodes[left][point_key],
            nodes[right][point_key],
            PRESSURE_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_broadcast(
    image: np.ndarray,
    detections: list[dict],
    projected: list[dict],
    identities: dict[int, dict],
    ball: dict | None,
) -> np.ndarray:
    accepted, points = set(), {}
    visible = []
    for detection in detections:
        if detection.get("track_id") is None or float(detection["confidence"]) < 0.45:
            continue
        track = int(detection["track_id"])
        identity = identities.get(track, {"label": "other", "goalkeeper": False})
        box = [int(round(value)) for value in detection["bbox"]]
        label = identity["label"]
        color = TEAM_COLORS.get(label, OTHER_COLOR)
        accepted.add(track)
        points[track] = ((box[0] + box[2]) // 2, box[3])
        visible.append((track, identity, box, color))
    nodes = frame_nodes(projected, accepted, identities, points)
    draw_edges(image, nodes, "broadcast")
    for track, identity, box, color in visible:
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        suffix = " GK" if identity["goalkeeper"] else ""
        cv2.putText(
            image,
            f"#{track}{suffix}",
            (box[0], max(18, box[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    if ball:
        point = tuple(int(round(value)) for value in ball["image_xy"])
        cv2.circle(image, point, 10, BALL_COLOR, 2)
        cv2.drawMarker(image, point, BALL_COLOR, cv2.MARKER_CROSS, 22, 2)
    return image


def draw_canonical(
    detections: list[dict],
    projected: list[dict],
    identities: dict[int, dict],
    ball: dict | None,
) -> np.ndarray:
    accepted = {
        int(row["track_id"])
        for row in detections
        if row.get("track_id") is not None and float(row["confidence"]) >= 0.45
    }
    points = {int(row["track_id"]): (0, 0) for row in projected}
    nodes = frame_nodes(projected, accepted, identities, points)
    image = pitch_background()
    draw_edges(image, nodes, "canonical")
    for node in nodes:
        color = TEAM_COLORS[node["team"]]
        thickness = 3 if node["estimated"] else -1
        cv2.circle(image, node["canonical"], 10 if node["goalkeeper"] else 8, color, thickness)
        cv2.circle(image, node["canonical"], 10 if node["goalkeeper"] else 8, PITCH_LINE, 2)
        cv2.putText(
            image,
            str(node["track"]),
            (node["canonical"][0] + 10, node["canonical"][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            PITCH_LINE,
            1,
        )
    if ball and ball.get("pitch_xy"):
        point = to_canvas(*ball["pitch_xy"])
        cv2.circle(image, point, 7, BALL_COLOR, -1)
        cv2.circle(image, point, 9, (20, 20, 20), 2)
    return image


def writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not result.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return result


def transcode(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source), "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_clip(
    base: Path,
    output: Path,
    clip_id: str,
    manual: dict,
    output_names: tuple[str, str] | None = None,
) -> None:
    detections = read_json(
        base / "results" / "yolo26m-botsort-high-recall" / f"{clip_id}.json"
    )
    projections = read_json(base / "pitch-projections" / f"{clip_id}.json")
    balls = read_json(base / "ball-tracking" / f"{clip_id}.json")
    identities = identity_map(base, clip_id, manual)
    detection_frames = {int(row["frame"]): row["detections"] for row in detections["frames"]}
    projection_frames = {int(row["frame"]): row["objects"] for row in projections["frames"]}
    ball_frames = {int(row["frame"]): row.get("ball") for row in balls["frames"]}
    capture = cv2.VideoCapture(str(detections["clip_path"]))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    broadcast_name, canonical_name = output_names or OUTPUT_NAMES[clip_id]
    with tempfile.TemporaryDirectory(prefix="presslens-render-") as temporary:
        temporary = Path(temporary)
        broadcast_raw, canonical_raw = temporary / "broadcast.mp4", temporary / "canonical.mp4"
        broadcast_writer = writer(broadcast_raw, fps, (width, height))
        canonical_writer = writer(canonical_raw, fps, (1050, 590))
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_detections = detection_frames.get(frame_index, [])
            frame_projection = projection_frames.get(frame_index, [])
            frame_ball = ball_frames.get(frame_index)
            broadcast_writer.write(
                draw_broadcast(
                    image, frame_detections, frame_projection, identities, frame_ball
                )
            )
            canonical_writer.write(
                draw_canonical(
                    frame_detections, frame_projection, identities, frame_ball
                )
            )
            frame_index += 1
        capture.release()
        broadcast_writer.release()
        canonical_writer.release()
        transcode(broadcast_raw, output / broadcast_name)
        transcode(canonical_raw, output / canonical_name)
    print(f"{clip_id}: {frame_index} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/published-tracking-review"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.directory / "track-labels.json"
    manual = read_json(labels_path)["labels"] if labels_path.exists() else {}
    for clip_id in OUTPUT_NAMES:
        render_clip(args.directory, args.output_dir, clip_id, manual)


if __name__ == "__main__":
    main()
