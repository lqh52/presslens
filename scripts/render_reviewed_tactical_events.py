#!/usr/bin/env python3
"""Render fixed-label, low-FPS broadcast and canonical tactical event videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = {"team_a": (255, 167, 89), "team_b": (104, 104, 255)}
DISPLAY = {
    "unstructured": "Unstructured",
    "central_screen": "Central screen",
    "trap_left": "Trap left",
    "trap_right": "Trap right",
    "high_press": "High press",
}


def runs(frames: list[dict], minimum: int) -> list[tuple[str, int, int]]:
    output, label, start, last = [], None, None, None
    for row in frames:
        frame = int(row["frame"])
        current = (row.get("prediction") or {}).get("label")
        if current == "abstain":
            current = None
        if current == label and current is not None and frame == last + 1:
            last = frame
            continue
        if label is not None and last - start + 1 >= minimum:
            output.append((label, start, last))
        label, start, last = (
            (current, frame, frame) if current is not None else (None, None, None)
        )
    if label is not None and last - start + 1 >= minimum:
        output.append((label, start, last))
    return output


def mst_edges(nodes: list[dict]) -> list[tuple[int, int]]:
    if len(nodes) < 2:
        return []
    connected, remaining, edges = [0], set(range(1, len(nodes))), []
    while remaining:
        best = min(
            (
                (
                    math.dist(nodes[left]["pitch"], nodes[right]["pitch"]),
                    left,
                    right,
                )
                for left in connected
                for right in remaining
            ),
            key=lambda item: item[0],
        )
        edges.append((best[1], best[2]))
        connected.append(best[2])
        remaining.remove(best[2])
    return edges


def draw_banner(image, label: str, frame: int) -> None:
    text = f"{DISPLAY[label].upper()}  |  fixed event label  |  source frame {frame}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (5, 15, 9), -1)
    cv2.putText(
        image, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (106, 240, 255), 2
    )


def broadcast_frame(
    image,
    frame: int,
    label: str,
    detections: list[dict],
    identities: dict[int, dict],
    ball: dict | None,
) -> Any:
    nodes = []
    for detection in detections:
        if detection.get("track_id") is None or float(detection["confidence"]) < 0.45:
            continue
        track = int(detection["track_id"])
        identity = identities.get(track)
        if not identity or identity["label"] not in COLORS:
            continue
        box = [int(round(value)) for value in detection["bbox"]]
        color = COLORS[identity["label"]]
        cv2.rectangle(image, box[:2], box[2:], color, 3)
        cv2.putText(
            image,
            f"{identity['label'].replace('_', ' ').upper()} #{track}",
            (box[0], max(62, box[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )
        nodes.append(
            {
                "team": identity["label"],
                "point": ((box[0] + box[2]) // 2, box[3]),
                "pitch": ((box[0] + box[2]) / 2, box[3]),
            }
        )
    for team in COLORS:
        members = [node for node in nodes if node["team"] == team]
        for left, right in mst_edges(members):
            cv2.line(
                image,
                members[left]["point"],
                members[right]["point"],
                COLORS[team],
                2,
                cv2.LINE_AA,
            )
    if ball:
        x, y = map(int, ball["image_xy"])
        cv2.circle(image, (x, y), 10, (106, 240, 255), 3)
        cv2.drawMarker(
            image, (x, y), (106, 240, 255), cv2.MARKER_CROSS, 24, 2
        )
    draw_banner(image, label, frame)
    return image


def canonical_frame(
    frame: int,
    label: str,
    objects: list[dict],
    identities: dict[int, dict],
    ball: dict | None,
) -> Any:
    width, height, pad = 1050, 680, 28
    image = np.full((height, width, 3), (42, 74, 42), dtype=np.uint8)
    line = (225, 238, 229)
    cv2.rectangle(image, (pad, pad), (width - pad, height - pad), line, 3)
    cv2.line(image, (width // 2, pad), (width // 2, height - pad), line, 3)
    cv2.circle(image, (width // 2, height // 2), 92, line, 3)
    to_canvas = lambda x, y: (
        int(pad + (x + 52.5) / 105 * (width - 2 * pad)),
        int(pad + (y + 34) / 68 * (height - 2 * pad)),
    )
    nodes = []
    for item in objects:
        track = int(item["track_id"])
        identity = identities.get(track)
        if not identity or identity["label"] not in COLORS:
            continue
        nodes.append(
            {
                "team": identity["label"],
                "point": to_canvas(float(item["x"]), float(item["y"])),
                "pitch": (float(item["x"]), float(item["y"])),
                "track": track,
            }
        )
    for team in COLORS:
        members = [node for node in nodes if node["team"] == team]
        for left, right in mst_edges(members):
            cv2.line(
                image,
                members[left]["point"],
                members[right]["point"],
                COLORS[team],
                3,
                cv2.LINE_AA,
            )
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if (
                nodes[left]["team"] != nodes[right]["team"]
                and math.dist(nodes[left]["pitch"], nodes[right]["pitch"]) <= 12
            ):
                cv2.line(
                    image,
                    nodes[left]["point"],
                    nodes[right]["point"],
                    (65, 190, 220),
                    2,
                    cv2.LINE_AA,
                )
    for node in nodes:
        cv2.circle(image, node["point"], 9, COLORS[node["team"]], -1)
        cv2.circle(image, node["point"], 9, line, 2)
        cv2.putText(
            image,
            str(node["track"]),
            (node["point"][0] + 9, node["point"][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            line,
            1,
        )
    if ball and ball.get("pitch_xy"):
        cv2.circle(
            image, to_canvas(*ball["pitch_xy"]), 8, (106, 240, 255), -1
        )
    draw_banner(image, label, frame)
    return image


def encode(frames: list, output: Path, fps: int) -> None:
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(prefix="tactic-render-") as temporary:
        intermediate = Path(temporary) / "raw.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(intermediate),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, default=5)
    parser.add_argument("--minimum-frames", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    step = max(1, 25 // args.output_fps)
    for model_path in sorted((args.directory / "tactical-model").glob("*.json")):
        tactical = json.loads(model_path.read_text())
        clip_id = tactical["clip_id"]
        result = json.loads(
            (
                args.directory
                / "results"
                / "yolo26m-botsort-high-recall"
                / f"{clip_id}.json"
            ).read_text()
        )
        projection = json.loads(
            (args.directory / "pitch-projections" / f"{clip_id}.json").read_text()
        )
        balls = json.loads(
            (args.directory / "ball-tracking" / f"{clip_id}.json").read_text()
        )
        identity = json.loads(
            (args.directory / "identities" / f"{clip_id}.json").read_text()
        )
        detections = {
            int(row["frame"]): row["detections"] for row in result["frames"]
        }
        projected = {int(row["frame"]): row["objects"] for row in projection["frames"]}
        ball_frames = {int(row["frame"]): row.get("ball") for row in balls["frames"]}
        identities = {int(row["track_id"]): row for row in identity["tracks"]}
        capture = cv2.VideoCapture(result["clip_path"])
        source_images = {}
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            source_images[frame_index] = image
            frame_index += 1
        capture.release()
        for event_index, (label, first, last) in enumerate(
            runs(tactical["frames"], args.minimum_frames), 1
        ):
            samples = list(range(first, last + 1, step))
            if last not in samples:
                samples.append(last)
            samples = [frame for frame in samples if frame in source_images]
            if not samples:
                continue
            event_id = f"{clip_id}-{label}-{event_index}"
            broadcast = [
                broadcast_frame(
                    source_images[frame].copy(),
                    frame,
                    label,
                    detections.get(frame, []),
                    identities,
                    ball_frames.get(frame),
                )
                for frame in samples
            ]
            canonical = [
                canonical_frame(
                    frame,
                    label,
                    projected.get(frame, []),
                    identities,
                    ball_frames.get(frame),
                )
                for frame in samples
            ]
            broadcast_name, canonical_name = (
                f"{event_id}-broadcast.mp4",
                f"{event_id}-canonical.mp4",
            )
            encode(broadcast, args.output_dir / broadcast_name, args.output_fps)
            encode(canonical, args.output_dir / canonical_name, args.output_fps)
            manifest.append(
                {
                    "id": event_id,
                    "clip_id": clip_id,
                    "label": label,
                    "display": DISPLAY[label],
                    "source_frames": [first, last],
                    "fps": args.output_fps,
                    "sampled_frames": samples,
                    "broadcast_video": broadcast_name,
                    "canonical_video": canonical_name,
                }
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "events": manifest}, indent=2) + "\n"
    )
    print(f"Wrote {len(manifest)} fixed-label tactical event pairs")


if __name__ == "__main__":
    main()
