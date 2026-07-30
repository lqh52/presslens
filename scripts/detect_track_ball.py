#!/usr/bin/env python3
"""Detect, temporally track, and pitch-project the football in short clips."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .project_tracking_to_pitch import (
        build_frame_homographies,
        unproject_homography,
    )
except ImportError:
    from project_tracking_to_pitch import (
        build_frame_homographies,
        unproject_homography,
    )


def candidate_rows(result) -> list[dict[str, Any]]:
    rows = []
    if result.boxes is None:
        return rows
    for bbox, confidence in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
    ):
        left, top, right, bottom = map(float, bbox)
        rows.append(
            {
                "confidence": float(confidence),
                "bbox": [left, top, right, bottom],
                "image_xy": [(left + right) / 2.0, (top + bottom) / 2.0],
                "ground_xy": [(left + right) / 2.0, bottom],
            }
        )
    return rows


def select_trajectory(
    candidates: list[list[dict[str, Any]]], width: int, height: int
) -> dict[int, dict[str, Any]]:
    """Viterbi selection balancing detector confidence and image continuity."""
    diagonal = math.hypot(width, height)
    active: list[tuple[float, list[tuple[int, int]]]] = []
    best_path: list[tuple[int, int]] = []
    best_score = -math.inf
    for frame_index, rows in enumerate(candidates):
        if not rows:
            continue
        next_active = []
        for candidate_index, row in enumerate(rows):
            local = 0.35 + 4.0 * row["confidence"]
            score, path = local, [(frame_index, candidate_index)]
            for previous_score, previous_path in active:
                previous_frame, previous_index = previous_path[-1]
                gap = frame_index - previous_frame
                if gap > 8:
                    continue
                previous = candidates[previous_frame][previous_index]
                distance = math.dist(row["image_xy"], previous["image_xy"]) / diagonal
                proposal = (
                    previous_score
                    + local
                    - 7.0 * distance
                    - 0.18 * (gap - 1)
                )
                if proposal > score:
                    score = proposal
                    path = previous_path + [(frame_index, candidate_index)]
            next_active.append((score, path))
            if score > best_score:
                best_score, best_path = score, path
        active = sorted(next_active, reverse=True, key=lambda item: item[0])[:20]
    return {
        frame: {**candidates[frame][index], "method": "detected"}
        for frame, index in best_path
    }


def bridge_short_gaps(
    trajectory: dict[int, dict[str, Any]], maximum_gap: int = 5
) -> dict[int, dict[str, Any]]:
    output = dict(trajectory)
    frames = sorted(trajectory)
    for left_frame, right_frame in zip(frames, frames[1:]):
        if right_frame - left_frame <= 1 or right_frame - left_frame > maximum_gap + 1:
            continue
        left, right = trajectory[left_frame], trajectory[right_frame]
        for frame in range(left_frame + 1, right_frame):
            ratio = (frame - left_frame) / (right_frame - left_frame)
            image_xy = [
                left["image_xy"][axis]
                + ratio * (right["image_xy"][axis] - left["image_xy"][axis])
                for axis in range(2)
            ]
            ground_xy = [
                left["ground_xy"][axis]
                + ratio * (right["ground_xy"][axis] - left["ground_xy"][axis])
                for axis in range(2)
            ]
            output[frame] = {
                "confidence": min(left["confidence"], right["confidence"]) * 0.65,
                "bbox": None,
                "image_xy": image_xy,
                "ground_xy": ground_xy,
                "method": "interpolated",
            }
    return output


def process_clip(
    model: YOLO,
    video_path: Path,
    state_path: Path,
    frame_offset: int,
    output_path: Path,
    *,
    confidence: float,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    predictions = model.predict(
        str(video_path),
        stream=True,
        verbose=False,
        classes=[32],
        conf=confidence,
        imgsz=imgsz,
        device=device,
    )
    candidates = [candidate_rows(result) for result in predictions]
    trajectory = bridge_short_gaps(select_trajectory(candidates, width, height))
    homographies, calibration_diagnostics = build_frame_homographies(
        video_path, state_path, frame_offset, frame_count
    )
    frames = []
    projected = 0
    for frame_index in range(frame_count):
        ball = trajectory.get(frame_index)
        if ball is not None:
            frame_homography = homographies.get(frame_index)
            pitch = (
                unproject_homography(frame_homography, tuple(ball["ground_xy"]))
                if frame_homography is not None
                else None
            )
            if (
                pitch is not None
                and np.isfinite(pitch).all()
                and abs(float(pitch[0])) <= 57.5
                and abs(float(pitch[1])) <= 39.0
            ):
                ball["pitch_xy"] = [
                    round(float(pitch[0]), 4),
                    round(float(pitch[1]), 4),
                ]
                projected += 1
            else:
                ball["pitch_xy"] = None
            ball["confidence"] = round(float(ball["confidence"]), 6)
            ball["image_xy"] = [round(float(value), 3) for value in ball["image_xy"]]
            ball["ground_xy"] = [round(float(value), 3) for value in ball["ground_xy"]]
        frames.append(
            {
                "frame": frame_index,
                "calibration": calibration_diagnostics.get(
                    frame_index, {"projection_method": "unreliable"}
                ),
                "ball": ball,
            }
        )
    detected = sum(row["ball"] is not None for row in frames)
    output = {
        "schema_version": 1,
        "clip_id": output_path.stem,
        "video_path": str(video_path),
        "model": str(model.ckpt_path),
        "configuration": {
            "class": 32,
            "class_name": "sports ball",
            "confidence": confidence,
            "imgsz": imgsz,
            "maximum_interpolation_gap": 5,
        },
        "coverage": round(detected / frame_count, 6) if frame_count else 0.0,
        "pitch_coverage": round(projected / frame_count, 6) if frame_count else 0.0,
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo/yolo26m.pt"))
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    videos = json.loads(args.video_manifest.read_text())["clips"]
    projections = json.loads(args.projection_manifest.read_text())["clips"]
    model = YOLO(str(args.model))
    for row in videos:
        clip_id = row["id"]
        destination = args.output_dir / f"{clip_id}.json"
        if args.skip_existing and destination.exists():
            continue
        video_path = Path(row["clip_path"])
        if not video_path.is_absolute():
            video_path = (
                video_path.resolve()
                if video_path.exists()
                else (args.video_manifest.parent / video_path).resolve()
            )
        config = projections[clip_id]
        output = process_clip(
            model,
            video_path,
            Path(config["state"]),
            int(config["frame_offset"]),
            destination,
            confidence=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
        )
        print(
            f"{clip_id}: image {output['coverage']:.1%}, "
            f"pitch {output['pitch_coverage']:.1%}"
        )


if __name__ == "__main__":
    main()
