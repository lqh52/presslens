#!/usr/bin/env python3
"""Project tracked player foot points onto a canonical 105 x 68 m pitch."""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .convert_tracklab_state import homography
except ImportError:
    from convert_tracklab_state import homography


def load_parameters(path: Path) -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        images = pickle.loads(archive.read("0_image.pkl"))
    return {
        int(row.frame): row.parameters
        for _, row in images.iterrows()
        if isinstance(row.parameters, dict) and row.parameters
    }


def load_calibration_records(path: Path) -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        images = pickle.loads(archive.read("0_image.pkl"))
    records = {}
    for _, row in images.iterrows():
        if not isinstance(row.parameters, dict) or not row.parameters:
            continue
        records[int(row.frame)] = {
            "parameters": row.parameters,
            "keypoints": len(row.keypoints) if isinstance(row.keypoints, dict) else 0,
            "lines": len(row.lines) if isinstance(row.lines, dict) else 0,
        }
    return records


def calibration_image_scale(
    camera: dict[str, Any], image_width: int, image_height: int
) -> tuple[float, float]:
    """Map source-image pixels into the calibration model's pixel space.

    SN GameState configurations commonly calibrate at 1920x1080 even when the
    input video and downstream detections are 1280x720. Camera intrinsics are
    therefore not necessarily expressed in the detection image coordinate
    system.
    """
    principal_point = camera.get("principal_point", ())
    if (
        len(principal_point) != 2
        or image_width <= 0
        or image_height <= 0
        or float(principal_point[0]) <= 0
        or float(principal_point[1]) <= 0
    ):
        return 1.0, 1.0
    calibration_width = 2.0 * float(principal_point[0])
    calibration_height = 2.0 * float(principal_point[1])
    return calibration_width / image_width, calibration_height / image_height


def calibration_is_reliable(record: dict[str, Any]) -> bool:
    parameters = record["parameters"]
    focal = float(parameters.get("x_focal_length", 0.0))
    return (
        record["keypoints"] >= 6
        and record["lines"] >= 2
        and 1000.0 <= focal <= 20000.0
    )


def source_homography(
    camera: dict[str, Any], image_width: int, image_height: int
) -> np.ndarray:
    scale_x, scale_y = calibration_image_scale(camera, image_width, image_height)
    source_to_calibration = np.diag([scale_x, scale_y, 1.0])
    return np.linalg.inv(source_to_calibration) @ homography(camera)


def image_features(
    detector: Any, image: np.ndarray
) -> tuple[list[Any], np.ndarray | None]:
    return detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)


def feature_transport_features(
    anchor_features: tuple[list[Any], np.ndarray | None],
    target_features: tuple[list[Any], np.ndarray | None],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate an anchor-to-target transform from cached image features."""
    anchor_points, anchor_descriptors = anchor_features
    target_points, target_descriptors = target_features
    if anchor_descriptors is None or target_descriptors is None:
        return None, {"matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        anchor_descriptors, target_descriptors, k=2
    )
    matches = [first for first, second in pairs if first.distance < 0.7 * second.distance]
    if len(matches) < 30:
        return None, {"matches": len(matches), "inliers": 0, "inlier_ratio": 0.0}
    source = np.float32(
        [anchor_points[match.queryIdx].pt for match in matches]
    )
    target = np.float32(
        [target_points[match.trainIdx].pt for match in matches]
    )
    transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / len(matches)
    diagnostics = {
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
    }
    if transform is None or inliers < 20 or ratio < 0.3:
        return None, diagnostics
    return transform, diagnostics


def build_frame_homographies(
    video_path: Path,
    state_path: Path,
    frame_offset: int,
    frame_count: int,
    *,
    maximum_anchor_gap: int = 30,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """Build observed or feature-transported pitch homographies."""
    records = load_calibration_records(state_path)
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    images = {}
    for frame_index in range(frame_count):
        ok, image = capture.read()
        if not ok:
            break
        images[frame_index] = image
    capture.release()
    reliable = {
        frame_index: records[frame_index + frame_offset]
        for frame_index in images
        if frame_index + frame_offset in records
        and calibration_is_reliable(records[frame_index + frame_offset])
    }
    homographies = {
        frame_index: source_homography(record["parameters"], width, height)
        for frame_index, record in reliable.items()
    }
    diagnostics = {
        frame_index: {
            "projection_method": "observed",
            "anchor_frame": frame_index,
            "keypoints": record["keypoints"],
            "lines": record["lines"],
        }
        for frame_index, record in reliable.items()
    }
    detector = cv2.SIFT_create(nfeatures=3000)
    features: dict[int, tuple[list[Any], np.ndarray | None]] = {}

    def features_for(frame_index: int) -> tuple[list[Any], np.ndarray | None]:
        if frame_index not in features:
            features[frame_index] = image_features(detector, images[frame_index])
        return features[frame_index]

    for frame_index, image in images.items():
        if frame_index in homographies:
            continue
        anchor = min(
            reliable,
            key=lambda candidate: abs(candidate - frame_index),
            default=None,
        )
        if anchor is None or abs(anchor - frame_index) > maximum_anchor_gap:
            diagnostics[frame_index] = {"projection_method": "unreliable"}
            continue
        transform, matching = feature_transport_features(
            features_for(anchor), features_for(frame_index)
        )
        if transform is None:
            diagnostics[frame_index] = {
                "projection_method": "unreliable",
                "anchor_frame": anchor,
                **matching,
            }
            continue
        homographies[frame_index] = transform @ homographies[anchor]
        diagnostics[frame_index] = {
            "projection_method": "feature_transported",
            "anchor_frame": anchor,
            **matching,
        }
    return homographies, diagnostics


def unproject_homography(
    world_to_image: np.ndarray, image_point: tuple[float, float]
) -> np.ndarray | None:
    point = np.linalg.inv(world_to_image) @ np.array([*image_point, 1.0])
    if abs(point[2]) < 1e-8:
        return None
    point = point[:2] / point[2]
    if not np.isfinite(point).all():
        return None
    return point


def project_clip(
    result_path: Path,
    state_path: Path,
    frame_offset: int,
    output_path: Path,
    *,
    frame_homographies: dict[int, np.ndarray] | None = None,
    frame_calibration_diagnostics: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    capture = cv2.VideoCapture(str(result["clip_path"]))
    image_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if frame_homographies is None or frame_calibration_diagnostics is None:
        homographies, calibration_diagnostics = build_frame_homographies(
            Path(result["clip_path"]), state_path, frame_offset, frame_count
        )
    else:
        homographies = frame_homographies
        calibration_diagnostics = frame_calibration_diagnostics
    frames = []
    projected, eligible = 0, 0
    observed: dict[int, set[int]] = defaultdict(set)
    raw_points: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for frame in result["frames"]:
        frame_index = int(frame["frame"])
        calibration_frame = frame_index + frame_offset
        frame_homography = homographies.get(frame_index)
        objects = []
        for detection in frame["detections"]:
            if detection.get("track_id") is None:
                continue
            track_id = int(detection["track_id"])
            observed[track_id].add(frame_index)
            eligible += 1
            left, _top, right, bottom = detection["bbox"]
            point = (
                unproject_homography(
                    frame_homography, ((left + right) / 2.0, bottom)
                )
                if frame_homography is not None
                else None
            )
            if point is None or not np.isfinite(point).all():
                continue
            x, y = map(float, point)
            if abs(x) > 57.5 or abs(y) > 39.0:
                continue
            raw_points[track_id][frame_index] = (x, y)
            objects.append({"track_id": track_id, "x": x, "y": y})
            projected += 1
        frames.append(
            {
                "frame": int(frame["frame"]),
                "calibration_frame": calibration_frame,
                "calibration": calibration_diagnostics.get(
                    frame_index, {"projection_method": "unreliable"}
                ),
                "objects": objects,
            }
        )
    # Robustly smooth each track in pitch space and bridge only short internal
    # gaps where the same tracked detection exists on both sides.
    stabilized: dict[int, dict[int, tuple[float, float, bool]]] = defaultdict(dict)
    for track_id, points in raw_points.items():
        valid_frames = sorted(points)
        for frame_index in valid_frames:
            neighbours = [
                points[other]
                for other in valid_frames
                if abs(other - frame_index) <= 2
            ]
            values = np.asarray(neighbours, dtype=np.float64)
            stabilized[track_id][frame_index] = (
                float(np.median(values[:, 0])),
                float(np.median(values[:, 1])),
                "observed",
            )
        for frame_index in sorted(observed[track_id] - set(valid_frames)):
            previous = max(
                (frame for frame in valid_frames if frame < frame_index),
                default=None,
            )
            following = min(
                (frame for frame in valid_frames if frame > frame_index),
                default=None,
            )
            if (
                previous is None
                or following is None
                or following - previous > 8
            ):
                continue
            ratio = (frame_index - previous) / (following - previous)
            left, right = points[previous], points[following]
            stabilized[track_id][frame_index] = (
                left[0] + ratio * (right[0] - left[0]),
                left[1] + ratio * (right[1] - left[1]),
                "interpolated",
            )
        # At shot boundaries or brief calibration dropouts there may be no
        # valid point on both sides. Hold the nearest measured position for at
        # most four detected frames; longer gaps remain explicitly missing.
        remaining = observed[track_id] - set(stabilized[track_id])
        measured = sorted(stabilized[track_id])
        for frame_index in sorted(remaining):
            nearest = min(
                measured,
                key=lambda other: abs(other - frame_index),
                default=None,
            )
            if nearest is None or abs(nearest - frame_index) > 4:
                continue
            x, y, _method = stabilized[track_id][nearest]
            stabilized[track_id][frame_index] = (x, y, "held")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    interpolated = 0
    for track_id, points in stabilized.items():
        for frame_index, (x, y, method) in points.items():
            by_frame[frame_index].append(
                {
                    "track_id": track_id,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "estimated": method != "observed",
                    "projection_method": method,
                }
            )
            interpolated += int(method == "interpolated")
    for frame in frames:
        frame["objects"] = sorted(
            by_frame.get(frame["frame"], []), key=lambda row: row["track_id"]
        )
    projected = sum(len(frame["objects"]) for frame in frames)
    output = {
        "schema_version": 1,
        "clip_id": result["clip_id"],
        "source_result": str(result_path),
        "calibration_state": str(state_path),
        "frame_offset": frame_offset,
        "pitch_length_m": 105.0,
        "pitch_width_m": 68.0,
        "source_image_size": [image_width, image_height],
        "calibration_pixel_scaling": "principal_point",
        "reliable_calibration_frames": sum(
            row.get("projection_method") == "observed"
            for row in calibration_diagnostics.values()
        ),
        "feature_transported_frames": sum(
            row.get("projection_method") == "feature_transported"
            for row in calibration_diagnostics.values()
        ),
        "unreliable_calibration_frames": sum(
            row.get("projection_method") == "unreliable"
            for row in calibration_diagnostics.values()
        ),
        "coverage": round(projected / eligible, 6) if eligible else 0.0,
        "interpolated_points": interpolated,
        "held_points": sum(
            int(row["projection_method"] == "held")
            for frame in frames
            for row in frame["objects"]
        ),
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["clips"]
    for clip_id, config in manifest.items():
        output = project_clip(
            args.results_dir / f"{clip_id}.json",
            Path(config["state"]),
            int(config["frame_offset"]),
            args.output_dir / f"{clip_id}.json",
        )
        print(f"{clip_id}: {output['coverage']:.1%} projected")


if __name__ == "__main__":
    main()
