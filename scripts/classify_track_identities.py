#!/usr/bin/env python3
"""Classify tracked people as Team A, Team B, goalkeeper, or other.

Role supervision comes from hard SoccerNet-GSR annotations. Team identity is
deliberately fixture-local: stable torso-colour features are clustered only
after non-outfield tracks have been removed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROLE_MAP = {
    "player": "outfield",
    "goalkeeper": "goalkeeper",
    "referee": "other",
    "other": "other",
}


def torso_crop(image: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    left, top, right, bottom = bbox
    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)
    x1 = max(0, int(left + 0.16 * box_width))
    x2 = min(width, int(right - 0.16 * box_width))
    y1 = max(0, int(top + 0.08 * box_height))
    y2 = min(height, int(top + 0.55 * box_height))
    crop = image[y1:y2, x1:x2]
    return crop if crop.size >= 90 else None


def appearance_features(image: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    crop = torso_crop(image, bbox)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, None).reshape(-1)
    lab_histograms = []
    for channel in (1, 2):
        histogram = cv2.calcHist([lab], [channel], None, [12], [0, 256])
        lab_histograms.append(cv2.normalize(histogram, None).reshape(-1))
    pixels = crop.reshape(-1, 3).astype(np.float32)
    image_height, image_width = image.shape[:2]
    left, top, right, bottom = bbox
    geometry = np.asarray(
        [
            (right - left) / image_width,
            (bottom - top) / image_height,
            (right - left) / max(bottom - top, 1.0),
            bottom / image_height,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            hsv_hist.astype(np.float32),
            *[value.astype(np.float32) for value in lab_histograms],
            pixels.mean(axis=0) / 255.0,
            pixels.std(axis=0) / 255.0,
            geometry,
        ]
    )


def bbox_from_annotation(annotation: dict[str, Any]) -> list[float]:
    box = annotation["bbox_image"]
    return [
        float(box["x"]),
        float(box["y"]),
        float(box["x"] + box["w"]),
        float(box["y"] + box["h"]),
    ]


def sequence_names(labels_root: Path) -> list[str]:
    return sorted(path.parent.name for path in labels_root.glob("*/Labels-GameState.json"))


def sampled_role_features(
    archive_path: Path,
    labels_root: Path,
    *,
    sequences: int,
    frame_stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = sequence_names(labels_root)
    random.Random(seed).shuffle(names)
    names = sorted(names[:sequences])
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for name in names:
            payload = json.loads(
                (labels_root / name / "Labels-GameState.json").read_text()
            )
            images = {
                str(row["image_id"]): row
                for row in payload["images"]
                if row.get("is_labeled", True)
                and (int(Path(row["file_name"]).stem) - 1) % frame_stride == 0
            }
            annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for annotation in payload["annotations"]:
                role = annotation.get("attributes", {}).get("role")
                if str(annotation["image_id"]) in images and role in ROLE_MAP:
                    annotations[str(annotation["image_id"])].append(annotation)
            for image_id, image_row in images.items():
                member = f"{name}/img1/{image_row['file_name']}"
                if member not in members:
                    continue
                image = cv2.imdecode(
                    np.frombuffer(archive.read(member), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    continue
                for annotation in annotations.get(image_id, []):
                    vector = appearance_features(image, bbox_from_annotation(annotation))
                    if vector is None:
                        continue
                    features.append(vector)
                    labels.append(ROLE_MAP[annotation["attributes"]["role"]])
                    groups.append(name)
    if not features:
        raise RuntimeError(f"No labelled crops read from {archive_path}")
    return np.stack(features), np.asarray(labels), np.asarray(groups)


def train_role_model(
    train_archive: Path,
    train_labels: Path,
    valid_archive: Path,
    valid_labels: Path,
    output: Path,
    *,
    train_sequences: int,
    valid_sequences: int,
    frame_stride: int,
) -> dict[str, Any]:
    train_x, train_y, train_groups = sampled_role_features(
        train_archive,
        train_labels,
        sequences=train_sequences,
        frame_stride=frame_stride,
        seed=17,
    )
    valid_x, valid_y, valid_groups = sampled_role_features(
        valid_archive,
        valid_labels,
        sequences=valid_sequences,
        frame_stride=frame_stride,
        seed=29,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=17,
    ).fit(train_x, train_y)
    predictions = model.predict(valid_x)
    report = {
        "schema_version": 1,
        "task": "outfield_goalkeeper_other",
        "hard_label_source": "SoccerNet-GSR role annotations",
        "train_sequences": len(set(train_groups)),
        "valid_sequences": len(set(valid_groups)),
        "train_crops": len(train_y),
        "valid_crops": len(valid_y),
        "train_classes": dict(Counter(train_y.tolist())),
        "valid_classes": dict(Counter(valid_y.tolist())),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(valid_y, predictions)), 5
        ),
        "classification": classification_report(
            valid_y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "frame_stride": frame_stride,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump({"model": model, "report": report}, handle)
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def extract_track_features(
    video_path: Path,
    frames: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    detections_by_frame = {
        int(frame["frame"]): frame["detections"] for frame in frames
    }
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    confidences: dict[int, list[float]] = defaultdict(list)
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        for detection in detections_by_frame.get(frame_index, []):
            if detection.get("track_id") is None:
                continue
            track_id = int(detection["track_id"])
            vector = appearance_features(image, detection["bbox"])
            if vector is not None:
                samples[track_id].append(vector)
            confidences[track_id].append(float(detection.get("confidence", 0.0)))
            boxes[track_id].append(detection["bbox"])
        frame_index += 1
    capture.release()
    return {
        track_id: {
            "features": np.median(np.stack(values), axis=0),
            "samples": len(values),
            "detections": len(confidences[track_id]),
            "mean_detection_confidence": statistics.fmean(confidences[track_id]),
            "median_bbox": np.median(np.asarray(boxes[track_id]), axis=0).tolist(),
        }
        for track_id, values in samples.items()
        if values
    }


def classify_tracks(
    result_path: Path,
    role_model_path: Path,
    output_path: Path,
    *,
    role_threshold: float,
    team_margin_threshold: float,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text())
    with role_model_path.open("rb") as handle:
        role_artifact = pickle.load(handle)
    model = role_artifact["model"]
    tracks = extract_track_features(Path(payload["clip_path"]), payload["frames"])
    if len(tracks) < 2:
        raise RuntimeError(f"Too few tracks with appearance features in {result_path}")
    track_ids = sorted(tracks)
    matrix = np.stack([tracks[track_id]["features"] for track_id in track_ids])
    role_probabilities = model.predict_proba(matrix)
    role_classes = model.classes_.tolist()
    outfield_index = role_classes.index("outfield")
    goalkeeper_index = role_classes.index("goalkeeper")
    eligible_indices = [
        index
        for index, track_id in enumerate(track_ids)
        if role_probabilities[index, outfield_index] >= role_threshold
        and tracks[track_id]["detections"] >= 3
    ]
    if len(eligible_indices) < 2:
        eligible_indices = sorted(
            range(len(track_ids)),
            key=lambda index: role_probabilities[index, outfield_index],
            reverse=True,
        )[: max(2, min(12, len(track_ids)))]
    # Team clustering uses appearance only; geometry is reserved for role filtering.
    appearance_dimensions = matrix.shape[1] - 4
    team_matrix = matrix[eligible_indices, :appearance_dimensions]
    team_weights = np.asarray(
        [min(tracks[track_ids[index]]["detections"], 25) for index in eligible_indices],
        dtype=np.float64,
    )
    clustering = KMeans(n_clusters=2, n_init=20, random_state=17).fit(
        team_matrix,
        sample_weight=team_weights,
    )
    centers = clustering.cluster_centers_
    distances = np.linalg.norm(
        matrix[:, None, :appearance_dimensions] - centers[None, :, :],
        axis=2,
    )
    assignments = []
    for index, track_id in enumerate(track_ids):
        probabilities = {
            role: round(float(role_probabilities[index, role_classes.index(role)]), 6)
            for role in role_classes
        }
        ordered = np.argsort(distances[index])
        nearest, second = int(ordered[0]), int(ordered[1])
        margin = float(
            (distances[index, second] - distances[index, nearest])
            / max(distances[index, second], 1e-6)
        )
        role = role_classes[int(np.argmax(role_probabilities[index]))]
        role_confidence = float(np.max(role_probabilities[index]))
        if role == "goalkeeper" and role_confidence >= role_threshold:
            label = f"team_{'a' if nearest == 0 else 'b'}"
            goalkeeper = True
        elif (
            role == "outfield"
            and role_confidence >= role_threshold
            and margin >= team_margin_threshold
        ):
            label = f"team_{'a' if nearest == 0 else 'b'}"
            goalkeeper = False
        else:
            label = "other"
            goalkeeper = False
        assignments.append(
            {
                "track_id": track_id,
                "label": label,
                "goalkeeper": goalkeeper,
                "role_prediction": role,
                "role_confidence": round(role_confidence, 6),
                "role_probabilities": probabilities,
                "team_cluster": nearest if label != "other" else None,
                "team_margin": round(margin, 6),
                **{
                    key: value
                    for key, value in tracks[track_id].items()
                    if key != "features"
                },
            }
        )
    output = {
        "schema_version": 1,
        "clip_id": payload["clip_id"],
        "source_result": str(result_path),
        "role_model": str(role_model_path),
        "configuration": {
            "role_threshold": role_threshold,
            "team_margin_threshold": team_margin_threshold,
            "team_signal": "fixture-local track-level torso appearance KMeans",
        },
        "counts": dict(Counter(item["label"] for item in assignments)),
        "goalkeepers": sum(item["goalkeeper"] for item in assignments),
        "tracks": assignments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def fixture_id(clip_id: str) -> str:
    """Return a match-level key shared by clips from the same fixture."""
    return "-".join(clip_id.split("-")[:3])


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def classify_fixtures(
    results_dir: Path,
    reid_dir: Path,
    dino_dir: Path,
    role_model_path: Path,
    output_dir: Path,
    *,
    role_threshold: float,
    team_margin_threshold: float,
    dino_weight: float,
    reid_weight: float,
    color_weight: float,
    validation_strength: float,
) -> list[dict[str, Any]]:
    """Cluster teams jointly across every clip belonging to one fixture."""
    with role_model_path.open("rb") as handle:
        role_model = pickle.load(handle)["model"]
    role_classes = role_model.classes_.tolist()
    outfield_index = role_classes.index("outfield")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result_path in sorted(results_dir.glob("*.json")):
        payload = json.loads(result_path.read_text())
        clip_id = payload["clip_id"]
        reid_path = reid_dir / result_path.name
        if not reid_path.exists():
            raise FileNotFoundError(f"Missing PRTReID cache: {reid_path}")
        reid_payload = json.loads(reid_path.read_text())
        reid_tracks = {
            int(row["track_id"]): row for row in reid_payload["tracks"]
        }
        dino_path = dino_dir / result_path.name
        if not dino_path.exists():
            raise FileNotFoundError(f"Missing DINO cache: {dino_path}")
        dino_payload = json.loads(dino_path.read_text())
        dino_tracks = {
            int(row["track_id"]): row for row in dino_payload["tracks"]
        }
        tracks = extract_track_features(Path(payload["clip_path"]), payload["frames"])
        track_ids = sorted(set(tracks) & set(reid_tracks) & set(dino_tracks))
        if not track_ids:
            raise RuntimeError(f"No common tracks in {result_path} and {reid_path}")
        appearance = np.stack([tracks[track_id]["features"] for track_id in track_ids])
        role_probabilities = role_model.predict_proba(appearance)
        for index, track_id in enumerate(track_ids):
            grouped[fixture_id(clip_id)].append(
                {
                    "clip_id": clip_id,
                    "result_path": result_path,
                    "track_id": track_id,
                    "track": tracks[track_id],
                    "color": appearance[index, :-4],
                    "reid": np.asarray(
                        reid_tracks[track_id]["embedding"], dtype=np.float32
                    ),
                    "dino": np.asarray(
                        dino_tracks[track_id]["embedding"], dtype=np.float32
                    ),
                    "dino_model": dino_payload["model"],
                    "prt": reid_tracks[track_id],
                    "role_probabilities": role_probabilities[index],
                }
            )

    outputs: list[dict[str, Any]] = []
    total_weight = dino_weight + reid_weight + color_weight
    if total_weight <= 0:
        raise ValueError("At least one feature weight must be positive")
    dino_weight /= total_weight
    reid_weight /= total_weight
    color_weight /= total_weight
    match_anchors: dict[str, Any] = {}
    for match_id, rows in grouped.items():
        eligible = [
            index
            for index, row in enumerate(rows)
            if row["role_probabilities"][outfield_index] >= role_threshold
            and row["track"]["detections"] >= 3
        ]
        if len(eligible) < 2:
            eligible = sorted(
                range(len(rows)),
                key=lambda index: rows[index]["role_probabilities"][outfield_index],
                reverse=True,
            )[: max(2, min(20, len(rows)))]

        scaler = StandardScaler().fit(np.stack([rows[index]["color"] for index in eligible]))
        colors = normalized_rows(
            scaler.transform(np.stack([row["color"] for row in rows]))
        )
        embeddings = normalized_rows(np.stack([row["reid"] for row in rows]))
        dino_embeddings = normalized_rows(np.stack([row["dino"] for row in rows]))
        fused = np.concatenate(
            [
                np.sqrt(dino_weight) * dino_embeddings,
                np.sqrt(reid_weight) * embeddings,
                np.sqrt(color_weight) * colors,
            ],
            axis=1,
        )
        weights = np.asarray(
            [min(rows[index]["track"]["detections"], 25) for index in eligible],
            dtype=np.float64,
        )
        clustering = KMeans(n_clusters=2, n_init=30, random_state=17).fit(
            fused[eligible],
            sample_weight=weights,
        )
        distances = np.linalg.norm(
            fused[:, None, :] - clustering.cluster_centers_[None, :, :],
            axis=2,
        )
        labels = clustering.labels_
        component_matrices = {
            "dino": dino_embeddings,
            "prtreid": embeddings,
            "color": colors,
        }
        component_distances: dict[str, np.ndarray] = {}
        anchor_summary = []
        for cluster_index in range(2):
            member_indices = [
                eligible[position]
                for position, label in enumerate(labels)
                if int(label) == cluster_index
            ]
            representatives = sorted(
                member_indices, key=lambda index: distances[index, cluster_index]
            )[:3]
            anchor_summary.append(
                {
                    "cluster": cluster_index,
                    "label": f"team_{'a' if cluster_index == 0 else 'b'}",
                    "support_tracks": len(member_indices),
                    "support_detections": int(
                        sum(rows[index]["track"]["detections"] for index in member_indices)
                    ),
                    "representative_tracks": [
                        {
                            "clip_id": rows[index]["clip_id"],
                            "track_id": rows[index]["track_id"],
                        }
                        for index in representatives
                    ],
                }
            )
        for signal, signal_matrix in component_matrices.items():
            centers = np.stack(
                [
                    normalized_rows(
                        np.average(
                            signal_matrix[
                                [
                                    eligible[position]
                                    for position, label in enumerate(labels)
                                    if int(label) == cluster_index
                                ]
                            ],
                            axis=0,
                            weights=[
                                weights[position]
                                for position, label in enumerate(labels)
                                if int(label) == cluster_index
                            ],
                        )[None, :]
                    )[0]
                    for cluster_index in range(2)
                ]
            )
            component_distances[signal] = np.linalg.norm(
                signal_matrix[:, None, :] - centers[None, :, :], axis=2
            )
        quality = (
            float(silhouette_score(fused[eligible], labels))
            if len(eligible) > 2 and len(set(labels.tolist())) == 2
            else None
        )

        provisional: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            probabilities = {
                role: round(
                    float(row["role_probabilities"][role_classes.index(role)]), 6
                )
                for role in role_classes
            }
            ordered = np.argsort(distances[index])
            nearest, second = int(ordered[0]), int(ordered[1])
            margin = float(
                (distances[index, second] - distances[index, nearest])
                / max(distances[index, second], 1e-6)
            )
            role = role_classes[int(np.argmax(row["role_probabilities"]))]
            role_confidence = float(np.max(row["role_probabilities"]))
            goalkeeper = role == "goalkeeper" and role_confidence >= role_threshold
            if (
                (role == "outfield" or goalkeeper)
                and role_confidence >= role_threshold
                and (goalkeeper or margin >= team_margin_threshold)
            ):
                label = f"team_{'a' if nearest == 0 else 'b'}"
            else:
                label = "other"
            provisional.append(
                {
                    "track_id": row["track_id"],
                    "label": label,
                    "goalkeeper": goalkeeper,
                    "role_prediction": role,
                    "role_confidence": round(role_confidence, 6),
                    "role_probabilities": probabilities,
                    "team_cluster": nearest if label != "other" else None,
                    "team_margin": round(margin, 6),
                    "component_distances": {
                        signal: [round(float(value), 6) for value in values[index]]
                        for signal, values in component_distances.items()
                    },
                    "tuning_features": {
                        signal: [
                            round(float(value), 6)
                            for value in signal_matrix[index]
                        ]
                        for signal, signal_matrix in component_matrices.items()
                    },
                    "prt_role_vote": row["prt"]["prt_role_vote"],
                    "prt_role_votes": row["prt"]["prt_role_votes"],
                    **{
                        key: value
                        for key, value in row["track"].items()
                        if key != "features"
                    },
                }
            )

        # Match-level sanity pass. It only rescues player-like tracks that the
        # supervised role model left as "other"; referees remain protected.
        other_support = sum(
            item["detections"] for item in provisional if item["label"] == "other"
        )
        total_support = sum(item["detections"] for item in provisional)
        target_other_ratio = 0.35 - 0.23 * validation_strength
        candidates = sorted(
            (
                (index, item)
                for index, item in enumerate(provisional)
                if item["label"] == "other"
                and item["prt_role_vote"] == "player"
                and item["role_probabilities"]["outfield"]
                >= role_threshold * (1.0 - 0.55 * validation_strength)
                and item["team_margin"]
                >= team_margin_threshold * (1.0 - 0.75 * validation_strength)
            ),
            key=lambda pair: (
                pair[1]["team_margin"]
                * pair[1]["role_probabilities"]["outfield"]
                * min(pair[1]["detections"], 25)
            ),
            reverse=True,
        )
        relabelled = 0
        for index, item in candidates:
            if total_support <= 0 or other_support / total_support <= target_other_ratio:
                break
            cluster = int(np.argmin(distances[index]))
            item["label"] = f"team_{'a' if cluster == 0 else 'b'}"
            item["team_cluster"] = cluster
            item["postprocess_relabelled"] = True
            item["postprocess_reason"] = "excess player-like other support"
            other_support -= item["detections"]
            relabelled += 1

        by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row, item in zip(rows, provisional):
            by_clip[row["clip_id"]].append(item)
        match_anchors[match_id] = {
            "fixture_id": match_id,
            "signals": {
                "dino": rows[0]["dino_model"],
                "prtreid": "SoccerNet PRTReID baseline",
                "color": "torso HSV/Lab",
            },
            "weights": {
                "dino": dino_weight,
                "prtreid": reid_weight,
                "color": color_weight,
            },
            "anchors": anchor_summary,
            "validation": {
                "strength": validation_strength,
                "target_other_support_ratio": target_other_ratio,
                "relabelled_tracks": relabelled,
            },
        }
        for clip_id, assignments in by_clip.items():
            source = next(row["result_path"] for row in rows if row["clip_id"] == clip_id)
            output = {
                "schema_version": 2,
                "clip_id": clip_id,
                "fixture_id": match_id,
                "source_result": str(source),
                "role_model": str(role_model_path),
                "configuration": {
                    "role_threshold": role_threshold,
                    "team_margin_threshold": team_margin_threshold,
                    "team_signal": "fixture-level DINO + SoccerNet PRTReID + torso colour anchors",
                    "dino_model": rows[0]["dino_model"],
                    "dino_weight": dino_weight,
                    "reid_weight": reid_weight,
                    "color_weight": color_weight,
                    "validation_strength": validation_strength,
                    "postprocess_relabelled_fixture_tracks": relabelled,
                    "fixture_silhouette": (
                        round(quality, 6) if quality is not None else None
                    ),
                },
                "counts": dict(Counter(item["label"] for item in assignments)),
                "goalkeepers": sum(item["goalkeeper"] for item in assignments),
                "anchors": anchor_summary,
                "tracks": assignments,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / source.name).write_text(json.dumps(output, indent=2) + "\n")
            outputs.append(output)
    (output_dir / "match-anchors.json").write_text(
        json.dumps({"schema_version": 1, "fixtures": match_anchors}, indent=2) + "\n"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-role")
    train.add_argument("--train-archive", type=Path, default=Path("data/raw/gamestate-2025/train.zip"))
    train.add_argument("--train-labels", type=Path, default=Path("data/raw/gamestate-2025/labels/train"))
    train.add_argument("--valid-archive", type=Path, default=Path("data/raw/gamestate-2025/valid.zip"))
    train.add_argument("--valid-labels", type=Path, default=Path("data/raw/gamestate-2025/labels/valid"))
    train.add_argument("--output", type=Path, default=Path("models/gsr_role_classifier.pkl"))
    train.add_argument("--train-sequences", type=int, default=18)
    train.add_argument("--valid-sequences", type=int, default=6)
    train.add_argument("--frame-stride", type=int, default=25)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--result", type=Path, required=True)
    classify.add_argument("--role-model", type=Path, default=Path("models/gsr_role_classifier.pkl"))
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--role-threshold", type=float, default=0.55)
    classify.add_argument("--team-margin-threshold", type=float, default=0.08)
    fixtures = subparsers.add_parser("classify-fixtures")
    fixtures.add_argument("--results-dir", type=Path, required=True)
    fixtures.add_argument("--reid-dir", type=Path, required=True)
    fixtures.add_argument("--dino-dir", type=Path, required=True)
    fixtures.add_argument("--role-model", type=Path, default=Path("models/gsr_role_classifier.pkl"))
    fixtures.add_argument("--output-dir", type=Path, required=True)
    fixtures.add_argument("--role-threshold", type=float, default=0.55)
    fixtures.add_argument("--team-margin-threshold", type=float, default=0.08)
    fixtures.add_argument("--dino-weight", type=float, default=0.45)
    fixtures.add_argument("--reid-weight", type=float, default=0.35)
    fixtures.add_argument("--color-weight", type=float, default=0.20)
    fixtures.add_argument("--validation-strength", type=float, default=0.75)
    args = parser.parse_args()
    if args.command == "train-role":
        report = train_role_model(
            args.train_archive,
            args.train_labels,
            args.valid_archive,
            args.valid_labels,
            args.output,
            train_sequences=args.train_sequences,
            valid_sequences=args.valid_sequences,
            frame_stride=args.frame_stride,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "classify":
        result = classify_tracks(
            args.result,
            args.role_model,
            args.output,
            role_threshold=args.role_threshold,
            team_margin_threshold=args.team_margin_threshold,
        )
        print(json.dumps({"clip_id": result["clip_id"], "counts": result["counts"], "goalkeepers": result["goalkeepers"]}))
    else:
        for name in ("dino_weight", "reid_weight", "color_weight", "validation_strength"):
            if not 0.0 <= getattr(args, name) <= 1.0:
                parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
        results = classify_fixtures(
            args.results_dir,
            args.reid_dir,
            args.dino_dir,
            args.role_model,
            args.output_dir,
            role_threshold=args.role_threshold,
            team_margin_threshold=args.team_margin_threshold,
            dino_weight=args.dino_weight,
            reid_weight=args.reid_weight,
            color_weight=args.color_weight,
            validation_strength=args.validation_strength,
        )
        print(
            json.dumps(
                [
                    {
                        "clip_id": result["clip_id"],
                        "fixture_id": result["fixture_id"],
                        "counts": result["counts"],
                        "silhouette": result["configuration"]["fixture_silhouette"],
                    }
                    for result in results
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
