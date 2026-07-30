#!/usr/bin/env python3
"""Train a match-calibrated team classifier from reviewed player tracks."""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from team_identity import aggregate_track_embedding, read_human_labels


def latest_state(root: Path, video_id: str) -> Path:
    pattern = f"multi-{video_id}/**/states/multi-{video_id}.pklz"
    paths = [path for path in root.glob(pattern) if path.stat().st_size]
    if not paths:
        raise FileNotFoundError(f"No TrackLab state matching {pattern}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_detections(path: Path):
    with zipfile.ZipFile(path) as archive:
        return pickle.loads(archive.read("0.pkl"))


def fit(x: np.ndarray, y: np.ndarray, regularization: float):
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=3000,
        random_state=17,
    ).fit(scaler.transform(x), y)
    return scaler, classifier


def validation_report(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    regularization: float,
    threshold: float,
) -> dict:
    predictions = []
    truths = []
    scores = []
    skipped = []
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if len(set(y[train].tolist())) < 2:
            skipped.append(group)
            continue
        scaler, classifier = fit(x[train], y[train], regularization)
        probabilities = classifier.predict_proba(scaler.transform(x[test]))
        indices = probabilities.argmax(axis=1)
        predictions.extend(classifier.classes_[indices].tolist())
        truths.extend(y[test].tolist())
        scores.extend(probabilities.max(axis=1).tolist())
    if not truths:
        return {"evaluated_tracks": 0, "skipped_groups": skipped}

    labels = sorted(set(y.tolist()))
    accepted = np.asarray(scores) >= threshold
    truth_array = np.asarray(truths)
    prediction_array = np.asarray(predictions)
    return {
        "split": "leave-one-video-out",
        "evaluated_tracks": len(truths),
        "skipped_groups": skipped,
        "accuracy": round(float(accuracy_score(truth_array, prediction_array)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truth_array, prediction_array)), 4
        ),
        "recall": {
            label: round(value, 4)
            for label, value in zip(
                labels,
                recall_score(
                    truth_array,
                    prediction_array,
                    labels=labels,
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion": confusion_matrix(
            truth_array, prediction_array, labels=labels
        ).tolist(),
        "threshold": threshold,
        "accepted_coverage": round(float(accepted.mean()), 4),
        "accepted_accuracy": (
            round(
                float(
                    accuracy_score(
                        truth_array[accepted], prediction_array[accepted]
                    )
                ),
                4,
            )
            if accepted.any()
            else None
        ),
        "score_note": "Maximum logistic probability; not independently calibrated",
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-directory",
        type=Path,
        default=Path("data/review/team_tracks"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("third_party/sn-gamestate/outputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/team_identity_burnley_arsenal.npz"),
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.3,
        help="Inverse L2 regularization strength passed to LogisticRegression",
    )
    parser.add_argument("--minimum-per-club", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads((args.review_directory / "manifest.json").read_text())
    human_labels = read_human_labels(args.review_directory / "labels.json")
    items = {item["key"]: item for item in manifest["items"]}
    usable = {
        key: label
        for key, label in human_labels.items()
        if key in items and label in {"arsenal", "burnley", "ignore"}
    }
    counts = Counter(usable.values())
    if min(counts.get("arsenal", 0), counts.get("burnley", 0)) < args.minimum_per_club:
        raise RuntimeError(
            "Need at least "
            f"{args.minimum_per_club} reviewed tracks per club; currently {dict(counts)}"
        )

    detections_by_video = {}
    vectors = []
    labels = []
    groups = []
    keys = []
    for key, label in sorted(usable.items()):
        item = items[key]
        video_id = item["video_id"]
        if video_id not in detections_by_video:
            detections_by_video[video_id] = load_detections(
                latest_state(args.state_root, video_id)
            )
        detections = detections_by_video[video_id]
        rows = detections[
            (detections.track_id == int(item["track_id"]))
            & (detections.role == "player")
        ]
        embedding = aggregate_track_embedding(rows)
        if embedding is None:
            continue
        vectors.append(embedding)
        labels.append(label)
        groups.append(video_id)
        keys.append(key)

    x = np.stack(vectors)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    report = validation_report(
        x, y, group_array, args.regularization, args.threshold
    )
    scaler, classifier = fit(x, y, args.regularization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        artifact_version=np.asarray([1], dtype=np.int64),
        coef=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        classes=classifier.classes_.astype("U"),
        threshold=np.asarray([args.threshold], dtype=np.float32),
        training_tracks=np.asarray([len(x)], dtype=np.int64),
    )
    temporary.replace(args.output)
    report.update(
        {
            "training_tracks": len(x),
            "training_labels": dict(Counter(labels)),
            "training_videos": len(set(groups)),
            "embedding_dimension": int(x.shape[1]),
            "regularization": args.regularization,
            "artifact": str(args.output),
            "manual_labels_take_precedence": True,
            "ignore_semantics": (
                "Explicit rejection class for non-outfield or invalid tracks"
            ),
        }
    )
    write_json_atomic(args.output.with_suffix(".metrics.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
