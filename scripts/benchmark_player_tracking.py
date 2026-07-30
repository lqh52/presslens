#!/usr/bin/env python3
"""Run and compare player detector/tracker combinations on football clips.

The runner deliberately emits a small, backend-neutral JSON format.  Ultralytics
is supported directly; RF-DETR/BoxMOT or other systems can be evaluated by
exporting the same format and using ``--results`` without changing this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
ATHLETE_CATEGORY_IDS = {1, 2}


def manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("clips", "videos", "jobs", "candidates")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
    else:
        rows = None
    if rows is None or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Manifest must be a list or contain clips/videos/jobs/candidates")
    return rows


def resolve_clips(manifest: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = manifest_rows(json.loads(manifest.read_text()))
    clips = []
    for row in rows[:limit]:
        clip_id = row.get("id") or row.get("name")
        raw_path = next(
            (
                row[key]
                for key in ("clip_path", "video_path", "video", "path")
                if row.get(key)
            ),
            None,
        )
        if not clip_id or not raw_path:
            raise ValueError("Every benchmark clip needs id/name and a clip/video path")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (manifest.parent / path).resolve() if not path.exists() else path.resolve()
        clips.append({"id": str(clip_id), "path": path, "source": row})
    return clips


def load_experiments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("experiments") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Experiment file must contain a non-empty experiments list")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("name") or not row.get("model"):
            raise ValueError("Every experiment needs name and model")
        if row["name"] in names:
            raise ValueError(f"Duplicate experiment name: {row['name']}")
        names.add(row["name"])
    return rows


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def greedy_matches(
    truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (box_iou(target["bbox"], prediction["bbox"]), target_index, pred_index)
            for target_index, target in enumerate(truth)
            for pred_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches = []
    for overlap, target_index, pred_index in candidates:
        if overlap < threshold:
            break
        if target_index in matched_truth or pred_index in matched_predictions:
            continue
        matched_truth.add(target_index)
        matched_predictions.add(pred_index)
        matches.append((target_index, pred_index, overlap))
    return matches


def load_soccernet_truth(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    images = {
        str(image["image_id"]): int(Path(image["file_name"]).stem) - 1
        for image in payload["images"]
        if image.get("is_labeled", True)
    }
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if int(annotation.get("category_id", -1)) not in ATHLETE_CATEGORY_IDS:
            continue
        frame = images.get(str(annotation["image_id"]))
        bbox = annotation.get("bbox_image")
        if frame is None or not bbox:
            continue
        frames[frame].append(
            {
                "track_id": int(annotation["track_id"]),
                "bbox": [
                    float(bbox["x"]),
                    float(bbox["y"]),
                    float(bbox["x"] + bbox["w"]),
                    float(bbox["y"] + bbox["h"]),
                ],
            }
        )
    return dict(frames)


def continuity_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(frame["detections"]) for frame in frames]
    lengths: Counter[int] = Counter()
    adjacent = 0
    retained = 0
    previous: set[int] | None = None
    for frame in frames:
        current = {
            int(item["track_id"])
            for item in frame["detections"]
            if item.get("track_id") is not None
        }
        lengths.update(current)
        if previous:
            adjacent += len(previous)
            retained += len(previous & current)
        previous = current
    track_lengths = list(lengths.values())
    return {
        "frames": len(frames),
        "detections": sum(counts),
        "mean_players_per_frame": round(statistics.fmean(counts), 4) if counts else 0.0,
        "median_players_per_frame": round(statistics.median(counts), 4) if counts else 0.0,
        "empty_frame_rate": round(sum(count == 0 for count in counts) / len(counts), 6)
        if counts
        else 1.0,
        "plausible_count_rate": round(
            sum(6 <= count <= 22 for count in counts) / len(counts), 6
        )
        if counts
        else 0.0,
        "unique_tracks": len(lengths),
        "median_track_length": round(statistics.median(track_lengths), 4)
        if track_lengths
        else 0.0,
        "short_track_rate": round(
            sum(length < 5 for length in track_lengths) / len(track_lengths), 6
        )
        if track_lengths
        else 1.0,
        "adjacent_id_retention": round(retained / adjacent, 6) if adjacent else 0.0,
    }


def labelled_metrics(
    frames: list[dict[str, Any]],
    truth: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
) -> dict[str, Any]:
    by_frame = {int(frame["frame"]): frame["detections"] for frame in frames}
    true_positives = false_positives = false_negatives = switches = 0
    overlaps: list[float] = []
    last_prediction_for_truth: dict[int, int] = {}
    for frame_index, targets in sorted(truth.items()):
        predictions = by_frame.get(frame_index, [])
        matches = greedy_matches(targets, predictions, iou_threshold)
        true_positives += len(matches)
        false_positives += len(predictions) - len(matches)
        false_negatives += len(targets) - len(matches)
        for target_index, pred_index, overlap in matches:
            overlaps.append(overlap)
            prediction_id = predictions[pred_index].get("track_id")
            truth_id = int(targets[target_index]["track_id"])
            if prediction_id is None:
                continue
            prediction_id = int(prediction_id)
            if (
                truth_id in last_prediction_for_truth
                and last_prediction_for_truth[truth_id] != prediction_id
            ):
                switches += 1
            last_prediction_for_truth[truth_id] = prediction_id
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    return {
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6)
        if precision + recall
        else 0.0,
        "mean_matched_iou": round(statistics.fmean(overlaps), 6) if overlaps else 0.0,
        "id_switches": switches,
    }


def run_ultralytics_clip(
    model: Any,
    clip: Path,
    experiment: dict[str, Any],
    raw_model: Any | None = None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = model.track(
        source=str(clip),
        stream=True,
        verbose=False,
        classes=[0],
        conf=float(experiment.get("confidence", 0.15)),
        iou=float(experiment.get("iou", 0.7)),
        imgsz=int(experiment.get("imgsz", 1280)),
        tracker=str(experiment.get("tracker", "botsort.yaml")),
    )
    frames = []
    for frame_index, result in enumerate(results):
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confidence = boxes.conf.cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
            for bbox, score, track_id in zip(xyxy, confidence, track_ids):
                detections.append(
                    {
                        "track_id": track_id,
                        "confidence": round(float(score), 6),
                        "bbox": [round(float(value), 3) for value in bbox],
                    }
                )
        frames.append({"frame": frame_index, "detections": detections})
    if experiment.get("preserve_untracked"):
        if raw_model is None:
            raise ValueError("preserve_untracked requires a separate raw detector")
        raw_results = raw_model.predict(
            source=str(clip),
            stream=True,
            verbose=False,
            classes=[0],
            conf=float(experiment.get("confidence", 0.15)),
            iou=float(experiment.get("iou", 0.7)),
            imgsz=int(experiment.get("imgsz", 1280)),
        )
        for frame, raw_result in zip(frames, raw_results):
            raw_boxes = raw_result.boxes
            raw_detections = []
            if raw_boxes is not None:
                for bbox, score in zip(
                    raw_boxes.xyxy.cpu().tolist(),
                    raw_boxes.conf.cpu().tolist(),
                ):
                    raw_detections.append(
                        {
                            "track_id": None,
                            "confidence": round(float(score), 6),
                            "bbox": [round(float(value), 3) for value in bbox],
                            "tracking_status": "untracked",
                        }
                    )
            matches = greedy_matches(
                frame["detections"], raw_detections, threshold=0.5
            )
            for tracked_index, raw_index, _ in matches:
                raw_detections[raw_index]["track_id"] = frame["detections"][
                    tracked_index
                ]["track_id"]
                raw_detections[raw_index]["tracking_status"] = "confirmed"
            frame["detections"] = raw_detections
    return frames, time.perf_counter() - started


def run_benchmark(
    manifest: Path,
    experiments_path: Path,
    output: Path,
    *,
    limit: int | None,
    device: str | None,
    only: set[str] | None,
    skip_existing: bool,
    only_clips: set[str] | None,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
        import ultralytics
    except ImportError as error:
        raise RuntimeError("Run mode requires the research environment with ultralytics") from error

    clips = resolve_clips(manifest, limit)
    if only_clips:
        clips = [clip for clip in clips if clip["id"] in only_clips]
        missing_clips = only_clips - {clip["id"] for clip in clips}
        if missing_clips:
            raise ValueError(f"Unknown clip IDs: {', '.join(sorted(missing_clips))}")
    experiments = load_experiments(experiments_path)
    if only:
        experiments = [row for row in experiments if str(row["name"]) in only]
        missing = only - {str(row["name"]) for row in experiments}
        if missing:
            raise ValueError(f"Unknown experiment names: {', '.join(sorted(missing))}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "ultralytics_version": ultralytics.__version__,
        "experiments": [],
    }
    for experiment in experiments:
        model = YOLO(str(experiment["model"]))
        raw_model = (
            YOLO(str(experiment["model"]))
            if experiment.get("preserve_untracked")
            else None
        )
        if device:
            model.to(f"cuda:{device}" if device.isdigit() else device)
            if raw_model is not None:
                raw_model.to(f"cuda:{device}" if device.isdigit() else device)
        experiment_rows = []
        for clip in clips:
            destination = output / str(experiment["name"]) / f"{clip['id']}.json"
            if skip_existing and destination.is_file():
                payload = json.loads(destination.read_text())
                experiment_rows.append(
                    {
                        "clip_id": clip["id"],
                        "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
                        **continuity_metrics(payload["frames"]),
                    }
                )
                print(f"{experiment['name']}: {clip['id']} (existing)")
                continue
            frames, seconds = run_ultralytics_clip(
                model, clip["path"], experiment, raw_model
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "experiment": experiment,
                "clip_id": clip["id"],
                "clip_path": str(clip["path"]),
                "runtime_seconds": round(seconds, 4),
                "frames": frames,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2) + "\n")
            experiment_rows.append(
                {
                    "clip_id": clip["id"],
                    "runtime_seconds": round(seconds, 4),
                    **continuity_metrics(frames),
                }
            )
            print(f"{experiment['name']}: {clip['id']} ({seconds:.1f}s)")
        summary["experiments"].append(
            {"name": experiment["name"], "configuration": experiment, "clips": experiment_rows}
        )
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    keys = (
        "runtime_seconds",
        "mean_players_per_frame",
        "empty_frame_rate",
        "plausible_count_rate",
        "unique_tracks",
        "median_track_length",
        "short_track_rate",
        "adjacent_id_retention",
    )
    return {
        "clips": len(rows),
        **{
            key: round(statistics.fmean(float(row[key]) for row in rows), 6)
            for key in keys
            if rows and all(key in row for row in rows)
        },
    }


def summarize_results(
    results_root: Path,
    *,
    labels: Path | None,
    iou_threshold: float,
    output: Path | None,
) -> dict[str, Any]:
    truth = load_soccernet_truth(labels) if labels else None
    experiments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(results_root.glob("*/*.json")):
        payload = json.loads(path.read_text())
        if "frames" not in payload or "experiment" not in payload:
            continue
        row = {
            "clip_id": payload.get("clip_id", path.stem),
            "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
            **continuity_metrics(payload["frames"]),
        }
        if truth is not None:
            row["labelled"] = labelled_metrics(payload["frames"], truth, iou_threshold)
        name = str(payload["experiment"].get("name", path.parent.name))
        experiments[name].append(row)
    if not experiments:
        raise ValueError(f"No experiment result JSON files found under {results_root}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "results_root": str(results_root),
        "labels": str(labels) if labels else None,
        "experiments": [],
    }
    for name, rows in experiments.items():
        result = {"name": name, "aggregate": aggregate(rows), "clips": rows}
        if truth is not None:
            labelled = [row["labelled"] for row in rows]
            result["labelled_aggregate"] = {
                key: round(statistics.fmean(float(item[key]) for item in labelled), 6)
                for key in ("precision", "recall", "f1", "mean_matched_iou", "id_switches")
            }
        report["experiments"].append(result)
    report["experiments"].sort(
        key=lambda item: (
            -float(item.get("labelled_aggregate", {}).get("f1", -1)),
            -float(item["aggregate"].get("adjacent_id_retention", 0)),
        )
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        print(rendered, end="")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run Ultralytics experiments")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--experiments", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--device")
    run.add_argument(
        "--only",
        action="append",
        help="Run only this experiment name; repeat to select multiple",
    )
    run.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse result JSON already present for a clip and experiment",
    )
    run.add_argument(
        "--only-clip",
        action="append",
        help="Run only this clip ID; repeat to select multiple",
    )

    summarize = subparsers.add_parser("summarize", help="Score existing result JSON")
    summarize.add_argument("--results", type=Path, required=True)
    summarize.add_argument("--labels", type=Path)
    summarize.add_argument("--iou-threshold", type=float, default=0.5)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(
            args.manifest,
            args.experiments,
            args.output,
            limit=args.limit,
            device=args.device,
            only=set(args.only) if args.only else None,
            skip_existing=args.skip_existing,
            only_clips=set(args.only_clip) if args.only_clip else None,
        )
    else:
        summarize_results(
            args.results,
            labels=args.labels,
            iou_threshold=args.iou_threshold,
            output=args.output,
        )


if __name__ == "__main__":
    main()
