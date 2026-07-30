#!/usr/bin/env python3
"""Core helpers for auditable, track-level Gemini identity labelling."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SCHEMA_VERSION = 1
ALLOWED_LABELS = {
    "team_a",
    "team_b",
    "other",
    "team_a_goalkeeper",
    "team_b_goalkeeper",
}
AGENT_LABEL_SCHEMA = {
    "type": "object",
    "required": [
        "participant_type",
        "role",
        "label",
        "abstain",
        "kit_visible",
        "identity_visible",
        "official_evidence_visible",
        "goalkeeper_seed_available",
        "goalkeeper_kit_match",
        "matched_seed_images",
        "consistent_crop_count",
        "primary_visual_cues",
        "contradicting_evidence",
        "reason",
    ],
    "additionalProperties": False,
    "properties": {
        "participant_type": {
            "type": "string",
            "enum": ["team_a", "team_b", "other", "unknown"],
        },
        "role": {
            "type": "string",
            "enum": ["outfield", "goalkeeper", "other", "unknown"],
        },
        "label": {
            "type": "string",
            "enum": sorted(ALLOWED_LABELS | {"unknown"}),
        },
        "abstain": {"type": "boolean"},
        "kit_visible": {"type": "boolean"},
        "identity_visible": {"type": "boolean"},
        "official_evidence_visible": {"type": "boolean"},
        "goalkeeper_seed_available": {"type": "boolean"},
        "goalkeeper_kit_match": {"type": "boolean"},
        "matched_seed_images": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": 8,
        },
        "consistent_crop_count": {"type": "integer", "minimum": 0, "maximum": 12},
        "primary_visual_cues": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "contradicting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "reason": {"type": "string"},
    },
}


def fixture_id(clip_id: str) -> str:
    return "-".join(clip_id.replace("-published", "").split("-")[:3])


def canonical_clip_id(clip_id: str) -> str:
    return clip_id.removesuffix("-published")


def load_dotenv_key(path: Path, names: Iterable[str]) -> str | None:
    """Read one key without mutating the process environment."""

    wanted = set(names)
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() not in wanted:
                continue
            clean = value.strip().strip("\"'")
            if clean:
                return clean
    return next((os.environ[name] for name in wanted if os.environ.get(name)), None)


def load_reviewed_labels(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{path} does not contain a labels object")
    output = {}
    for key, row in labels.items():
        label = row.get("label")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported reviewed label {label!r} at {key}")
        output[key] = row
    return output


def load_result_paths(results_dir: Path) -> list[Path]:
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No tracking result JSON found in {results_dir}")
    return paths


def detections_by_track(
    frames: list[dict[str, Any]],
    *,
    minimum_confidence: float = 0.0,
    minimum_detections: int = 1,
) -> dict[int, list[dict[str, Any]]]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        frame_index = int(frame["frame"])
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is None or float(detection.get("confidence", 0.0)) <= minimum_confidence:
                continue
            tracks[int(track_id)].append({**detection, "frame": frame_index})
    return {
        track_id: rows
        for track_id, rows in tracks.items()
        if len(rows) >= minimum_detections
    }


def track_quality(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    areas = [
        max(0.0, row["bbox"][2] - row["bbox"][0])
        * max(0.0, row["bbox"][3] - row["bbox"][1])
        for row in rows
    ]
    confidences = [float(row.get("confidence", 0.0)) for row in rows]
    return math.log1p(len(rows)) * math.sqrt(max(areas)) * (
        sum(confidences) / len(confidences)
    )


def select_representatives(
    rows: list[dict[str, Any]], count: int = 6
) -> list[dict[str, Any]]:
    """Select quality-weighted samples across the complete track lifetime."""

    if len(rows) <= count:
        return sorted(rows, key=lambda row: row["frame"])
    ordered = sorted(rows, key=lambda row: row["frame"])
    selected = []
    for bucket in np.array_split(np.asarray(ordered, dtype=object), count):
        selected.append(
            max(
                bucket.tolist(),
                key=lambda row: (
                    (row["bbox"][2] - row["bbox"][0])
                    * (row["bbox"][3] - row["bbox"][1])
                    * max(float(row.get("confidence", 0.0)), 0.05)
                ),
            )
        )
    return selected


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _read_frames(video: Path, indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {video}")
    output = {}
    for index in sorted(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if ok:
            output[index] = image
    capture.release()
    return output


def render_track_evidence(
    video: Path,
    rows: list[dict[str, Any]],
    output: Path,
    *,
    clip_id: str,
    track_id: int,
) -> dict[str, Any]:
    representatives = select_representatives(rows)
    images = _read_frames(video, {row["frame"] for row in representatives})
    crop_width, crop_height = 180, 210
    width = crop_width * 4
    canvas = np.full((crop_height * 4 + 230, width, 3), 20, dtype=np.uint8)
    crop_levels = (
        ("TIGHT", 0.12, 0.08),
        ("WIDE", 0.75, 0.35),
    )
    for level_index, (level, horizontal_pad, vertical_pad) in enumerate(crop_levels):
        for index, row in enumerate(representatives):
            image = images.get(row["frame"])
            if image is None:
                continue
            left, top, right, bottom = row["bbox"]
            pad_x = horizontal_pad * (right - left)
            pad_y = vertical_pad * (bottom - top)
            x1 = max(0, int(left - pad_x))
            y1 = max(0, int(top - pad_y))
            x2 = min(image.shape[1], int(right + pad_x))
            y2 = min(image.shape[0], int(bottom + pad_y))
            cell = np.full((crop_height, crop_width, 3), 24, dtype=np.uint8)
            cell[: crop_height - 24] = _letterbox(
                image[y1:y2, x1:x2], crop_width, crop_height - 24
            )
            cv2.putText(
                cell,
                f"{level} frame {row['frame']}",
                (6, crop_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            row_y, column = divmod(index, 4)
            y = (level_index * 2 + row_y) * crop_height
            x = column * crop_width
            canvas[y : y + crop_height, x : x + crop_width] = cell

    context_rows = (
        [representatives[0], representatives[-1]]
        if len(representatives) > 1
        else representatives * 2
    )
    for index, row in enumerate(context_rows):
        image = images.get(row["frame"])
        if image is None:
            continue
        context = image.copy()
        left, top, right, bottom = map(int, row["bbox"])
        cv2.rectangle(context, (left, top), (right, bottom), (0, 0, 255), 4)
        cv2.putText(
            context,
            f"TARGET {track_id}",
            (max(4, left), max(22, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cell = _letterbox(context, width // 2, 205)
        canvas[crop_height * 4 : crop_height * 4 + 205, index * width // 2 : (index + 1) * width // 2] = cell

    cv2.putText(
        canvas,
        f"{clip_id}  track {track_id}  {len(rows)} detections",
        (8, canvas.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not write {output}")
    return {
        "representative_frames": [int(row["frame"]) for row in representatives],
        "detections": len(rows),
        "first_frame": min(int(row["frame"]) for row in rows),
        "last_frame": max(int(row["frame"]) for row in rows),
        "quality": round(track_quality(rows), 6),
    }


def assign_seed_splits(
    tracks: list[dict[str, Any]], seed_per_label: int
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        if row.get("reviewed_label"):
            grouped[(row["fixture_id"], row["reviewed_label"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["quality"], row["key"]))
        count = min(seed_per_label, len(rows) - 1) if len(rows) > 1 else 1
        for index, row in enumerate(rows):
            row["split"] = "seed" if index < count else "evaluation"


def build_evidence_manifest(
    results_dir: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    seed_per_label: int = 2,
    minimum_confidence: float = 0.0,
    minimum_detections: int = 1,
) -> dict[str, Any]:
    labels = load_reviewed_labels(labels_path)
    tracks = []
    for result_path in load_result_paths(results_dir):
        payload = json.loads(result_path.read_text())
        raw_clip_id = str(payload["clip_id"])
        clip_id = canonical_clip_id(raw_clip_id)
        video = Path(payload["clip_path"])
        if not video.is_absolute():
            video = (result_path.parent / video).resolve()
        by_track = detections_by_track(
            payload["frames"],
            minimum_confidence=minimum_confidence,
            minimum_detections=minimum_detections,
        )
        for track_id, rows in sorted(by_track.items()):
            key = f"{clip_id}:{track_id}"
            reviewed = labels.get(key)
            relative_image = Path("images") / clip_id / f"track-{track_id}.jpg"
            evidence = render_track_evidence(
                video,
                rows,
                output_dir / relative_image,
                clip_id=clip_id,
                track_id=track_id,
            )
            tracks.append(
                {
                    "key": key,
                    "fixture_id": fixture_id(clip_id),
                    "clip_id": clip_id,
                    "source_clip_id": raw_clip_id,
                    "track_id": track_id,
                    "source_result": str(result_path.resolve()),
                    "source_video": str(video.resolve()),
                    "evidence_image": relative_image.as_posix(),
                    "reviewed_label": reviewed["label"] if reviewed else None,
                    "split": "unreviewed",
                    **evidence,
                }
            )
    assign_seed_splits(tracks, seed_per_label)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "track_contact_sheet_tight_wide_v2",
        "source_results": str(results_dir.resolve()),
        "source_labels": str(labels_path.resolve()),
        "seed_per_label": seed_per_label,
        "minimum_confidence": minimum_confidence,
        "minimum_detections": minimum_detections,
        "counts": dict(Counter(row["split"] for row in tracks)),
        "tracks": tracks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prompt_for_track(target: dict[str, Any], seeds: list[dict[str, Any]]) -> str:
    examples = "\n".join(
        f"- preceding image {index + 1}: {row['reviewed_label']}"
        for index, row in enumerate(seeds)
    )
    available = sorted({row["reviewed_label"] for row in seeds})
    return f"""You classify one tracked person in association-football broadcast images.

INPUT ORDER
- The first {len(seeds)} images are human-reviewed examples from the same fixture.
- The final image is TARGET track {target['track_id']} from clip {target['clip_id']}.
- Each image contains tight crops, wider crops, and full-frame views.
- In full-frame views, classify only the person inside the red box.

REVIEWED EXAMPLES
{examples}
Labels represented by reviewed examples: {", ".join(available)}

STEP 1 — PARTICIPANT TYPE
First decide team_a, team_b, other, or unknown. Compare shirt, shorts, and socks
against reviewed examples using evidence consistent across multiple frames.
Use other only with positive evidence of a referee, assistant referee, staff,
spectator, or non-player. Being near a touchline is not sufficient. A dark or
blurred shirt is not sufficient. Assistant-referee evidence should include an
official uniform, visible flag, or repeated movement outside the playing boundary.
If blur, occlusion, lighting, or insufficient pixels prevent reliable comparison,
use unknown. Do not invent colours that are not consistent across crops.

STEP 2 — ROLE
Only after choosing team_a or team_b, decide outfield, goalkeeper, or unknown.
Goal position or penalty-area position alone is not sufficient. Choose goalkeeper
only if a reviewed goalkeeper example for the same team is present, the distinctive
kit visibly matches it in at least two target crops, and context is compatible.
If no same-team goalkeeper seed exists, goalkeeper is forbidden: choose that
team's outfield label or unknown.

CONSTRAINTS
- team_a/team_b are neutral fixture-local identities; do not guess club names.
- Do not use the scoreboard, goal side, or assumed attacking direction.
- Tight crops supply kit evidence; wider crops supply context.
- A single blurry crop must not override clearer crops.
- Prefer a supported outfield label over an unsupported goalkeeper label.
- matched_seed_images uses the 1-based reviewed image numbers above.
- Set official_evidence_visible=true only for positive official/non-player evidence.
- Set goalkeeper_seed_available from the reviewed labels, not from inference.
- Return unknown rather than making an unsupported inference.

FINAL MAPPING
team_a+outfield=team_a; team_b+outfield=team_b;
team_a+goalkeeper=team_a_goalkeeper; team_b+goalkeeper=team_b_goalkeeper;
other=other; insufficient evidence=unknown.

Return only the required structured JSON."""


def image_part(path: Path) -> dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(path.read_bytes()).decode(),
        }
    }


def request_fingerprint(
    model: str, target: dict[str, Any], seeds: list[dict[str, Any]], root: Path
) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(prompt_for_track(target, seeds).encode())
    for row in [*seeds, target]:
        digest.update((root / row["evidence_image"]).read_bytes())
    return digest.hexdigest()


def validate_agent_label(
    payload: Any, available_seed_labels: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be an object")
    required = set(AGENT_LABEL_SCHEMA["required"])
    if not required.issubset(payload):
        raise ValueError(f"Agent response misses {sorted(required - set(payload))}")
    if payload["label"] not in ALLOWED_LABELS | {"unknown"}:
        raise ValueError(f"Invalid label {payload['label']!r}")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("abstain must be boolean")
    if payload["abstain"] and payload["label"] != "unknown":
        raise ValueError("An abstention must use the unknown label")
    expected = {
        ("team_a", "outfield"): "team_a",
        ("team_b", "outfield"): "team_b",
        ("team_a", "goalkeeper"): "team_a_goalkeeper",
        ("team_b", "goalkeeper"): "team_b_goalkeeper",
        ("other", "other"): "other",
        ("unknown", "unknown"): "unknown",
    }.get((payload["participant_type"], payload["role"]))
    if expected != payload["label"]:
        raise ValueError("participant_type, role, and label are inconsistent")
    if payload["label"] == "other" and not payload["official_evidence_visible"]:
        raise ValueError("other requires positive official/non-participant evidence")
    if payload["role"] == "goalkeeper":
        seed_label = f"{payload['participant_type']}_goalkeeper"
        if available_seed_labels is not None and seed_label not in available_seed_labels:
            raise ValueError(f"goalkeeper label is forbidden without seed {seed_label}")
        if not payload["goalkeeper_seed_available"] or not payload["goalkeeper_kit_match"]:
            raise ValueError("goalkeeper requires an available matching goalkeeper seed")
    return payload


def reconcile_manifest(
    evidence_manifest: Path, predictions_path: Path, output_dir: Path
) -> dict[str, Any]:
    evidence = json.loads(evidence_manifest.read_text())
    predictions = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                predictions[row["key"]] = row
    decisions = {}
    for row in evidence["tracks"]:
        if row["split"] == "seed" and row.get("reviewed_label"):
            label, source, status = row["reviewed_label"], "human_seed", "accepted"
        elif row["key"] in predictions:
            prediction = predictions[row["key"]]["response"]
            label = prediction["label"]
            source = "gemini"
            status = "needs_review" if label == "unknown" else "agent_proposal"
        else:
            label, source, status = "unknown", "missing", "needs_review"
        decisions[row["key"]] = {
            "clip_id": row["clip_id"],
            "track_id": row["track_id"],
            "label": label,
            "source": source,
            "status": status,
            "split": row["split"],
        }

    by_result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence["tracks"]:
        by_result[row["source_result"]].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_outputs = []
    for source_result, rows in by_result.items():
        payload = json.loads(Path(source_result).read_text())
        clip_id = rows[0]["clip_id"]
        labels = {
            int(row["track_id"]): decisions[row["key"]]
            for row in rows
        }
        frames = []
        for frame in payload["frames"]:
            detections = []
            for detection in frame.get("detections", []):
                item = dict(detection)
                track_id = item.get("track_id")
                decision = labels.get(int(track_id)) if track_id is not None else None
                item["identity"] = (
                    {
                        "label": decision["label"],
                        "source": decision["source"],
                        "status": decision["status"],
                    }
                    if decision
                    else {"label": "unknown", "source": "untracked", "status": "needs_review"}
                )
                detections.append(item)
            frames.append({"frame": int(frame["frame"]), "detections": detections})
        result = {
            "schema_version": SCHEMA_VERSION,
            "clip_id": clip_id,
            "fixture_id": fixture_id(clip_id),
            "source_result": source_result,
            "tracks": sorted(labels.values(), key=lambda item: item["track_id"]),
            "frames": frames,
        }
        path = output_dir / f"{clip_id}.json"
        path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        clip_outputs.append(str(path))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "counts": dict(Counter(row["status"] for row in decisions.values())),
        "clips": clip_outputs,
        "decisions": decisions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_predictions(
    evidence_manifest: Path, predictions_path: Path
) -> dict[str, Any]:
    evidence = json.loads(evidence_manifest.read_text())
    truth = {
        row["key"]: row
        for row in evidence["tracks"]
        if row["split"] == "evaluation" and row.get("reviewed_label")
    }
    predictions = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                predictions[row["key"]] = row["response"]["label"]
    covered = {key: value for key, value in predictions.items() if key in truth and value != "unknown"}
    correct = sum(covered[key] == truth[key]["reviewed_label"] for key in covered)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for key, row in truth.items():
        confusion[row["reviewed_label"]][predictions.get(key, "missing")] += 1
    return {
        "evaluation_tracks": len(truth),
        "predicted_tracks": len(covered),
        "coverage": round(len(covered) / len(truth), 6) if truth else 0.0,
        "accuracy_at_coverage": round(correct / len(covered), 6) if covered else 0.0,
        "overall_accuracy": round(correct / len(truth), 6) if truth else 0.0,
        "confusion": {
            label: dict(values) for label, values in sorted(confusion.items())
        },
    }
