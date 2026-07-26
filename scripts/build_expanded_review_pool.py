#!/usr/bin/env python3
"""Build a local, match-neutral tactical-video review pool.

The builder consumes the clip manifest used by ``run_gsr_batch.py`` together
with one reconstructed graph artifact, prediction JSONL, and TrackLab state per
clip.  It deliberately preflights the complete input set before opening an
encoder, so an incomplete GSR/classification run cannot leave a misleading
partial review pool.

Run this script with the isolated ``sn-gamestate`` Python environment because
TrackLab states contain pandas objects and the renderer uses OpenCV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from team_identity import infer_nonparticipant_track_ids
except ModuleNotFoundError:
    from scripts.team_identity import infer_nonparticipant_track_ids


TACTICAL_LABELS = (
    "high_press",
    "trap_left",
    "trap_right",
    "central_screen",
    "unstructured",
)
TARGET_LABELS = ("high_press", "trap_left", "trap_right")
FALLBACK_LABELS = ("central_screen", "unstructured")
DIRECTION_DEPENDENT_LABELS = frozenset(
    {"high_press", "trap_left", "trap_right", "central_screen"}
)
LABEL_TITLES = {
    "high_press": "High press",
    "trap_left": "Left touchline trap (attacker-relative)",
    "trap_right": "Right touchline trap (attacker-relative)",
    "central_screen": "Central screen",
    "unstructured": "Unstructured",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TEAM_COLOURS = {
    0: (255, 190, 40),  # BGR: cyan-blue
    1: (55, 90, 255),  # BGR: coral-red
}
NONTERMINAL_DOWNSTREAM_STATUSES = {
    "waiting",
    "waiting_for_state",
    "queued",
    "pending",
    "running",
    "processing",
    "in_progress",
    "unknown",
}


def _nonterminal_downstream_status(status: str) -> bool:
    return status in NONTERMINAL_DOWNSTREAM_STATUSES or any(
        marker in status
        for marker in ("waiting", "running", "queued", "pending", "in_progress")
    )


class ArtifactError(RuntimeError):
    """Raised when a review pool would be built from incomplete artifacts."""


@dataclass(frozen=True)
class ClipArtifacts:
    item: dict[str, Any]
    clip_path: Path
    graph_path: Path
    predictions_path: Path
    state_path: Path

    @property
    def id(self) -> str:
        return str(self.item["id"])


@dataclass(frozen=True)
class PredictionSummary:
    model_label: str
    classification_confidence: float
    temporal_agreement: float
    majority_frames: int
    valid_graph_frames: int
    representative_index: int
    representative_frame: int
    window_start_frame: int
    window_end_frame: int
    direction_usable: bool
    direction_confidence: float | None
    direction_sources: tuple[str, ...]
    direction_statuses: tuple[str, ...]
    direction_evidence: dict[str, Any]

    @property
    def quality(self) -> float:
        """Selection quality, separate from the reported model probability."""

        graph_support = min(self.valid_graph_frames, 10) / 10
        return (
            0.55 * self.classification_confidence
            + 0.35 * self.temporal_agreement
            + 0.10 * graph_support
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactError(
                f"{path}:{line_number}: invalid prediction JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ArtifactError(f"{path}:{line_number}: prediction must be an object")
        rows.append(row)
    if not rows:
        raise ArtifactError(f"{path}: prediction file is empty")
    return rows


def _finite_probability(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{context}: expected a numeric probability") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ArtifactError(f"{context}: probability must be between 0 and 1")
    return result


def summarize_predictions(
    predictions: list[dict[str, Any]],
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    require_possession_confident: bool = False,
) -> PredictionSummary:
    """Return one majority class and a video-level probability summary.

    ``classification_confidence`` is the mean model probability assigned to
    the majority class across possession-reliable graph frames.  It is not the
    temporal vote share and it is unrelated to the video-text retrieval score.
    """

    if not predictions:
        raise ArtifactError("Cannot summarize an empty prediction sequence")
    indexed = [
        (index, row)
        for index, row in enumerate(predictions)
        if (
            (start_frame is None or int(row.get("frame", -1)) >= start_frame)
            and (end_frame is None or int(row.get("frame", -1)) < end_frame)
        )
    ]
    if not indexed:
        raise ArtifactError("Temporal window contains no graph predictions")
    for index, row in indexed:
        label = row.get("predicted_situation")
        if label not in TACTICAL_LABELS:
            raise ArtifactError(
                f"Prediction {index} has unsupported tactical label {label!r}"
            )
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or label not in probabilities:
            raise ArtifactError(
                f"Prediction {index} lacks a complete probabilities object"
            )
        for tactical_label in TACTICAL_LABELS:
            if tactical_label not in probabilities:
                raise ArtifactError(
                    f"Prediction {index} lacks probability for {tactical_label!r}"
                )
            _finite_probability(
                probabilities[tactical_label],
                context=f"Prediction {index} / {tactical_label}",
            )
        try:
            int(row["frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"Prediction {index} has no integer frame") from error
    eligible = [
        (index, row)
        for index, row in indexed
        if bool(row.get("possession_confident"))
    ]
    if not eligible:
        if require_possession_confident:
            raise ArtifactError(
                "Temporal window has no possession-confident graph predictions"
            )
        eligible = indexed

    labels = [str(row["predicted_situation"]) for _, row in eligible]
    counts = Counter(labels)
    # The explicit class order makes ties reproducible across Python versions.
    model_label = max(
        TACTICAL_LABELS,
        key=lambda label: (counts[label], -TACTICAL_LABELS.index(label)),
    )
    majority_frames = counts[model_label]
    class_probabilities = [
        _finite_probability(
            row["probabilities"][model_label],
            context=f"Prediction {index} / {model_label}",
        )
        for index, row in eligible
    ]
    confidence = sum(class_probabilities) / len(class_probabilities)
    representative_index, representative = max(
        (
            (index, row)
            for index, row in eligible
            if row["predicted_situation"] == model_label
        ),
        key=lambda pair: (
            float(pair[1]["probabilities"][model_label]),
            -abs(pair[0] - len(predictions) / 2),
        ),
    )
    (
        direction_usable,
        direction_confidence,
        direction_sources,
        direction_statuses,
        direction_evidence,
    ) = summarize_direction_evidence([row for _, row in eligible])
    return PredictionSummary(
        model_label=model_label,
        classification_confidence=confidence,
        temporal_agreement=majority_frames / len(eligible),
        majority_frames=majority_frames,
        valid_graph_frames=len(eligible),
        representative_index=representative_index,
        representative_frame=int(representative["frame"]),
        window_start_frame=(
            int(start_frame)
            if start_frame is not None
            else min(int(row["frame"]) for _, row in indexed)
        ),
        window_end_frame=(
            int(end_frame)
            if end_frame is not None
            else max(int(row["frame"]) for _, row in indexed) + 1
        ),
        direction_usable=direction_usable,
        direction_confidence=direction_confidence,
        direction_sources=direction_sources,
        direction_statuses=direction_statuses,
        direction_evidence=direction_evidence,
    )


def _bad_direction_marker(value: Any) -> bool:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return any(
        marker in normalized
        for marker in (
            "ambiguous",
            "abstain",
            "invalid",
            "unknown",
            "undetermined",
            "unavailable",
        )
    )


def _row_direction_confidence(row: dict[str, Any]) -> float | None:
    """Return one internally consistent direction confidence.

    New graph rows expose ``direction_confidence`` directly. Older rows retain
    the same value inside ``direction_evidence``; accepting that form keeps
    already converted graphs reviewable without weakening validation.
    """

    candidates: list[Any] = []
    if "direction_confidence" in row:
        candidates.append(row["direction_confidence"])
    evidence = row.get("direction_evidence")
    if isinstance(evidence, dict) and "confidence" in evidence:
        candidates.append(evidence["confidence"])
    if not candidates:
        return None
    normalized = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        confidence = float(value)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
        normalized.append(confidence)
    if max(normalized) - min(normalized) > 1e-9:
        return None
    return min(normalized)


def _direction_row_usable(row: dict[str, Any]) -> bool:
    # Trust must be explicit. A plausible raw direction/label pair is not
    # sufficient provenance when calibration confidence is missing or mistyped.
    if row.get("direction_confident") is not True:
        return False
    if _row_direction_confidence(row) is None:
        return False
    for key in (
        "direction_valid",
        "orientation_valid",
    ):
        if key in row and row[key] is not None and not bool(row[key]):
            return False
    for key in ("direction_status", "orientation_status", "direction_source"):
        if key in row and row[key] is not None and _bad_direction_marker(row[key]):
            return False
    raw = row.get("attacking_direction_raw")
    try:
        raw_valid = int(raw) in (-1, 1) and not isinstance(raw, bool)
    except (TypeError, ValueError):
        raw_valid = False
    label = str(row.get("attacking_direction_label", "")).strip().lower()
    expected_label = (
        "left_to_right"
        if raw_valid and int(raw) == 1
        else "right_to_left"
    )
    return raw_valid and label == expected_label


def summarize_direction_evidence(
    rows: list[dict[str, Any]],
) -> tuple[
    bool,
    float | None,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    sources = tuple(
        sorted(
            {
                str(row["direction_source"])
                for row in rows
                if row.get("direction_source") is not None
            }
        )
    )
    statuses = tuple(
        sorted(
            {
                str(row[key])
                for row in rows
                for key in ("direction_status", "orientation_status")
                if row.get(key) is not None
            }
        )
    )
    raw_values = sorted(
        {
            int(row["attacking_direction_raw"])
            for row in rows
            if row.get("attacking_direction_raw") in (-1, 1)
        }
    )
    labels = sorted(
        {
            str(row["attacking_direction_label"])
            for row in rows
            if row.get("attacking_direction_label") is not None
        }
    )
    explicit = []
    for row in rows:
        value = row.get("direction_evidence")
        if isinstance(value, dict) and value not in explicit:
            explicit.append(value)
    confidence_values = [
        confidence
        for row in rows
        if (confidence := _row_direction_confidence(row)) is not None
    ]
    usable_rows = sum(_direction_row_usable(row) for row in rows)
    direction_confidence = (
        min(confidence_values)
        if len(confidence_values) == len(rows)
        else None
    )
    evidence = {
        "usable_rows": usable_rows,
        "eligible_rows": len(rows),
        "confidence_aggregation": "minimum_across_eligible_rows",
        "direction_confidence": direction_confidence,
        "raw_directions": raw_values,
        "direction_labels": labels,
        "explicit": explicit,
    }
    return (
        usable_rows == len(rows),
        direction_confidence,
        sources,
        statuses,
        evidence,
    )


def temporal_window_proposals(
    predictions: list[dict[str, Any]],
    *,
    nframes: int,
    span_frames: int,
    stride_frames: int,
    min_graph_frames: int,
) -> dict[str, PredictionSummary]:
    """Find the strongest four-second majority window offered by each class."""

    if nframes <= 0 or span_frames <= 0 or stride_frames <= 0:
        raise ArtifactError("Temporal-window dimensions must be positive")
    span_frames = min(span_frames, nframes)
    final_start = max(0, nframes - span_frames)
    starts = list(range(0, final_start + 1, stride_frames))
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    proposals: dict[str, PredictionSummary] = {}
    for start in starts:
        end = start + span_frames
        try:
            summary = summarize_predictions(
                predictions,
                start_frame=start,
                end_frame=end,
                require_possession_confident=True,
            )
        except ArtifactError as error:
            if (
                "contains no graph predictions" in str(error)
                or "no possession-confident" in str(error)
            ):
                continue
            raise
        if summary.valid_graph_frames < min_graph_frames:
            continue
        if (
            summary.model_label in DIRECTION_DEPENDENT_LABELS
            and not summary.direction_usable
        ):
            # High press and central screen use canonical ball-x/ahead
            # descriptors, while touchline traps additionally use canonical
            # left/right. All therefore require a trusted attacker-relative
            # rotation. Only the distance-based unstructured fallback is
            # meaningful after direction abstention.
            continue
        current = proposals.get(summary.model_label)
        if current is None or (
            summary.quality,
            summary.valid_graph_frames,
            -abs(
                summary.representative_frame
                - (summary.window_start_frame + summary.window_end_frame) / 2
            ),
            -summary.window_start_frame,
        ) > (
            current.quality,
            current.valid_graph_frames,
            -abs(
                current.representative_frame
                - (current.window_start_frame + current.window_end_frame) / 2
            ),
            -current.window_start_frame,
        ):
            proposals[summary.model_label] = summary
    return proposals


def _workspace_path(value: str | os.PathLike[str], workspace: Path) -> Path:
    path = Path(value)
    return (workspace / path).resolve() if not path.is_absolute() else path.resolve()


def latest_state(state_root: Path, clip_id: str) -> Path | None:
    """Resolve the newest exact TrackLab state for one manifest clip ID."""

    candidates = [
        path
        for path in state_root.glob(f"{clip_id}/**/states/{clip_id}.pklz")
        if path.is_file() and path.stat().st_size
    ]
    if not candidates:
        candidates = [
            path
            for path in state_root.glob(f"**/states/{clip_id}.pklz")
            if path.is_file() and path.stat().st_size
        ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def completed_downstream_rows(
    manifest_path: Path,
    status_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return terminal completed rows and an auditable exclusion report.

    Missing/nonterminal statuses block the review build. Terminal conversion
    failures do not: a clip with no detected ball or classifiable graph is a
    legitimate exclusion, provided the downstream batch has finished deciding
    that outcome.
    """

    try:
        manifest = json.loads(manifest_path.read_text())
        status_payload = json.loads(status_path.read_text())
    except FileNotFoundError as error:
        raise ArtifactError(
            f"Downstream status is not ready: {error.filename}"
        ) from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid downstream JSON: {error}") from error
    clips = manifest.get("clips") if isinstance(manifest, dict) else None
    statuses = (
        status_payload.get("clips")
        if isinstance(status_payload, dict)
        else None
    )
    if not isinstance(clips, list) or not clips:
        raise ArtifactError(f"{manifest_path}: expected a non-empty clips array")
    if not isinstance(statuses, dict):
        raise ArtifactError(f"{status_path}: expected a clips status object")
    status_manifest = status_payload.get("manifest")
    if status_manifest and Path(status_manifest).resolve() != manifest_path.resolve():
        raise ArtifactError(
            f"{status_path}: status belongs to a different source manifest "
            f"({status_manifest})"
        )

    completed: dict[str, dict[str, Any]] = {}
    excluded = []
    nonterminal = []
    seen = set()
    for item in clips:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ArtifactError(f"{manifest_path}: every clip needs a string id")
        clip_id = item["id"]
        if clip_id in seen:
            raise ArtifactError(f"{manifest_path}: duplicate clip id {clip_id!r}")
        seen.add(clip_id)
        row = statuses.get(clip_id)
        if not isinstance(row, dict):
            nonterminal.append(
                {"id": clip_id, "status": "missing_status", "reason": "No status row"}
            )
            continue
        status = str(row.get("status", "unknown")).strip().lower()
        if status == "completed":
            completed[clip_id] = row
            continue
        reason = str(
            row.get("error")
            or row.get("reason")
            or row.get("failed_stage")
            or status
        )
        record = {
            "id": clip_id,
            "status": status,
            "reason": reason,
        }
        if row.get("failed_stage") is not None:
            record["failed_stage"] = row["failed_stage"]
        if _nonterminal_downstream_status(status):
            nonterminal.append(record)
        else:
            excluded.append(record)
    if nonterminal:
        preview = "; ".join(
            f"{row['id']}={row['status']}" for row in nonterminal[:12]
        )
        remainder = (
            f"; +{len(nonterminal)-12} more" if len(nonterminal) > 12 else ""
        )
        raise ArtifactError(
            "Downstream batch is not terminal; no videos were rendered: "
            f"{preview}{remainder}"
        )
    if not completed:
        raise ArtifactError(
            "Downstream batch is terminal but has no completed clips; no "
            "videos were rendered"
        )
    failed_count = sum(row["status"] == "failed" for row in excluded)
    report = {
        "status_path": str(status_path.resolve()),
        "total_manifest_clips": len(clips),
        "completed_count": len(completed),
        "failed_count": failed_count,
        "excluded_count": len(excluded) - failed_count,
        "terminal_exclusions": excluded,
    }
    return completed, report


def resolve_artifacts(
    manifest_path: Path,
    graph_dir: Path,
    state_root: Path,
    workspace: Path,
    downstream_rows: dict[str, dict[str, Any]] | None = None,
) -> list[ClipArtifacts]:
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read clip manifest {manifest_path}: {error}") from error
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list) or not clips:
        raise ArtifactError(f"{manifest_path}: expected a non-empty clips array")
    seen: set[str] = set()
    resolved = []
    missing: list[str] = []
    for index, item in enumerate(clips):
        if not isinstance(item, dict):
            raise ArtifactError(f"{manifest_path}: clips[{index}] must be an object")
        clip_id = str(item.get("id", ""))
        if not SAFE_ID.fullmatch(clip_id):
            raise ArtifactError(
                f"{manifest_path}: clips[{index}] has unsafe id {clip_id!r}"
            )
        if clip_id in seen:
            raise ArtifactError(f"{manifest_path}: duplicate clip id {clip_id!r}")
        seen.add(clip_id)
        if downstream_rows is not None and clip_id not in downstream_rows:
            continue
        downstream = downstream_rows.get(clip_id, {}) if downstream_rows else {}
        clip_value = downstream.get("clip_path", item.get("clip_path"))
        if not clip_value:
            raise ArtifactError(f"{manifest_path}: {clip_id} has no clip_path")
        clip_path = _workspace_path(clip_value, workspace)
        graph_path = _workspace_path(
            downstream.get(
                "graph_path",
                item.get("graph_path", graph_dir / f"{clip_id}.npz"),
            ),
            workspace,
        )
        predictions_path = _workspace_path(
            downstream.get(
                "predictions_path",
                item.get(
                    "predictions_path",
                    graph_dir / f"{clip_id}-predictions.jsonl",
                ),
            ),
            workspace,
        )
        state_value = downstream.get("state_path", item.get("state_path"))
        if state_value:
            state_path = _workspace_path(state_value, workspace)
        else:
            state_path = latest_state(state_root, clip_id)
        paths = {
            "source clip": clip_path,
            "graph": graph_path,
            "predictions": predictions_path,
            "TrackLab state": state_path,
        }
        for name, path in paths.items():
            if path is None or not path.is_file() or not path.stat().st_size:
                missing.append(f"{clip_id}: {name} ({path or 'not found'})")
        if state_path is None:
            # A placeholder keeps type checkers happy; missing artifacts are
            # reported together before any rendering starts.
            state_path = state_root / clip_id / "states" / f"{clip_id}.pklz"
        resolved.append(
            ClipArtifacts(
                item=item,
                clip_path=clip_path,
                graph_path=graph_path,
                predictions_path=predictions_path,
                state_path=state_path.resolve(),
            )
        )
    if missing:
        raise ArtifactError(
            "Review pool preflight failed; no videos were rendered:\n- "
            + "\n- ".join(missing)
        )
    return resolved


def preflight_artifacts(
    artifacts: Iterable[ClipArtifacts],
) -> dict[str, list[dict[str, Any]]]:
    """Validate every state/graph/prediction set before returning any work."""

    # Heavy dependencies are intentionally lazy so ``--help`` and manifest
    # diagnostics still work outside the TrackLab environment.
    import numpy as np

    prepared = {}
    problems = []
    for artifact in artifacts:
        try:
            with zipfile.ZipFile(artifact.state_path) as archive:
                required = {"0.pkl", "0_image.pkl"}
                missing = required - set(archive.namelist())
                if missing:
                    raise ArtifactError(
                        f"state is missing archive members {sorted(missing)}"
                    )
            with np.load(artifact.graph_path) as graph:
                if not {"features", "masks"} <= set(graph.files):
                    raise ArtifactError("graph must contain features and masks")
                graph_rows = len(graph["features"])
                if graph_rows != len(graph["masks"]):
                    raise ArtifactError("graph features/masks have different lengths")
            predictions = read_jsonl(artifact.predictions_path)
            if graph_rows != len(predictions):
                raise ArtifactError(
                    f"graph has {graph_rows} rows but predictions have "
                    f"{len(predictions)}"
                )
            # This full-clip summary validates every row. Actual selection is
            # performed later over four-second direction-normalized windows.
            summarize_predictions(predictions)
            neutral_corrected_track_assignments(
                predictions,
                context=artifact.id,
            )
            prepared[artifact.id] = predictions
        except (ArtifactError, OSError, ValueError, zipfile.BadZipFile) as error:
            problems.append(f"{artifact.id}: {error}")
    if problems:
        raise ArtifactError(
            "Review pool preflight failed; no videos were rendered:\n- "
            + "\n- ".join(problems)
        )
    return prepared


def _match_key(item: dict[str, Any]) -> str:
    return str(item.get("match_id") or item.get("game") or item.get("match"))


def _source_nframes(
    artifact: ClipArtifacts,
    predictions: list[dict[str, Any]],
    fps: int,
) -> int:
    item = artifact.item
    candidates = [
        item.get("nframes"),
        (
            round(float(item["duration_seconds"]) * fps)
            if item.get("duration_seconds") is not None
            else None
        ),
        (
            round(
                (
                    float(item["end_seconds"])
                    - float(item["start_seconds"])
                )
                * fps
            )
            if item.get("end_seconds") is not None
            and item.get("start_seconds") is not None
            else None
        ),
        max(int(row["frame"]) for row in predictions) + 1,
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            nframes = int(value)
        except (TypeError, ValueError):
            continue
        if nframes > 0:
            return nframes
    raise ArtifactError(f"{artifact.id}: cannot determine source frame count")


@dataclass
class _FlowEdge:
    """One mutable residual edge used by the small review-set optimiser."""

    to: int
    reverse: int
    capacity: int
    cost: int


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    *,
    capacity: int,
    cost: int,
) -> _FlowEdge:
    forward = _FlowEdge(
        to=target,
        reverse=len(graph[target]),
        capacity=capacity,
        cost=cost,
    )
    backward = _FlowEdge(
        to=source,
        reverse=len(graph[source]),
        capacity=0,
        cost=-cost,
    )
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _minimum_cost_maximum_flow(
    graph: list[list[_FlowEdge]],
    source: int,
    sink: int,
) -> int:
    """Send maximum flow, choosing the minimum-cost assignment.

    The graph here is tiny (roughly 63 clips and three labels), so a
    Bellman-Ford shortest-path pass is simpler and safer than adding a solver
    dependency. Negative reverse edges let later augmentations repair an
    earlier choice when a shared clip is the only route to a scarce class.
    """

    flow = 0
    node_count = len(graph)
    while True:
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[node] + edge.cost
                    if (
                        distances[edge.to] is None
                        or candidate < distances[edge.to]
                    ):
                        distances[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break

        node = sink
        path_nodes = {sink}
        while node != source:
            predecessor = previous[node]
            if predecessor is None:
                raise ArtifactError("Internal assignment optimiser lost its path")
            previous_node, edge_index = predecessor
            if previous_node in path_nodes:
                raise ArtifactError(
                    "Internal assignment optimiser found a residual cycle"
                )
            path_nodes.add(previous_node)
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous_node
        flow += 1
    return flow


def _select_target_proposals(
    artifacts: list[ClipArtifacts],
    proposals: dict[str, dict[str, PredictionSummary]],
    *,
    per_target: int,
) -> list[tuple[ClipArtifacts, PredictionSummary]]:
    """Globally assign clips to priority classes under one-clip/one-label.

    This is a maximum-cardinality bipartite b-matching with deterministic,
    lexicographic preferences:

    1. fill as many priority slots as the proposal graph permits;
    2. spread unavoidable shortages across the three target classes;
    3. cover more matches per class, then more matches overall;
    4. maximise aggregate proposal quality.

    The decreasing class-slot rewards are what prevent an abundant label from
    starving another label. A one-clip tie between labels still falls through
    to proposal quality, so a strong trap is not consumed as a weak high press.
    """

    if per_target <= 0:
        return []
    ordered_artifacts = sorted(artifacts, key=lambda artifact: artifact.id)
    artifact_by_id = {artifact.id: artifact for artifact in ordered_artifacts}
    if len(artifact_by_id) != len(ordered_artifacts):
        raise ArtifactError("Cannot optimise review selection with duplicate clip IDs")
    eligible_ids = [
        artifact.id
        for artifact in ordered_artifacts
        if any(label in proposals.get(artifact.id, {}) for label in TARGET_LABELS)
    ]
    if not eligible_ids:
        return []

    match_for_clip = {
        clip_id: _match_key(artifact_by_id[clip_id].item)
        for clip_id in eligible_ids
    }
    clips_by_match: dict[str, list[str]] = {}
    for clip_id in eligible_ids:
        clips_by_match.setdefault(match_for_clip[clip_id], []).append(clip_id)
    label_match_pairs = sorted(
        {
            (label, match_for_clip[clip_id])
            for clip_id in eligible_ids
            for label in TARGET_LABELS
            if label in proposals[clip_id]
        },
        key=lambda pair: (TARGET_LABELS.index(pair[0]), pair[1]),
    )

    graph: list[list[_FlowEdge]] = []

    def node() -> int:
        graph.append([])
        return len(graph) - 1

    source = node()
    sink = node()
    match_nodes = {match: node() for match in sorted(clips_by_match)}
    clip_nodes = {clip_id: node() for clip_id in eligible_ids}
    label_match_nodes = {pair: node() for pair in label_match_pairs}
    label_nodes = {label: node() for label in TARGET_LABELS}

    flow_bound = min(len(eligible_ids), len(TARGET_LABELS) * per_target)
    quality_scale = 1_000_000
    # Each successive unit strictly dominates every possible total of the
    # lower-priority objective. Python integers keep this exact and cheap.
    maximum_quality_total = flow_bound * quality_scale
    overall_match_bonus = maximum_quality_total + 1
    maximum_overall_match_objective = (
        flow_bound * overall_match_bonus + maximum_quality_total
    )
    label_match_bonus = maximum_overall_match_objective + 1
    maximum_match_objective = (
        flow_bound * label_match_bonus + maximum_overall_match_objective
    )
    balanced_class_slot_bonus = maximum_match_objective + 1

    for match in sorted(clips_by_match):
        match_node = match_nodes[match]
        # The first selected clip from a match earns an overall-diversity
        # preference; later clips remain available without that reward.
        _add_flow_edge(
            graph,
            source,
            match_node,
            capacity=1,
            cost=-overall_match_bonus,
        )
        remainder = len(clips_by_match[match]) - 1
        if remainder:
            _add_flow_edge(
                graph,
                source,
                match_node,
                capacity=remainder,
                cost=0,
            )
        for clip_id in sorted(clips_by_match[match]):
            _add_flow_edge(
                graph,
                match_node,
                clip_nodes[clip_id],
                capacity=1,
                cost=0,
            )

    assignment_edges: dict[tuple[str, str], _FlowEdge] = {}
    for clip_id in eligible_ids:
        for label in TARGET_LABELS:
            summary = proposals[clip_id].get(label)
            if summary is None:
                continue
            quality = max(0, min(quality_scale, round(summary.quality * quality_scale)))
            assignment_edges[(clip_id, label)] = _add_flow_edge(
                graph,
                clip_nodes[clip_id],
                label_match_nodes[(label, match_for_clip[clip_id])],
                capacity=1,
                cost=-quality,
            )

    for label, match in label_match_pairs:
        pair_node = label_match_nodes[(label, match)]
        pair_capacity = min(
            per_target,
            sum(
                label in proposals[clip_id]
                for clip_id in clips_by_match[match]
            ),
        )
        _add_flow_edge(
            graph,
            pair_node,
            label_nodes[label],
            capacity=1,
            cost=-label_match_bonus,
        )
        if pair_capacity > 1:
            _add_flow_edge(
                graph,
                pair_node,
                label_nodes[label],
                capacity=pair_capacity - 1,
                cost=0,
            )

    for label in TARGET_LABELS:
        # Concave slot rewards make one selection in each class preferable to
        # two in one class, two each preferable to a 3/1 split, and so on.
        for slot in range(per_target):
            _add_flow_edge(
                graph,
                label_nodes[label],
                sink,
                capacity=1,
                cost=-(per_target - slot) * balanced_class_slot_bonus,
            )

    _minimum_cost_maximum_flow(graph, source, sink)
    selected = [
        (artifact_by_id[clip_id], proposals[clip_id][label])
        for clip_id in eligible_ids
        for label in TARGET_LABELS
        if (
            (edge := assignment_edges.get((clip_id, label))) is not None
            and edge.capacity == 0
        )
    ]
    selected.sort(
        key=lambda pair: (
            TARGET_LABELS.index(pair[1].model_label),
            _match_key(pair[0].item),
            pair[0].id,
        )
    )
    return selected


def select_balanced_windows(
    artifacts: list[ClipArtifacts],
    prepared: dict[str, list[dict[str, Any]]],
    *,
    fps: int,
    duration: float,
    stride_seconds: float,
    min_graph_frames: int,
    per_target: int,
) -> tuple[
    list[tuple[ClipArtifacts, PredictionSummary]],
    dict[str, int],
    dict[str, dict[str, int]],
]:
    """Post-balance candidates using actual graph classes, not X-CLIP tags."""

    span_frames = max(1, int(round(duration * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    proposals: dict[str, dict[str, PredictionSummary]] = {}
    for artifact in artifacts:
        predictions = prepared[artifact.id]
        proposals[artifact.id] = temporal_window_proposals(
            predictions,
            nframes=_source_nframes(artifact, predictions, fps),
            span_frames=span_frames,
            stride_frames=stride_frames,
            min_graph_frames=min_graph_frames,
        )

    selected = _select_target_proposals(
        artifacts,
        proposals,
        per_target=per_target,
    )
    used = {artifact.id for artifact, _ in selected}
    counts = Counter()
    match_counts = Counter()
    per_label_match_counts: dict[str, Counter[str]] = {
        label: Counter() for label in TACTICAL_LABELS
    }
    for artifact, summary in selected:
        label = summary.model_label
        match = _match_key(artifact.item)
        counts[label] += 1
        match_counts[match] += 1
        per_label_match_counts[label][match] += 1

    # Lower-priority classes only fill places left empty by unavailable target
    # windows. They never displace a high press or either touchline trap.
    target_capacity = len(TARGET_LABELS) * per_target
    while len(selected) < target_capacity:
        choices = [
            (artifact, proposals[artifact.id][label])
            for artifact in sorted(artifacts, key=lambda value: value.id)
            if artifact.id not in used
            for label in FALLBACK_LABELS
            if label in proposals[artifact.id]
        ]
        if not choices:
            break
        artifact, summary = min(
            choices,
            key=lambda pair: (
                per_label_match_counts[pair[1].model_label][
                    _match_key(pair[0].item)
                ],
                match_counts[_match_key(pair[0].item)],
                -pair[1].quality,
                -pair[1].valid_graph_frames,
                -pair[1].classification_confidence,
                FALLBACK_LABELS.index(pair[1].model_label),
                pair[0].id,
            ),
        )
        selected.append((artifact, summary))
        used.add(artifact.id)
        label = summary.model_label
        counts[label] += 1
        match = _match_key(artifact.item)
        match_counts[match] += 1
        per_label_match_counts[label][match] += 1

    actual_counts = {label: int(counts[label]) for label in TACTICAL_LABELS}
    actual_by_match = {
        match: {
            label: sum(
                1
                for artifact, summary in selected
                if _match_key(artifact.item) == match
                and summary.model_label == label
            )
            for label in TACTICAL_LABELS
        }
        for match in sorted({_match_key(artifact.item) for artifact, _ in selected})
    }
    return selected, actual_counts, actual_by_match


def neutral_corrected_track_assignments(
    predictions: list[dict[str, Any]],
    *,
    context: str = "predictions",
) -> dict[int, str]:
    """Read the converter's anonymous, colour-corrected per-track teams.

    These are the exact assignments used to build graph possession/pressing
    features.  Refusing raw ``team_cluster`` fallback is intentional: that
    cluster precedes the converter's decisive shirt-colour outlier correction
    and can therefore disagree with the tactical graph shown beside it.
    """

    expected_names = {"left": "Team A", "right": "Team B"}
    normalized: dict[int, str] | None = None
    for index, row in enumerate(predictions):
        if row.get("team_identity_status") != "unreviewed":
            raise ArtifactError(
                f"{context}: prediction {index} is not an unreviewed neutral "
                "team assignment"
            )
        if row.get("team_identity_map") != expected_names:
            raise ArtifactError(
                f"{context}: prediction {index} does not use anonymous "
                "Team A/Team B names"
            )
        evidence = row.get("team_cluster_evidence")
        if not isinstance(evidence, dict):
            raise ArtifactError(
                f"{context}: prediction {index} lacks team-cluster evidence"
            )
        identity_inputs = evidence.get("neutral_identity_inputs")
        if identity_inputs != {
            "manual_labels_enabled": False,
            "identity_model_enabled": False,
        }:
            raise ArtifactError(
                f"{context}: prediction {index} was not produced with manual "
                "team labels and the identity model disabled"
            )
        raw = evidence.get("neutral_corrected_track_assignments")
        if not isinstance(raw, dict) or not raw:
            raise ArtifactError(
                f"{context}: prediction {index} lacks the converter's neutral "
                "corrected track assignments"
            )
        try:
            current = {int(track_id): str(team) for track_id, team in raw.items()}
        except (TypeError, ValueError) as error:
            raise ArtifactError(
                f"{context}: prediction {index} has invalid track IDs"
            ) from error
        if any(track_id < 0 for track_id in current) or any(
            team not in {"left", "right"} for team in current.values()
        ):
            raise ArtifactError(
                f"{context}: prediction {index} has invalid neutral track "
                "assignments"
            )
        if normalized is None:
            normalized = current
        elif current != normalized:
            raise ArtifactError(
                f"{context}: corrected track assignments change between "
                "prediction rows"
            )
    if normalized is None:
        raise ArtifactError(f"{context}: no corrected track assignments")
    return normalized


def _neutral_team_slot(
    row: Any,
    track_assignments: dict[int, str],
) -> int | None:
    value = getattr(row, "track_id", None)
    try:
        track_id = int(value)
    except (TypeError, ValueError):
        return None
    return {"left": 0, "right": 1}.get(track_assignments.get(track_id))


def _pitch_point(row: Any) -> Any | None:
    import numpy as np

    pitch = getattr(row, "bbox_pitch", None)
    if not isinstance(pitch, dict):
        return None
    try:
        point = np.asarray(
            [pitch["x_bottom_middle"], pitch["y_bottom_middle"]],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(point).all() or abs(point[0]) > 54.5 or abs(point[1]) > 36:
        return None
    return point


def render_neutral_overlay(
    artifact: ClipArtifacts,
    output_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    label: str,
    fps: int,
    track_assignments: dict[int, str],
    nonparticipant_model: Path | None,
    nonparticipant_threshold: float | None,
) -> tuple[int, int, dict[str, Any]]:
    """Render one short overlay using the graph converter's neutral teams.

    No club-specific labels, manual identity annotations, or identity
    classifier are read here. The two colours distinguish anonymous Team A
    from Team B and exactly follow the per-track assignments used to construct
    this clip's graph, including decisive shirt-colour outlier corrections.
    """

    import cv2
    import numpy as np

    with zipfile.ZipFile(artifact.state_path) as archive:
        detections = pickle.loads(archive.read("0.pkl"))
        images = pickle.loads(archive.read("0_image.pkl"))
    excluded_track_ids, track_filter_evidence = infer_nonparticipant_track_ids(
        detections,
        video_path=artifact.clip_path,
        model_path=nonparticipant_model,
        threshold=nonparticipant_threshold,
    )
    by_frame = {
        int(frame): rows for frame, rows in detections.groupby("image_id")
    }
    available_frames = int(len(images))
    start = max(0, min(int(start_frame), available_frames - 1))
    end = min(available_frames, max(start + 1, int(end_frame)))

    capture = cv2.VideoCapture(str(artifact.clip_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ArtifactError(f"{artifact.id}: cannot open source video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        stdin=subprocess.PIPE,
    )
    frames_written = 0
    try:
        for frame in range(start, end):
            ok, image = capture.read()
            if not ok:
                break
            nodes = []
            rows = by_frame.get(frame)
            if rows is not None:
                for row in rows.itertuples(index=False):
                    if str(getattr(row, "role", "")) not in {
                        "player",
                        "goalkeeper",
                    }:
                        continue
                    try:
                        track_id = int(getattr(row, "track_id"))
                    except (TypeError, ValueError):
                        continue
                    if track_id in excluded_track_ids:
                        continue
                    cluster = _neutral_team_slot(row, track_assignments)
                    point = _pitch_point(row)
                    try:
                        left, top, box_width, box_height = map(
                            float, getattr(row, "bbox_ltwh")
                        )
                    except (TypeError, ValueError):
                        continue
                    if cluster is None or point is None:
                        continue
                    x1 = int(np.clip(left, 0, width - 1))
                    y1 = int(np.clip(top, 0, height - 1))
                    x2 = int(np.clip(left + box_width, x1 + 1, width))
                    y2 = int(np.clip(top + box_height, y1 + 1, height))
                    centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    nodes.append((cluster, point, centre, (x1, y1, x2, y2)))

            edges: set[tuple[int, int]] = set()
            for node_index, (cluster, point, _, _) in enumerate(nodes):
                neighbours = sorted(
                    (
                        (float(np.linalg.norm(point - other_point)), other_index)
                        for other_index, (
                            other_cluster,
                            other_point,
                            _,
                            _,
                        ) in enumerate(nodes)
                        if other_index != node_index and other_cluster == cluster
                    )
                )
                for distance_metres, other_index in neighbours[:2]:
                    if distance_metres <= 22:
                        edges.add(tuple(sorted((node_index, other_index))))
            for left_index, right_index in edges:
                cluster = nodes[left_index][0]
                cv2.line(
                    image,
                    nodes[left_index][2],
                    nodes[right_index][2],
                    TEAM_COLOURS[cluster],
                    1,
                    cv2.LINE_AA,
                )
            for left_index, (cluster, point, centre, _) in enumerate(nodes):
                for right_index in range(left_index + 1, len(nodes)):
                    other_cluster, other_point, other_centre, _ = nodes[right_index]
                    if (
                        cluster != other_cluster
                        and float(np.linalg.norm(point - other_point)) <= 12
                    ):
                        cv2.line(
                            image,
                            centre,
                            other_centre,
                            (45, 215, 245),
                            1,
                            cv2.LINE_AA,
                        )
            for cluster, _, _, (x1, y1, x2, y2) in nodes:
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    TEAM_COLOURS[cluster],
                    1,
                    cv2.LINE_AA,
                )
            caption = LABEL_TITLES[label].upper()
            text_size, baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                image,
                (14, 14),
                (30 + text_size[0], 25 + text_size[1] + baseline),
                (11, 23, 17),
                -1,
            )
            cv2.putText(
                image,
                caption,
                (22, 22 + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (242, 246, 243),
                1,
                cv2.LINE_AA,
            )
            assert encoder.stdin is not None
            encoder.stdin.write(image.tobytes())
            frames_written += 1
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    return_code = encoder.wait()
    if return_code or frames_written != end - start:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(
            f"{artifact.id}: overlay encoder wrote {frames_written}/{end-start} "
            f"frames (ffmpeg exit {return_code})"
        )
    temporary.replace(output_path)
    return start, end, track_filter_evidence


def _retrieval_score(item: dict[str, Any]) -> float | None:
    value = item.get("selected_for_query_score", item.get("query_score"))
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{item['id']}: retrieval score is not numeric") from error
    if not math.isfinite(result):
        raise ArtifactError(f"{item['id']}: retrieval score is not finite")
    return result


def _clock(half: int, half_seconds: float) -> str:
    absolute = max(0.0, half_seconds) + (45 * 60 if half == 2 else 0)
    minute = int(absolute // 60)
    seconds = absolute % 60
    return f"{minute:02d}:{seconds:04.1f}"


def review_item(
    artifact: ClipArtifacts,
    summary: PredictionSummary,
    video_path: Path,
    *,
    fps: int,
    overlay_track_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = artifact.item
    half = int(item.get("half", 1))
    source_start = float(
        item.get("source_start_seconds", item.get("start_seconds", 0))
    )
    clip_time = summary.representative_frame / fps
    excerpt_start_time = summary.window_start_frame / fps
    excerpt_end_time = summary.window_end_frame / fps
    retrieval_score = _retrieval_score(item)
    return {
        "id": artifact.id,
        "video": video_path.as_posix(),
        "match": str(item.get("game", item.get("match", "Unknown match"))),
        "match_id": item.get("match_id"),
        "half": half,
        "match_clock": _clock(half, source_start + clip_time),
        "excerpt_match_clock_start": _clock(
            half, source_start + excerpt_start_time
        ),
        "excerpt_match_clock_end": _clock(
            half, source_start + excerpt_end_time
        ),
        "source_start_seconds": source_start,
        "representative_frame": summary.representative_frame,
        "clip_time_seconds": round(clip_time, 3),
        "excerpt_start_frame": summary.window_start_frame,
        "excerpt_end_frame": summary.window_end_frame,
        "excerpt_duration_seconds": round(
            (summary.window_end_frame - summary.window_start_frame) / fps,
            3,
        ),
        "model_label": summary.model_label,
        "model_label_title": LABEL_TITLES[summary.model_label],
        "classification_confidence": round(
            summary.classification_confidence, 6
        ),
        "temporal_agreement": round(summary.temporal_agreement, 6),
        "majority_frames": summary.majority_frames,
        "valid_graph_frames": summary.valid_graph_frames,
        "retrieval_score": (
            round(retrieval_score, 6) if retrieval_score is not None else None
        ),
        "retrieval_query_id": item.get(
            "selected_for_query_id", item.get("matched_query_id")
        ),
        "retrieval_query": item.get(
            "selected_for_query", item.get("matched_query")
        ),
        "direction_usable": summary.direction_usable,
        "direction_confidence": (
            round(summary.direction_confidence, 6)
            if summary.direction_confidence is not None
            else None
        ),
        "direction_required": (
            summary.model_label in DIRECTION_DEPENDENT_LABELS
        ),
        "direction_sources": list(summary.direction_sources),
        "direction_statuses": list(summary.direction_statuses),
        "direction_evidence": summary.direction_evidence,
        "team_identity_mode": "neutral_corrected_track_teams",
        "team_assignment_source": (
            "graph_converter_clip_local_side_alignment_and_colour_correction"
        ),
        "overlay_track_filter": overlay_track_filter,
        "provenance": {
            "clip": str(artifact.clip_path),
            "graph": str(artifact.graph_path),
            "predictions": str(artifact.predictions_path),
            "tracklab_state": str(artifact.state_path),
        },
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    downstream_rows, downstream_report = completed_downstream_rows(
        args.manifest.resolve(),
        args.downstream_status.resolve(),
    )
    artifacts = resolve_artifacts(
        args.manifest.resolve(),
        args.graph_dir.resolve(),
        args.state_root.resolve(),
        workspace,
        downstream_rows,
    )
    prepared = preflight_artifacts(artifacts)
    selected, actual_counts, actual_by_match = select_balanced_windows(
        artifacts,
        prepared,
        fps=args.fps,
        duration=args.duration,
        stride_seconds=args.window_stride,
        min_graph_frames=args.min_window_graphs,
        per_target=args.per_target,
    )
    if not selected:
        raise ArtifactError(
            "No temporal window met the graph-support requirement; no videos "
            "were rendered"
        )
    output = args.output.resolve()
    video_dir = output / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, (artifact, summary) in enumerate(selected, 1):
        relative_video = Path("videos") / f"{artifact.id}.mp4"
        _, _, overlay_track_filter = render_neutral_overlay(
            artifact,
            output / relative_video,
            start_frame=summary.window_start_frame,
            end_frame=summary.window_end_frame,
            label=summary.model_label,
            fps=args.fps,
            track_assignments=neutral_corrected_track_assignments(
                prepared[artifact.id],
                context=artifact.id,
            ),
            nonparticipant_model=args.nonparticipant_model.resolve()
            if args.nonparticipant_model is not None
            else None,
            nonparticipant_threshold=args.nonparticipant_threshold,
        )
        items.append(
            review_item(
                artifact,
                summary,
                relative_video,
                fps=args.fps,
                overlay_track_filter=overlay_track_filter,
            )
        )
        print(
            f"[{index:03d}/{len(selected):03d}] {artifact.id}: "
            f"{summary.model_label} "
            f"({summary.classification_confidence:.1%})"
        )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "downstream": downstream_report,
        "team_identity_mode": "neutral_corrected_track_teams",
        "team_assignment_source": (
            "graph_converter_clip_local_side_alignment_and_colour_correction"
        ),
        "labels": list(TACTICAL_LABELS),
        "target_labels": list(TARGET_LABELS),
        "label_titles": LABEL_TITLES,
        "selection": {
            "method": "actual_trusted_direction_normalized_graph_class",
            "direction_gate": (
                "high_press, trap_left, trap_right, and central_screen require "
                "trusted attacker-relative direction; unstructured is "
                "direction-invariant"
            ),
            "per_target_cap": args.per_target,
            "excerpt_duration_seconds": args.duration,
            "window_stride_seconds": args.window_stride,
            "minimum_graph_frames": args.min_window_graphs,
            "xclip_role": "candidate_metadata_only",
            "actual_counts": actual_counts,
            "actual_counts_by_match": actual_by_match,
        },
        "confidence_definition": (
            "Mean graph-classifier probability for the majority tactical class "
            "across possession-confident frames in the selected excerpt."
        ),
        "retrieval_definition": (
            "Video-text cosine used to propose this clip; it is not a "
            "classification confidence."
        ),
        "items": items,
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(f"Wrote {len(items)} review videos to {output}")
    print(
        "Downstream terminal outcomes: "
        f"completed={downstream_report['completed_count']}, "
        f"failed={downstream_report['failed_count']}, "
        f"excluded={downstream_report['excluded_count']}"
    )
    print(
        "Actual graph-class counts: "
        + ", ".join(f"{label}={actual_counts[label]}" for label in TACTICAL_LABELS)
    )
    for match, counts_by_label in actual_by_match.items():
        compact = ", ".join(
            f"{label}={count}"
            for label, count in counts_by_label.items()
            if count
        )
        print(f"  {match}: {compact or 'no selected windows'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/arsenal_expansion/gsr-clips.json"),
        help="GSR clip manifest with one clip_path per item",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/graphs/arsenal_expansion"),
        help="<clip-id>.npz and <clip-id>-predictions.jsonl directory",
    )
    parser.add_argument(
        "--downstream-status",
        type=Path,
        default=Path(
            "data/manifests/arsenal_expansion/downstream-status.json"
        ),
        help=(
            "Terminal process_gsr_outputs status; only completed clips are "
            "preflighted, while failures are recorded as exclusions"
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("third_party/sn-gamestate/outputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/review/expanded_tactical"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths stored in the GSR manifest",
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--window-stride",
        type=float,
        default=0.5,
        help="Stride when searching four-second excerpts inside each source",
    )
    parser.add_argument(
        "--min-window-graphs",
        type=int,
        default=2,
        help="Minimum possession-reliable graph frames in a selected excerpt",
    )
    parser.add_argument(
        "--per-target",
        type=int,
        default=8,
        help="Maximum actual-graph videos for each priority class",
    )
    parser.add_argument(
        "--nonparticipant-model",
        type=Path,
        default=None,
        help=(
            "Optional expert-trained track model used only for its ignore "
            "class. Disabled by default because club-specific appearance "
            "models are not safe across unseen fixtures"
        ),
    )
    parser.add_argument(
        "--nonparticipant-threshold",
        type=float,
        default=None,
        help="Override the ignore-model acceptance threshold",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.window_stride <= 0:
        parser.error("--window-stride must be positive")
    if args.min_window_graphs <= 0:
        parser.error("--min-window-graphs must be positive")
    if args.per_target <= 0:
        parser.error("--per-target must be positive")
    if (
        args.nonparticipant_threshold is not None
        and not 0 <= args.nonparticipant_threshold <= 1
    ):
        parser.error("--nonparticipant-threshold must be between 0 and 1")
    if (
        args.nonparticipant_model is not None
        and not args.nonparticipant_model.is_file()
    ):
        parser.error(
            f"--nonparticipant-model does not exist: {args.nonparticipant_model}"
        )
    return args


def main() -> None:
    try:
        build(parse_args())
    except ArtifactError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
