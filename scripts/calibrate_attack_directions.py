#!/usr/bin/env python3
"""Aggregate conservative attack-direction evidence from completed GSR clips.

Run with the sn-gamestate environment because TrackLab state archives contain
pandas objects. This command only reads completed state files; it does not
launch a model or use a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from attack_direction import (
        aggregate_clip_calibrations,
        calibrate_detections,
    )
    from run_gsr_batch import (
        find_completed_state,
        load_jobs,
        manifest_records,
    )
    from team_identity import (
        infer_cluster_to_internal_from_tracklab_team,
        load_match_team_config,
    )
except ModuleNotFoundError:
    from scripts.attack_direction import (
        aggregate_clip_calibrations,
        calibrate_detections,
    )
    from scripts.run_gsr_batch import (
        find_completed_state,
        load_jobs,
        manifest_records,
    )
    from scripts.team_identity import (
        infer_cluster_to_internal_from_tracklab_team,
        load_match_team_config,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path("data/manifests/arsenal_expansion/gsr-clips.json")
DEFAULT_STATE_ROOT = Path("third_party/sn-gamestate/outputs")
DEFAULT_TEAM_REGISTRY = Path(
    "data/annotations/team_identity_registry.example.json"
)
DEFAULT_OUTPUT = Path(
    "data/annotations/attack_direction_registry.generated.json"
)


def absolute(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_detections(state: Path):
    with zipfile.ZipFile(state) as archive:
        return pickle.loads(archive.read("0.pkl"))


@dataclass(frozen=True)
class CalibrationJobContext:
    """Validated manifest/config context for one GSR clip."""

    job: Any
    match_id: str
    half: int
    team_config: dict[str, Any]


@dataclass(frozen=True)
class CollectionStats:
    """Non-overlapping clip counts collected while reading GSR states."""

    clips_waiting_for_state: int
    clips_with_completed_state: int


def align_internal_teams(
    detections,
    *,
    team_registry: Path,
    match_id: str,
    sequence_id: str,
    team_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Map clip-local KMeans clusters to the configured internal teams."""

    config = team_config
    if config is None:
        config = load_match_team_config(
            team_registry,
            match_id,
            sequence_id=sequence_id,
        )
    raw_cluster_to_side, side_evidence = (
        infer_cluster_to_internal_from_tracklab_team(detections)
    )
    cluster_to_internal = config["cluster_to_internal"]
    if cluster_to_internal is None:
        cluster_to_internal = raw_cluster_to_side
    side_to_internal = {
        raw_cluster_to_side[cluster]: internal
        for cluster, internal in cluster_to_internal.items()
    }

    aligned = detections.copy()
    athletes = aligned.role.isin(["player", "goalkeeper"])
    original_side = aligned.loc[athletes, "team"].copy()

    def mapped_cluster(value: Any) -> str | None:
        try:
            cluster = int(value)
        except (TypeError, ValueError):
            return None
        return cluster_to_internal.get(cluster)

    mapped = aligned.loc[athletes, "team_cluster"].map(
        mapped_cluster
    )
    goalkeeper_or_unclustered = mapped.isna()
    mapped.loc[goalkeeper_or_unclustered] = original_side.loc[
        goalkeeper_or_unclustered
    ].map(side_to_internal)
    aligned.loc[athletes, "team"] = mapped
    return aligned, {
        "mapping_status": config["mapping_status"],
        "cluster_mapping_scope": config["cluster_mapping_scope"],
        "raw_cluster_to_tracklab_side": {
            str(cluster): side
            for cluster, side in raw_cluster_to_side.items()
        },
        "cluster_to_internal": {
            str(cluster): internal
            for cluster, internal in cluster_to_internal.items()
        },
        "tracklab_side_to_internal": side_to_internal,
        "tracklab_side_evidence": side_evidence,
    }


def preflight_job_contexts(
    jobs: list[Any],
    rows: dict[str, dict[str, Any]],
    *,
    team_registry: Path,
) -> list[CalibrationJobContext]:
    """Validate all manifest metadata and registry entries before state reads.

    Configuration failures are deliberately fatal. Preflighting every selected
    sequence keeps a malformed manifest or reviewed team mapping from being
    mislabeled as a recoverable clip-data exclusion halfway through a run.
    """

    contexts: list[CalibrationJobContext] = []
    for job in jobs:
        row = rows[job.id]
        match_id = row.get("match_id")
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError(f"Clip {job.id!r} has no non-empty match_id")
        match_id = match_id.strip()
        raw_half = row.get("half")
        if isinstance(raw_half, bool) or not (
            isinstance(raw_half, int)
            or (
                isinstance(raw_half, str)
                and raw_half.strip() in {"1", "2"}
            )
        ):
            raise ValueError(f"Clip {job.id!r} has invalid half {raw_half!r}")
        half = int(raw_half)
        if half not in (1, 2):
            raise ValueError(f"Clip {job.id!r} has invalid half {half}")
        team_config = load_match_team_config(
            team_registry,
            match_id,
            sequence_id=job.id,
        )
        contexts.append(
            CalibrationJobContext(
                job=job,
                match_id=match_id,
                half=half,
                team_config=team_config,
            )
        )
    return contexts


def error_type(error: Exception) -> str:
    """Return the stable, fully qualified class name for an audited failure."""

    cls = type(error)
    return f"{cls.__module__}.{cls.__qualname__}"


def excluded_clip_evidence(
    *,
    sequence_id: str,
    stage: str,
    error: Exception,
    error_path: Path,
    state_path: Path | None,
) -> dict[str, Any]:
    """Represent one recoverable clip failure as unusable direction evidence."""

    return {
        "status": f"excluded_{stage}_error",
        "confident": False,
        "confidence": 0.0,
        "directions": None,
        "sequence_id": sequence_id,
        "state_path": str(state_path) if state_path is not None else None,
        "error": {
            "stage": stage,
            "type": error_type(error),
            "message": str(error),
            "path": str(error_path),
        },
    }


def process_completed_state(
    context: CalibrationJobContext,
    state: Path,
    *,
    team_registry: Path,
) -> dict[str, Any]:
    """Calibrate one state, converting clip-data failures into exclusions."""

    try:
        detections = load_detections(state)
    except Exception as error:
        return excluded_clip_evidence(
            sequence_id=context.job.id,
            stage="state_load",
            error=error,
            error_path=state,
            state_path=state,
        )
    try:
        aligned, alignment = align_internal_teams(
            detections,
            team_registry=team_registry,
            match_id=context.match_id,
            sequence_id=context.job.id,
            team_config=context.team_config,
        )
    except Exception as error:
        return excluded_clip_evidence(
            sequence_id=context.job.id,
            stage="team_alignment",
            error=error,
            error_path=state,
            state_path=state,
        )
    try:
        evidence = calibrate_detections(aligned)
    except Exception as error:
        return excluded_clip_evidence(
            sequence_id=context.job.id,
            stage="direction_calibration",
            error=error,
            error_path=state,
            state_path=state,
        )
    evidence.update(
        sequence_id=context.job.id,
        state_path=str(state),
        alignment=alignment,
    )
    return evidence


def collect_clip_calibrations(
    contexts: list[CalibrationJobContext],
    *,
    state_root: Path,
    team_registry: Path,
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    CollectionStats,
]:
    """Collect every selected clip without letting one bad state abort a run."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    waiting = 0
    completed = 0
    for index, context in enumerate(contexts, start=1):
        job = context.job
        try:
            state = find_completed_state(state_root, job)
        except Exception as error:
            evidence = excluded_clip_evidence(
                sequence_id=job.id,
                stage="state_discovery",
                error=error,
                error_path=state_root / job.experiment_name,
                state_path=None,
            )
            grouped[(context.match_id, context.half)].append(evidence)
            print(
                f"[{index}/{len(contexts)}] {job.id}: "
                f"{evidence['status']}"
            )
            continue
        if state is None:
            waiting += 1
            print(f"[{index}/{len(contexts)}] {job.id}: waiting_for_state")
            continue
        completed += 1
        evidence = process_completed_state(
            context,
            state,
            team_registry=team_registry,
        )
        grouped[(context.match_id, context.half)].append(evidence)
        print(
            f"[{index}/{len(contexts)}] {job.id}: {evidence['status']} "
            f"confidence={evidence['confidence']:.3f}"
        )
    return grouped, CollectionStats(
        clips_waiting_for_state=waiting,
        clips_with_completed_state=completed,
    )


def direction_object(direction: int, confidence: float) -> dict[str, Any]:
    return {
        "raw": int(direction),
        "label": "left_to_right" if direction == 1 else "right_to_left",
        "confidence": round(float(confidence), 6),
    }


def registry_payload(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    expected: set[tuple[str, int]],
    *,
    minimum_clips: int,
    minimum_vote_confidence: float,
) -> dict[str, Any]:
    matches: dict[str, Any] = {}
    for match_id, half in sorted(expected):
        clip_evidence = grouped.get((match_id, half), [])
        aggregate = aggregate_clip_calibrations(
            clip_evidence,
            minimum_clips=minimum_clips,
            minimum_vote_confidence=minimum_vote_confidence,
        )
        directions = aggregate.get("directions")
        attacking_direction = (
            {
                team: direction_object(
                    int(directions[team]),
                    float(aggregate["confidence"]),
                )
                for team in ("left", "right")
            }
            if directions is not None
            else None
        )
        match = matches.setdefault(match_id, {"halves": {}})
        match["halves"][str(half)] = {
            "status": aggregate["status"],
            "direction_confident": bool(aggregate["confident"]),
            "confidence": float(aggregate["confidence"]),
            "attacking_direction": attacking_direction,
            "evidence": {
                **aggregate,
                "clips": clip_evidence,
            },
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "coordinate_system": (
            "TrackLab raw pitch x; +1 attacks increasing x, -1 decreasing x"
        ),
        "method": (
            "track-level median pitch positions with goalkeeper-weighted "
            "clip votes; no screen-side, retrieval, kit, or club assumption"
        ),
        "matches": matches,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def summary_payload(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    clips_selected: int,
    collection_stats: CollectionStats,
    match_halves: int,
    match_halves_confident: int,
) -> dict[str, Any]:
    """Build auditable, mutually exclusive clip outcome counts."""

    clips = [
        clip
        for key in sorted(grouped)
        for clip in grouped[key]
    ]
    calibrated = sum(clip.get("status") == "calibrated" for clip in clips)
    abstained = sum(
        str(clip.get("status", "")).startswith("abstained_")
        for clip in clips
    )
    excluded = sum(
        str(clip.get("status", "")).startswith("excluded_")
        for clip in clips
    )
    other = len(clips) - calibrated - abstained - excluded
    exclusions_by_stage: dict[str, int] = defaultdict(int)
    for clip in clips:
        error = clip.get("error")
        if str(clip.get("status", "")).startswith("excluded_") and isinstance(
            error, dict
        ):
            exclusions_by_stage[str(error.get("stage", "unknown"))] += 1
    accounted_for = (
        len(clips) + collection_stats.clips_waiting_for_state
    )
    if accounted_for != clips_selected:
        raise RuntimeError(
            "Internal clip accounting error: "
            f"selected={clips_selected}, recorded={len(clips)}, "
            "waiting="
            f"{collection_stats.clips_waiting_for_state}"
        )
    return {
        "clips_selected": clips_selected,
        "clips_recorded": len(clips),
        "clips_with_completed_state": (
            collection_stats.clips_with_completed_state
        ),
        "clips_calibrated": calibrated,
        "clips_abstained": abstained,
        "clips_excluded": excluded,
        "clips_other_status": other,
        "clips_calibrated_or_abstained": calibrated + abstained,
        "clips_waiting_for_state": (
            collection_stats.clips_waiting_for_state
        ),
        "clips_accounted_for": accounted_for,
        "exclusions_by_stage": {
            stage: exclusions_by_stage[stage]
            for stage in sorted(exclusions_by_stage)
        },
        "match_halves": match_halves,
        "match_halves_confident": match_halves_confident,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--team-registry", type=Path, default=DEFAULT_TEAM_REGISTRY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-clips", type=int, default=2)
    parser.add_argument(
        "--minimum-vote-confidence", type=float, default=0.75
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help="Inspect only this exact clip ID; repeat as needed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed registry without writing --output",
    )
    args = parser.parse_args()
    if args.minimum_clips <= 0:
        parser.error("--minimum-clips must be positive")
    if not 0.5 < args.minimum_vote_confidence <= 1:
        parser.error("--minimum-vote-confidence must be in (0.5, 1]")
    return args


def main() -> int:
    args = parse_args()
    manifest = absolute(args.manifest)
    path_root = absolute(args.path_root)
    state_root = absolute(args.state_root)
    team_registry = absolute(args.team_registry)
    output = absolute(args.output)

    payload = json.loads(manifest.read_text())
    rows = {
        str(row["id"]): row for row in manifest_records(payload)
    }
    jobs = load_jobs(
        manifest,
        path_root,
        probe_frames=False,
    )
    if args.clip_id:
        requested = set(args.clip_id)
        unknown = requested - set(rows)
        if unknown:
            raise ValueError(f"Unknown clip IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if job.id in requested]
    if not jobs:
        raise ValueError("No clips selected")

    contexts = preflight_job_contexts(
        jobs,
        rows,
        team_registry=team_registry,
    )
    expected = {
        (context.match_id, context.half) for context in contexts
    }
    grouped, collection_stats = collect_clip_calibrations(
        contexts,
        state_root=state_root,
        team_registry=team_registry,
    )

    result = registry_payload(
        grouped,
        expected,
        minimum_clips=args.minimum_clips,
        minimum_vote_confidence=args.minimum_vote_confidence,
    )
    match_halves_confident = sum(
        bool(half["direction_confident"])
        for match in result["matches"].values()
        for half in match["halves"].values()
    )
    result["summary"] = summary_payload(
        grouped,
        clips_selected=len(jobs),
        collection_stats=collection_stats,
        match_halves=len(expected),
        match_halves_confident=match_halves_confident,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        atomic_json(output, result)
        print(f"Wrote attack-direction registry to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
