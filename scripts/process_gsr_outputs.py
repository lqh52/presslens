#!/usr/bin/env python3
"""Convert completed Arsenal-expansion TrackLab states into tactical graphs.

This is the resumable downstream half of ``run_gsr_batch.py``.  For every clip
in the extraction manifest it:

1. selects the newest *valid* completed TrackLab state;
2. converts the state to canonical graph features with neutral Team A/Team B
   identities from the unreviewed match registry;
3. derives auditable weak labels; and
4. applies the weakly supervised tactical graph classifier.

The original Burnley-Arsenal labels and identity model are explicitly disabled.
Missing states are recorded as waiting rather than mistaken for conversion
failures.  Existing artifacts are skipped only when they validate and are at
least as new as all inputs to their stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from run_gsr_batch import (
        GSRJob,
        find_completed_state,
        load_jobs,
        manifest_records,
        safe_identifier,
    )
    from team_identity import load_match_team_config
except ModuleNotFoundError:
    from scripts.run_gsr_batch import (
        GSRJob,
        find_completed_state,
        load_jobs,
        manifest_records,
        safe_identifier,
    )
    from scripts.team_identity import load_match_team_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path("data/manifests/arsenal_expansion/gsr-clips.json")
DEFAULT_STATE_ROOT = Path("third_party/sn-gamestate/outputs")
DEFAULT_OUTPUT_DIR = Path("data/graphs/arsenal_expansion")
DEFAULT_STATUS = Path(
    "data/manifests/arsenal_expansion/downstream-status.json"
)
DEFAULT_LOG_DIR = Path("data/logs/arsenal-expansion-downstream")
DEFAULT_TEAM_REGISTRY = Path(
    "data/annotations/team_identity_registry.example.json"
)
DEFAULT_DIRECTION_REGISTRY = Path(
    "data/annotations/attack_direction_registry.generated.json"
)
DEFAULT_YOLO = Path(
    "third_party/sn-gamestate/pretrained_models/yolo/yolo11m.pt"
)
DEFAULT_GRAPH_MODEL = Path("models/tactical_graph_weak.pt")
DEFAULT_TRACKLAB_PYTHON = Path("third_party/sn-gamestate/.venv/bin/python")
DEFAULT_ML_PYTHON = Path(".venv/bin/python")


class StageFailure(RuntimeError):
    """A subprocess stage failed; its full output is available in a log."""

    def __init__(self, stage: str, returncode: int, log_path: Path):
        super().__init__(
            f"{stage} exited with code {returncode}; see {log_path}"
        )
        self.stage = stage
        self.returncode = returncode
        self.log_path = log_path


@dataclass(frozen=True)
class DirectionSelection:
    """One optional direction override plus its auditable provenance."""

    directions: tuple[int, int] | None
    source: str
    status: str | None = None
    confident: bool | None = None
    confidence: float | None = None
    evidence: dict[str, Any] | None = None

    def converter_provenance(self) -> dict[str, Any] | None:
        """Return metadata for an explicit converter override, if present."""

        if self.directions is None:
            return None
        return {
            "source": self.source,
            "status": self.status or "calibrated",
            "confident": (
                True if self.confident is None else bool(self.confident)
            ),
            "confidence": (
                1.0 if self.confidence is None else float(self.confidence)
            ),
            "evidence": self.evidence or {},
        }


def absolute(path: Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a CLI path relative to the project rather than the caller."""

    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def executable_path(path: Path, root: Path = PROJECT_ROOT) -> Path:
    """Make an executable absolute while preserving virtualenv symlinks.

    Resolving ``.venv/bin/python`` to its base interpreter bypasses the
    virtualenv's ``pyvenv.cfg`` and therefore its installed packages.
    """

    path = path.expanduser()
    if not path.is_absolute():
        path = root / path
    return Path(os.path.abspath(path))


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest_rows(manifest: Path) -> dict[str, dict[str, Any]]:
    """Return manifest rows keyed by unique clip ID."""

    payload = json.loads(manifest.read_text())
    result: dict[str, dict[str, Any]] = {}
    for row in manifest_records(payload):
        clip_id = row.get("id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("Every downstream clip needs a non-empty string id")
        clip_id = clip_id.strip()
        if clip_id in result:
            raise ValueError(f"Duplicate clip ID in manifest: {clip_id!r}")
        match_id = row.get("match_id")
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError(f"Clip {clip_id!r} has no match_id")
        raw_half = row.get("half")
        if isinstance(raw_half, bool) or not (
            isinstance(raw_half, int)
            or (
                isinstance(raw_half, str)
                and raw_half.strip() in {"1", "2"}
            )
        ):
            raise ValueError(f"Clip {clip_id!r} has invalid half")
        half = int(raw_half)
        if half not in (1, 2):
            raise ValueError(f"Clip {clip_id!r} half must be 1 or 2")
        normalized = dict(row)
        normalized.update(
            id=clip_id,
            match_id=match_id.strip(),
            half=half,
        )
        result[clip_id] = normalized
    return result


def validate_neutral_registry(
    registry: Path,
    match_ids: Iterable[str],
) -> None:
    """Require every selected match to resolve to neutral unreviewed names."""

    for match_id in sorted(set(match_ids)):
        config = load_match_team_config(registry, match_id)
        if config["mapping_status"] != "unreviewed":
            raise ValueError(
                f"Match {match_id!r} is not unreviewed in {registry}; this "
                "batch deliberately emits neutral Team A/Team B identities"
            )
        if set(config["team_names"].values()) != {"Team A", "Team B"}:
            raise ValueError(
                f"Match {match_id!r} did not resolve to neutral team names"
            )


def raw_direction(value: Any, *, context: str) -> int:
    """Parse one raw pitch direction from a scalar or ``{\"raw\": ...}``."""

    if isinstance(value, dict):
        value = value.get("raw")
    if isinstance(value, bool):
        raise ValueError(f"{context} direction must be -1 or 1")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} direction must be -1 or 1") from error
    if result not in (-1, 1):
        raise ValueError(f"{context} direction must be -1 or 1")
    return result


def direction_pair(payload: Any, *, context: str) -> tuple[int, int] | None:
    """Read internal left/right directions from one optional object."""

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{context} direction override must be an object")
    nested = payload.get("attacking_direction")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError(f"{context}.attacking_direction must be an object")
        payload = nested
    left = payload.get("left", payload.get("left_team_direction"))
    right = payload.get("right", payload.get("right_team_direction"))
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError(f"{context} must provide both left and right directions")
    result = (
        raw_direction(left, context=f"{context}.left"),
        raw_direction(right, context=f"{context}.right"),
    )
    if result[0] == result[1]:
        raise ValueError(f"{context} team directions must be opposite")
    return result


def direction_override(
    row: dict[str, Any],
    registry_payload: dict[str, Any] | None,
) -> tuple[tuple[int, int] | None, str]:
    """Backward-compatible pair view of :func:`resolve_direction_selection`."""

    selection = resolve_direction_selection(row, registry_payload)
    return selection.directions, selection.source


def _direction_confidence(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} confidence must be between 0 and 1")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} confidence must be between 0 and 1"
        ) from error
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"{context} confidence must be between 0 and 1")
    return confidence


def _registry_direction_evidence(
    registry_payload: dict[str, Any],
    half_payload: dict[str, Any],
    *,
    match_id: str,
    half: int,
) -> dict[str, Any]:
    """Keep aggregate evidence without duplicating every clip record per frame."""

    raw_aggregate = half_payload.get("evidence", {})
    if raw_aggregate is None:
        raw_aggregate = {}
    if not isinstance(raw_aggregate, dict):
        raise ValueError(
            f"Direction registry {match_id!r} half {half} evidence "
            "must be an object"
        )
    aggregate = {
        key: value
        for key, value in raw_aggregate.items()
        if key != "clips"
    }
    registry_metadata = {
        key: registry_payload[key]
        for key in (
            "schema_version",
            "generated_at",
            "coordinate_system",
            "method",
        )
        if key in registry_payload
    }
    return {
        "match_id": match_id,
        "half": half,
        "registry": registry_metadata,
        "aggregate": aggregate,
    }


def _nonnegative_summary_int(
    summary: dict[str, Any],
    key: str,
) -> int:
    """Read one exact non-negative integer from a registry summary."""

    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Direction registry summary.{key} must be a non-negative integer"
        )
    return value


def resolve_direction_selection(
    row: dict[str, Any],
    registry_payload: dict[str, Any] | None,
) -> DirectionSelection:
    """Resolve manifest-first internal-team directions and full provenance.

    Accepted manifest forms are ``attacking_direction`` or
    ``left_team_direction``/``right_team_direction``.  A separate direction
    registry may contain
    ``matches[match_id].halves[half].attacking_direction`` (or direct
    ``left``/``right`` values).  These are internal graph-team directions, not
    unreviewed club claims.
    """

    direct = direction_pair(row, context=f"clip {row['id']!r}")
    if direct is not None:
        return DirectionSelection(
            directions=direct,
            source="manifest_match_half_override",
            status="calibrated",
            confident=True,
            confidence=1.0,
            evidence={"origin": "manifest_explicit_direction"},
        )
    if registry_payload is None:
        return DirectionSelection(
            directions=None,
            source="converter_goalkeeper_or_team_median",
        )
    match_id = str(row["match_id"])
    half = int(row["half"])
    try:
        half_payload = registry_payload["matches"][match_id]["halves"][str(half)]
    except (KeyError, TypeError):
        return DirectionSelection(
            directions=None,
            source="converter_goalkeeper_or_team_median",
        )
    if not isinstance(half_payload, dict):
        raise ValueError(
            f"Direction registry {match_id!r} half {half} must be an object"
        )
    configured = direction_pair(
        half_payload,
        context=f"direction registry {match_id!r} half {half}",
    )

    status = half_payload.get(
        "status",
        "calibrated" if configured is not None else "unavailable",
    )
    if not isinstance(status, str) or not status.strip():
        raise ValueError(
            f"Direction registry {match_id!r} half {half} status "
            "must be a non-empty string"
        )
    raw_confident = half_payload.get(
        "direction_confident",
        configured is not None,
    )
    if not isinstance(raw_confident, bool):
        raise ValueError(
            f"Direction registry {match_id!r} half {half} "
            "direction_confident must be boolean"
        )
    if configured is not None and not raw_confident:
        raise ValueError(
            f"Direction registry {match_id!r} half {half} supplies directions "
            "while direction_confident is false"
        )
    if configured is None and raw_confident:
        raise ValueError(
            f"Direction registry {match_id!r} half {half} is marked confident "
            "but supplies no attacking directions"
        )

    raw_confidence = half_payload.get("confidence")
    if raw_confidence is None:
        nested = half_payload.get("attacking_direction", {})
        if nested is None:
            nested = {}
        nested_confidences = [
            value.get("confidence")
            for value in nested.values()
            if isinstance(value, dict) and value.get("confidence") is not None
        ]
        raw_confidence = nested_confidences[0] if nested_confidences else (
            1.0 if configured is not None else 0.0
        )
    confidence = _direction_confidence(
        raw_confidence,
        context=f"direction registry {match_id!r} half {half}",
    )
    evidence = _registry_direction_evidence(
        registry_payload,
        half_payload,
        match_id=match_id,
        half=half,
    )
    if configured is None:
        if status.strip().lower() == "calibrated":
            raise ValueError(
                f"Direction registry {match_id!r} half {half} has calibrated "
                "status but supplies no attacking directions"
            )
        return DirectionSelection(
            directions=None,
            source="direction_registry_abstained_converter_inference",
            status=status.strip(),
            confident=False,
            confidence=confidence,
            evidence=evidence,
        )
    return DirectionSelection(
        directions=configured,
        source="direction_registry_match_half_override",
        status=status.strip(),
        confident=True,
        confidence=confidence,
        evidence=evidence,
    )


def validate_final_direction_registry(
    payload: Any,
    rows: dict[str, dict[str, Any]],
) -> None:
    """Require a complete generated registry covering the whole manifest."""

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Direction registry must be a schema_version 1 object")
    if not isinstance(payload.get("matches"), dict):
        raise ValueError("Direction registry matches must be an object")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            "Direction registry has no completion summary; run "
            "calibrate_attack_directions.py after GSR finishes"
        )
    selected = _nonnegative_summary_int(summary, "clips_selected")
    recorded = _nonnegative_summary_int(summary, "clips_recorded")
    waiting = _nonnegative_summary_int(
        summary, "clips_waiting_for_state"
    )
    accounted = _nonnegative_summary_int(summary, "clips_accounted_for")
    if selected != len(rows):
        raise ValueError(
            "Direction registry was not generated for this complete manifest: "
            f"registry={selected}, manifest={len(rows)} clips"
        )
    if waiting or recorded != selected or accounted != selected:
        raise ValueError(
            "Direction registry is not final: "
            f"selected={selected}, recorded={recorded}, waiting={waiting}, "
            f"accounted={accounted}"
        )

    expected_halves = {
        (str(row["match_id"]).strip(), int(row["half"]))
        for row in rows.values()
    }
    declared_halves = _nonnegative_summary_int(summary, "match_halves")
    if declared_halves != len(expected_halves):
        raise ValueError(
            "Direction registry match-half count does not match the manifest: "
            f"registry={declared_halves}, manifest={len(expected_halves)}"
        )
    for match_id, half in sorted(expected_halves):
        try:
            half_payload = payload["matches"][match_id]["halves"][str(half)]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Direction registry is missing {match_id!r} half {half}"
            ) from error
        if not isinstance(half_payload, dict):
            raise ValueError(
                f"Direction registry {match_id!r} half {half} must be an object"
            )
        resolve_direction_selection(
            {
                "id": f"registry-preflight:{match_id}:h{half}",
                "match_id": match_id,
                "half": half,
            },
            payload,
        )


def conversion_signature(
    command: list[str],
    selection: DirectionSelection,
) -> str:
    """Fingerprint every non-file conversion input used for graph reuse."""

    payload = {
        "schema_version": 1,
        "command": command,
        "direction_selection": {
            "directions": (
                list(selection.directions)
                if selection.directions is not None
                else None
            ),
            "source": selection.source,
            "status": selection.status,
            "confident": selection.confident,
            "confidence": selection.confidence,
            "evidence": selection.evidence,
        },
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_count(graph_path: Path) -> int:
    """Validate a graph NPZ/JSONL pair and return its number of frames."""

    metadata_path = graph_path.with_suffix(".jsonl")
    if not graph_path.is_file() or not metadata_path.is_file():
        raise ValueError("graph NPZ or JSONL is missing")
    with np.load(graph_path, allow_pickle=False) as payload:
        if set(payload.files) < {"features", "masks"}:
            raise ValueError("graph NPZ is missing features or masks")
        features = payload["features"]
        masks = payload["masks"]
        if (
            features.ndim != 3
            or masks.ndim != 2
            or len(features) == 0
            or features.shape[:2] != masks.shape
        ):
            raise ValueError("graph arrays have invalid or inconsistent shapes")
        count = len(features)
    rows = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != count:
        raise ValueError(
            f"graph metadata has {len(rows)} rows for {count} graphs"
        )
    return count


def graph_count_with_provenance(
    graph_path: Path,
    *,
    sequence_id: str,
    match_id: str,
    selection: DirectionSelection,
) -> int:
    """Validate graph identity and the direction metadata used to create it."""

    count = graph_count(graph_path)
    rows = [
        json.loads(line)
        for line in graph_path.with_suffix(".jsonl").read_text().splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        context = f"{graph_path} metadata row {index + 1}"
        if row.get("sequence") != sequence_id:
            raise ValueError(f"{context} has the wrong sequence")
        if row.get("team_identity_match_id") != match_id:
            raise ValueError(f"{context} has the wrong match identity")
        if row.get("team_identity_status") != "unreviewed":
            raise ValueError(f"{context} is not neutral/unreviewed")
        raw = row.get("attacking_direction_raw")
        if isinstance(raw, bool) or raw not in (-1, 1):
            raise ValueError(f"{context} has an invalid attacking direction")
        expected_label = (
            "left_to_right" if int(raw) == 1 else "right_to_left"
        )
        if row.get("attacking_direction_label") != expected_label:
            raise ValueError(f"{context} has inconsistent direction metadata")
        if selection.directions is None:
            if not isinstance(row.get("direction_confident"), bool):
                raise ValueError(
                    f"{context} has no boolean direction confidence"
                )
            if not str(row.get("direction_source", "")).strip():
                raise ValueError(f"{context} has no direction source")
            if not str(row.get("direction_status", "")).strip():
                raise ValueError(f"{context} has no direction status")
            continue

        possession_team = row.get("possession_team")
        if possession_team not in {"left", "right"}:
            raise ValueError(f"{context} has an invalid possession team")
        team_index = 0 if possession_team == "left" else 1
        if int(raw) != selection.directions[team_index]:
            raise ValueError(
                f"{context} does not use the selected match-half direction"
            )
        if row.get("direction_source") != selection.source:
            raise ValueError(f"{context} has the wrong direction source")
        if row.get("direction_status") != selection.status:
            raise ValueError(f"{context} has the wrong direction status")
        if row.get("direction_confident") is not True:
            raise ValueError(f"{context} is not direction-confident")
        try:
            confidence = float(row["direction_confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{context} has invalid direction confidence"
            ) from error
        if (
            selection.confidence is None
            or not np.isclose(
                confidence,
                selection.confidence,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"{context} has the wrong direction confidence")
    return count


def weak_count(weak_path: Path, expected: int) -> int:
    """Validate a weak-label NPZ/JSONL pair."""

    metadata_path = weak_path.with_suffix(".jsonl")
    if not weak_path.is_file() or not metadata_path.is_file():
        raise ValueError("weak-label NPZ or JSONL is missing")
    with np.load(weak_path, allow_pickle=False) as payload:
        if set(payload.files) < {"labels", "confidence", "label_names"}:
            raise ValueError("weak-label NPZ has missing arrays")
        count = len(payload["labels"])
        if len(payload["confidence"]) != count:
            raise ValueError("weak labels and confidence lengths disagree")
    rows = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    if count != expected or len(rows) != expected:
        raise ValueError(
            f"weak labels have {count} arrays/{len(rows)} rows; "
            f"expected {expected}"
        )
    return count


def prediction_count(prediction_path: Path, expected: int) -> int:
    """Validate classifier JSONL output."""

    if not prediction_path.is_file():
        raise ValueError("prediction JSONL is missing")
    rows = [
        json.loads(line)
        for line in prediction_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise ValueError(
            f"predictions have {len(rows)} rows; expected {expected}"
        )
    if any(
        "predicted_situation" not in row
        or "confidence" not in row
        or "probabilities" not in row
        for row in rows
    ):
        raise ValueError("prediction rows are missing classifier fields")
    return len(rows)


def artifacts_current(
    outputs: Iterable[Path],
    dependencies: Iterable[Path],
    validator,
) -> tuple[bool, str]:
    """Check both artifact integrity and dependency mtimes."""

    outputs = list(outputs)
    dependencies = list(dependencies)
    try:
        validator()
    except Exception as error:
        return False, str(error)
    existing_dependencies = [path for path in dependencies if path.is_file()]
    if len(existing_dependencies) != len(dependencies):
        missing = [
            str(path) for path in dependencies if not path.is_file()
        ]
        return False, f"missing dependencies: {', '.join(missing)}"
    if min(path.stat().st_mtime_ns for path in outputs) < max(
        path.stat().st_mtime_ns for path in existing_dependencies
    ):
        return False, "artifacts are older than their inputs"
    return True, "valid current artifacts"


def converter_command(
    *,
    tracklab_python: Path,
    state: Path,
    video: Path,
    yolo: Path,
    graph: Path,
    sequence_id: str,
    team_registry: Path,
    match_id: str,
    directions: tuple[int, int] | None,
    ball_confidence: float,
    direction_provenance: dict[str, Any] | None = None,
) -> list[str]:
    """Build the neutral, non-legacy TrackLab-state conversion command."""

    command = [
        str(tracklab_python),
        str(PROJECT_ROOT / "scripts/convert_tracklab_state.py"),
        "--state",
        str(state),
        "--video",
        str(video),
        "--yolo",
        str(yolo),
        "--output",
        str(graph),
        "--ball-confidence",
        str(ball_confidence),
        "--recluster-teams",
        "--sequence-id",
        sequence_id,
        "--match-registry",
        str(team_registry),
        "--match-id",
        match_id,
        "--disable-team-labels",
        "--disable-team-model",
    ]
    if directions is not None:
        command.extend(
            [
                "--left-team-direction",
                str(directions[0]),
                "--right-team-direction",
                str(directions[1]),
            ]
        )
        if direction_provenance is not None:
            command.extend(
                [
                    "--direction-provenance-json",
                    json.dumps(
                        direction_provenance,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ]
            )
    elif direction_provenance is not None:
        raise ValueError(
            "Direction provenance cannot be supplied without directions"
        )
    return command


def run_command(
    stage: str,
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
) -> None:
    """Run one stage while retaining a complete per-clip log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{iso_now()}] stage={stage}\ncommand={shlex.join(command)}\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"returncode={completed.returncode}\n")
    if completed.returncode:
        raise StageFailure(stage, completed.returncode, log_path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def status_payload(
    existing: dict[str, Any] | None,
    *,
    manifest: Path,
    state_root: Path,
    output_dir: Path,
    team_registry: Path,
    direction_registry: Path | None = None,
) -> dict[str, Any]:
    payload = existing if isinstance(existing, dict) else {}
    payload.update(
        schema_version=1,
        manifest=str(manifest),
        state_root=str(state_root),
        output_dir=str(output_dir),
        team_registry=str(team_registry),
        direction_registry=(
            str(direction_registry) if direction_registry is not None else None
        ),
        updated_at=iso_now(),
    )
    if not isinstance(payload.get("clips"), dict):
        payload["clips"] = {}
    return payload


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"Status file must contain an object: {path}")
    return payload


def stage_plan(
    *,
    graph: Path,
    weak: Path,
    predictions: Path,
    graph_dependencies: list[Path],
    weak_dependencies: list[Path],
    prediction_dependencies: list[Path],
    graph_validator=None,
    conversion_signature_matches: bool = True,
) -> tuple[dict[str, str], int | None]:
    """Return run/skip decisions for all stages and known graph count."""

    if graph_validator is None:
        graph_validator = lambda: graph_count(graph)
    graph_ok, graph_reason = artifacts_current(
        [graph, graph.with_suffix(".jsonl")],
        graph_dependencies,
        graph_validator,
    )
    if graph_ok and not conversion_signature_matches:
        graph_ok = False
        graph_reason = "conversion configuration signature changed or is absent"
    count: int | None = graph_validator() if graph_ok else None
    if graph_ok:
        weak_ok, weak_reason = artifacts_current(
            [weak, weak.with_suffix(".jsonl")],
            weak_dependencies,
            lambda: weak_count(weak, int(count)),
        )
        prediction_ok, prediction_reason = artifacts_current(
            [predictions],
            prediction_dependencies,
            lambda: prediction_count(predictions, int(count)),
        )
    else:
        weak_ok = prediction_ok = False
        weak_reason = prediction_reason = "graph conversion will run"
    return (
        {
            "convert": "skip" if graph_ok else f"run: {graph_reason}",
            "weak_labels": "skip" if weak_ok else f"run: {weak_reason}",
            "classify": (
                "skip" if prediction_ok else f"run: {prediction_reason}"
            ),
        },
        count,
    )


def summarize(payload: dict[str, Any]) -> dict[str, int]:
    counts = Counter(
        row.get("status", "unknown") for row in payload["clips"].values()
    )
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert completed Arsenal-expansion GSR states to graphs, weak "
            "labels, and tactical predictions."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--team-registry", type=Path, default=DEFAULT_TEAM_REGISTRY
    )
    parser.add_argument(
        "--direction-registry",
        type=Path,
        default=DEFAULT_DIRECTION_REGISTRY,
        help=(
            "Final calibrate_attack_directions.py output; confident "
            "match-half directions are applied and abstained halves retain "
            "audited converter inference"
        ),
    )
    parser.add_argument(
        "--no-direction-registry",
        action="store_true",
        help=(
            "Explicitly disable the generated registry and infer every clip "
            "inside the converter"
        ),
    )
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--model", type=Path, default=DEFAULT_GRAPH_MODEL)
    parser.add_argument(
        "--tracklab-python", type=Path, default=DEFAULT_TRACKLAB_PYTHON
    )
    parser.add_argument("--ml-python", type=Path, default=DEFAULT_ML_PYTHON)
    parser.add_argument("--ball-confidence", type=float, default=0.03)
    parser.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help="Process only this exact manifest clip ID; repeat as needed",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plans without writing artifacts or status",
    )
    args = parser.parse_args()
    if args.ball_confidence <= 0 or args.ball_confidence >= 1:
        parser.error("--ball-confidence must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    manifest = absolute(args.manifest)
    path_root = absolute(args.path_root)
    state_root = absolute(args.state_root)
    output_dir = absolute(args.output_dir)
    status_path = absolute(args.status)
    log_dir = absolute(args.log_dir)
    team_registry = absolute(args.team_registry)
    direction_registry = (
        absolute(args.direction_registry)
        if args.direction_registry is not None
        and not args.no_direction_registry
        else None
    )
    yolo = absolute(args.yolo)
    model = absolute(args.model)
    tracklab_python = executable_path(args.tracklab_python)
    ml_python = executable_path(args.ml_python)

    rows = load_manifest_rows(manifest)
    jobs = load_jobs(manifest, path_root)
    if args.clip_id:
        requested = set(args.clip_id)
        unknown = requested - set(rows)
        if unknown:
            raise ValueError(f"Unknown clip IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if job.id in requested]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No clips selected")

    required_inputs = [
        team_registry,
        yolo,
        model,
        tracklab_python,
        ml_python,
    ]
    if direction_registry is not None:
        required_inputs.append(direction_registry)
    for required in required_inputs:
        if not required.is_file():
            raise FileNotFoundError(f"Required downstream input is missing: {required}")
    validate_neutral_registry(
        team_registry,
        [str(rows[job.id]["match_id"]) for job in jobs],
    )
    direction_payload = (
        json.loads(direction_registry.read_text())
        if direction_registry is not None
        else None
    )
    if direction_payload is not None:
        validate_final_direction_registry(direction_payload, rows)
    direction_selections = {
        job.id: resolve_direction_selection(rows[job.id], direction_payload)
        for job in jobs
    }

    converter_script = PROJECT_ROOT / "scripts/convert_tracklab_state.py"
    weak_script = PROJECT_ROOT / "scripts/derive_weak_tactical_labels.py"
    classifier_script = PROJECT_ROOT / "scripts/classify_gamestate_graphs.py"
    team_script = PROJECT_ROOT / "scripts/team_identity.py"
    direction_script = PROJECT_ROOT / "scripts/attack_direction.py"
    classifier_definition = PROJECT_ROOT / "scripts/train_graph_classifier.py"
    for required_script in (
        converter_script,
        weak_script,
        classifier_script,
        team_script,
        direction_script,
        classifier_definition,
    ):
        if not required_script.is_file():
            raise FileNotFoundError(
                f"Required downstream script is missing: {required_script}"
            )

    status = status_payload(
        read_status(status_path),
        manifest=manifest,
        state_root=state_root,
        output_dir=output_dir,
        team_registry=team_registry,
        direction_registry=direction_registry,
    )
    status["clips"] = {
        clip_id: value
        for clip_id, value in status["clips"].items()
        if clip_id in rows and isinstance(value, dict)
    }
    failures = 0
    waiting = 0
    for index, job in enumerate(jobs, start=1):
        row = rows[job.id]
        safe_id = safe_identifier(job.id)
        graph = output_dir / f"{safe_id}.npz"
        weak = output_dir / f"{safe_id}-weak.npz"
        predictions = output_dir / f"{safe_id}-predictions.jsonl"
        log_path = log_dir / f"{safe_id}.log"
        direction_selection = direction_selections[job.id]
        directions = direction_selection.directions
        direction_source = direction_selection.source
        direction_provenance = direction_selection.converter_provenance()
        prefix = f"[{index}/{len(jobs)}] {job.id}"
        try:
            state = find_completed_state(state_root, job)
        except Exception as error:
            failures += 1
            print(
                f"{prefix}: FAILED state discovery: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if not args.dry_run:
                status["clips"][job.id] = {
                    "id": job.id,
                    "match_id": row["match_id"],
                    "half": int(row["half"]),
                    "clip_path": str(job.clip_path),
                    "status": "failed",
                    "failed_stage": "state_discovery",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": iso_now(),
                    "direction_source": direction_source,
                    "direction_status": direction_selection.status,
                    "direction_confident": direction_selection.confident,
                    "direction_confidence": direction_selection.confidence,
                    "direction_evidence": direction_selection.evidence,
                }
                status["updated_at"] = iso_now()
                status["summary"] = summarize(status)
                atomic_json(status_path, status)
            continue
        if state is None:
            waiting += 1
            print(f"{prefix}: waiting_for_state")
            if not args.dry_run:
                status["clips"][job.id] = {
                    "id": job.id,
                    "match_id": row["match_id"],
                    "half": int(row["half"]),
                    "status": "waiting_for_state",
                    "updated_at": iso_now(),
                    "direction_source": direction_source,
                    "direction_status": direction_selection.status,
                    "direction_confident": direction_selection.confident,
                    "direction_confidence": direction_selection.confidence,
                    "direction_evidence": direction_selection.evidence,
                }
                status["updated_at"] = iso_now()
                status["summary"] = summarize(status)
                atomic_json(status_path, status)
            continue

        graph_dependencies = [
            state,
            job.clip_path,
            yolo,
            team_registry,
            converter_script,
            team_script,
            direction_script,
        ]
        if direction_registry is not None:
            graph_dependencies.append(direction_registry)
        weak_dependencies = [
            graph,
            graph.with_suffix(".jsonl"),
            weak_script,
        ]
        prediction_dependencies = [
            graph,
            graph.with_suffix(".jsonl"),
            model,
            classifier_script,
            classifier_definition,
        ]
        conversion = converter_command(
            tracklab_python=tracklab_python,
            state=state,
            video=job.clip_path,
            yolo=yolo,
            graph=graph,
            sequence_id=job.id,
            team_registry=team_registry,
            match_id=str(row["match_id"]),
            directions=directions,
            ball_confidence=args.ball_confidence,
            direction_provenance=direction_provenance,
        )
        expected_conversion_signature = conversion_signature(
            conversion,
            direction_selection,
        )
        previous_clip_status = status["clips"].get(job.id)
        signature_matches = (
            isinstance(previous_clip_status, dict)
            and previous_clip_status.get("conversion_signature")
            == expected_conversion_signature
        )
        def graph_validator() -> int:
            return graph_count_with_provenance(
                graph,
                sequence_id=job.id,
                match_id=str(row["match_id"]),
                selection=direction_selection,
            )
        try:
            plan, existing_count = stage_plan(
                graph=graph,
                weak=weak,
                predictions=predictions,
                graph_dependencies=graph_dependencies,
                weak_dependencies=weak_dependencies,
                prediction_dependencies=prediction_dependencies,
                graph_validator=graph_validator,
                conversion_signature_matches=signature_matches,
            )
        except Exception as error:
            failures += 1
            print(
                f"{prefix}: FAILED artifact preflight: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if not args.dry_run:
                status["clips"][job.id] = {
                    "id": job.id,
                    "match_id": row["match_id"],
                    "half": int(row["half"]),
                    "clip_path": str(job.clip_path),
                    "state_path": str(state),
                    "status": "failed",
                    "failed_stage": "artifact_preflight",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": iso_now(),
                    "direction_source": direction_source,
                    "direction_status": direction_selection.status,
                    "direction_confident": direction_selection.confident,
                    "direction_confidence": direction_selection.confidence,
                    "direction_evidence": direction_selection.evidence,
                }
                status["updated_at"] = iso_now()
                status["summary"] = summarize(status)
                atomic_json(status_path, status)
            continue
        if args.force:
            plan = {
                "convert": "run: --force",
                "weak_labels": "run: --force",
                "classify": "run: --force",
            }
            existing_count = None
        print(
            f"{prefix}: state={state}; "
            + ", ".join(f"{key}={value}" for key, value in plan.items())
        )
        weak_command = [
            str(ml_python),
            str(weak_script),
            "--graphs",
            str(graph),
            "--output",
            str(weak),
        ]
        classify_command = [
            str(ml_python),
            str(classifier_script),
            "--graphs",
            str(graph),
            "--model",
            str(model),
            "--output",
            str(predictions),
        ]
        if args.dry_run:
            for stage, command in (
                ("convert", conversion),
                ("weak_labels", weak_command),
                ("classify", classify_command),
            ):
                if plan[stage].startswith("run:"):
                    print(f"  {stage}: {shlex.join(command)}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        clip_status: dict[str, Any] = {
            "id": job.id,
            "match_id": row["match_id"],
            "half": int(row["half"]),
            "clip_path": str(job.clip_path),
            "state_path": str(state),
            "direction_source": direction_source,
            "direction_status": direction_selection.status,
            "direction_confident": direction_selection.confident,
            "direction_confidence": direction_selection.confidence,
            "direction_evidence": direction_selection.evidence,
            "directions": (
                {"left": directions[0], "right": directions[1]}
                if directions is not None
                else None
            ),
            "team_identity": "unreviewed_neutral_team_a_team_b",
            "log_path": str(log_path),
            "updated_at": iso_now(),
            "stages": {},
        }
        try:
            if plan["convert"].startswith("run:"):
                run_command(
                    "convert",
                    conversion,
                    log_path=log_path,
                    cwd=PROJECT_ROOT,
                )
                count = graph_validator()
                clip_status["stages"]["convert"] = "completed"
            else:
                count = int(existing_count)
                clip_status["stages"]["convert"] = "skipped_current"
            clip_status["conversion_signature"] = (
                expected_conversion_signature
            )

            if plan["weak_labels"].startswith("run:"):
                run_command(
                    "weak_labels",
                    weak_command,
                    log_path=log_path,
                    cwd=PROJECT_ROOT,
                )
                weak_count(weak, count)
                clip_status["stages"]["weak_labels"] = "completed"
            else:
                weak_count(weak, count)
                clip_status["stages"]["weak_labels"] = "skipped_current"

            if plan["classify"].startswith("run:"):
                run_command(
                    "classify",
                    classify_command,
                    log_path=log_path,
                    cwd=PROJECT_ROOT,
                )
                prediction_count(predictions, count)
                clip_status["stages"]["classify"] = "completed"
            else:
                prediction_count(predictions, count)
                clip_status["stages"]["classify"] = "skipped_current"

            clip_status.update(
                status="completed",
                graph_count=count,
                graph_path=str(graph),
                weak_labels_path=str(weak),
                predictions_path=str(predictions),
                updated_at=iso_now(),
            )
        except Exception as error:
            failures += 1
            clip_status.update(
                status="failed",
                failed_stage=(
                    error.stage
                    if isinstance(error, StageFailure)
                    else next(
                        (
                            name
                            for name in ("convert", "weak_labels", "classify")
                            if name not in clip_status["stages"]
                        ),
                        "validation",
                    )
                ),
                error=f"{type(error).__name__}: {error}",
                updated_at=iso_now(),
            )
            print(f"{prefix}: FAILED: {clip_status['error']}", file=sys.stderr)
        status["clips"][job.id] = clip_status
        status["updated_at"] = iso_now()
        status["summary"] = summarize(status)
        atomic_json(status_path, status)

    if args.dry_run:
        print(
            f"Dry run: {len(jobs) - waiting - failures} ready, "
            f"{waiting} waiting, {failures} failed, {len(jobs)} selected."
        )
        return 1 if failures else 0
    status["updated_at"] = iso_now()
    status["summary"] = summarize(status)
    atomic_json(status_path, status)
    print(
        f"Downstream batch: {len(jobs) - waiting - failures} completed, "
        f"{waiting} waiting, {failures} failed. Status: {status_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
