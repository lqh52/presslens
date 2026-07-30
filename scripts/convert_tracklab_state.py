#!/usr/bin/env python3
"""Convert an official TrackLab GSR state plus recovered ball detections to graphs.

Run this script in the isolated sn-gamestate environment, which provides
TrackLab's pandas pickle dependencies and Ultralytics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from attack_direction import calibrate_detections
except ModuleNotFoundError:
    from scripts.attack_direction import calibrate_detections

try:
    from team_identity import (
        explicit_match_team_config,
        infer_cluster_to_internal_from_tracklab_team,
        load_match_team_config,
        neutral_match_team_config,
        resolve_team_assignments,
    )
except ModuleNotFoundError:
    from scripts.team_identity import (
        explicit_match_team_config,
        infer_cluster_to_internal_from_tracklab_team,
        load_match_team_config,
        neutral_match_team_config,
        resolve_team_assignments,
    )


PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
LEGACY_TEAM_LABELS = Path("data/review/team_tracks/labels.json")
LEGACY_TEAM_MODEL = Path("models/team_identity_burnley_arsenal.npz")


def explicit_direction_metadata(
    direction: dict[str, int],
    provenance: dict | None = None,
) -> dict:
    """Normalize metadata for explicit directions passed on the CLI.

    Direct manual CLI directions retain the historical confidence of 1.0.
    Pipeline callers can attach calibrated registry provenance so the original
    aggregate confidence and evidence are not overstated.
    """

    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("Direction provenance must be a JSON object")
    source = provenance.get("source", "match_half_metadata")
    status = provenance.get("status", "calibrated")
    confident = provenance.get("confident", True)
    confidence = provenance.get("confidence", 1.0)
    evidence = provenance.get("evidence", {})
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Direction provenance source must be non-empty")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("Direction provenance status must be non-empty")
    if not isinstance(confident, bool):
        raise ValueError("Direction provenance confident must be boolean")
    if not confident:
        raise ValueError(
            "Explicit directions cannot carry non-confident provenance"
        )
    if isinstance(confidence, bool):
        raise ValueError("Direction provenance confidence must be in [0, 1]")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Direction provenance confidence must be in [0, 1]"
        ) from error
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Direction provenance confidence must be in [0, 1]")
    if not isinstance(evidence, dict):
        raise ValueError("Direction provenance evidence must be an object")
    normalized_direction = {
        team: int(direction[team]) for team in ("left", "right")
    }
    direction_evidence = {
        "status": status.strip(),
        "confident": True,
        "confidence": confidence,
        "directions": normalized_direction,
        "source": source.strip(),
        "method": (
            "explicit_cli_match_half_metadata"
            if not provenance
            else "explicit_cli_with_preserved_provenance"
        ),
    }
    if evidence:
        direction_evidence["provenance"] = evidence
    return {
        "source": source.strip(),
        "status": status.strip(),
        "confident": True,
        "confidence": confidence,
        "evidence": direction_evidence,
    }


def homography(parameters: dict) -> np.ndarray:
    calibration = np.array(
        [
            [parameters["x_focal_length"], 0, parameters["principal_point"][0]],
            [0, parameters["y_focal_length"], parameters["principal_point"][1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(parameters["rotation_matrix"], dtype=np.float64)
    position = np.asarray(parameters["position_meters"], dtype=np.float64)
    return calibration @ rotation @ np.concatenate(
        (np.eye(3)[:, :2], -position.reshape(3, 1)), axis=1
    )


def unproject(parameters: dict, image_point: tuple[float, float]) -> np.ndarray | None:
    point = np.linalg.inv(homography(parameters)) @ np.array([*image_point, 1.0])
    if abs(point[2]) < 1e-8:
        return None
    point = point[:2] / point[2]
    if not np.isfinite(point).all() or abs(point[0]) > 54.5 or abs(point[1]) > 36:
        return None
    return point.astype(np.float32)


def detect_balls(video: Path, weights: Path, frames: int, confidence: float) -> dict[int, dict]:
    capture = cv2.VideoCapture(str(video))
    images = []
    for _ in range(frames):
        ok, image = capture.read()
        if not ok:
            break
        images.append(image)
    model = YOLO(str(weights))
    detections = {}
    for start in range(0, len(images), 16):
        results = model(
            images[start : start + 16],
            verbose=False,
            conf=confidence,
            classes=[32],
        )
        for offset, result in enumerate(results):
            boxes = result.boxes.cpu().numpy()
            if not len(boxes):
                continue
            box = max(boxes, key=lambda value: float(value.conf[0]))
            left, top, right, bottom = map(float, box.xyxy[0])
            detections[start + offset] = {
                "confidence": float(box.conf[0]),
                "image_xy": [(left + right) / 2, bottom],
            }
    return detections


def recluster_player_teams(
    video: Path,
    detections,
    *,
    sequence_id: str | None = None,
    labels_path: Path | None = Path("data/review/team_tracks/labels.json"),
    model_path: Path | None = Path("models/team_identity_burnley_arsenal.npz"),
    model_threshold: float | None = None,
    club_to_internal: dict[str, str] | None = None,
    cluster_to_internal: dict[int, str] | None = None,
    tracklab_side_to_internal: dict[str, str] | None = None,
) -> tuple[dict[int, str], dict]:
    """Stabilize learned team clusters and correct only clear colour outliers.

    TrackLab provides a track-level appearance cluster. With an explicit
    ``cluster_to_internal`` mapping, the cluster IDs are preserved as neutral or
    registry-backed teams. Without it, the original Burnley-Arsenal kit rule is
    retained for the existing demo. An individual outfield track is reassigned
    only when its median shirt chroma is decisively closer to the opposite
    cluster prototype. Goalkeepers retain their learned cluster because their
    kits use different colours.
    """
    athletes = detections[detections.role.isin(["player", "goalkeeper"])]
    players = athletes[athletes.role == "player"]
    wanted_frames = set(map(int, players.image_id.unique()))
    capture = cv2.VideoCapture(str(video))
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    frame = 0
    while frame <= max(wanted_frames, default=-1):
        ok, image = capture.read()
        if not ok:
            break
        if frame in wanted_frames:
            height, width = image.shape[:2]
            for _, row in players[players.image_id == frame].iterrows():
                left, top, box_width, box_height = map(float, row.bbox_ltwh)
                x1 = max(0, int(left + 0.2 * box_width))
                x2 = min(width, int(left + 0.8 * box_width))
                y1 = max(0, int(top + 0.12 * box_height))
                y2 = min(height, int(top + 0.52 * box_height))
                crop = image[y1:y2, x1:x2]
                if crop.size < 30:
                    continue
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                samples[int(row.track_id)].append(
                    np.median(lab.reshape(-1, 3), axis=0)[1:].astype(np.float32)
                )
        frame += 1
    capture.release()

    track_colours = {
        track_id: np.median(values, axis=0)
        for track_id, values in samples.items()
        if len(values) >= 2
    }
    track_clusters: dict[int, int] = {}
    track_roles: dict[int, str] = {}
    original_assignments: dict[int, str] = {}
    for track_id, rows in athletes.groupby("track_id"):
        integer_track_id = int(track_id)
        track_roles[integer_track_id] = str(rows.role.mode().iloc[0])
        labels = rows.team_cluster.dropna() if "team_cluster" in rows else []
        if len(labels):
            track_clusters[integer_track_id] = int(labels.mode().iloc[0])
        teams = rows.team.dropna() if "team" in rows else []
        if len(teams):
            team = str(teams.mode().iloc[0])
            if team in {"left", "right"}:
                original_assignments[integer_track_id] = team

    cluster_colours = {
        cluster: [
            track_colours[track_id]
            for track_id, value in track_clusters.items()
            if value == cluster and track_id in track_colours
        ]
        for cluster in (0, 1)
    }
    if cluster_to_internal is not None and (
        set(cluster_to_internal) != {0, 1}
        or set(cluster_to_internal.values()) != {"left", "right"}
    ):
        raise ValueError(
            "cluster_to_internal must map clusters 0/1 bijectively to left/right"
        )
    if tracklab_side_to_internal is not None and (
        set(tracklab_side_to_internal) != {"left", "right"}
        or set(tracklab_side_to_internal.values()) != {"left", "right"}
    ):
        raise ValueError(
            "tracklab_side_to_internal must map left/right bijectively to "
            "left/right"
        )
    if tracklab_side_to_internal is not None:
        original_assignments = {
            track_id: tracklab_side_to_internal[team]
            for track_id, team in original_assignments.items()
        }
    if min(map(len, cluster_colours.values())) < 2:
        if cluster_to_internal is not None:
            original_assignments.update(
                {
                    track_id: cluster_to_internal[cluster]
                    for track_id, cluster in track_clusters.items()
                }
            )
        assignments, identity_evidence = resolve_team_assignments(
            original_assignments,
            detections,
            sequence_id=sequence_id or video.stem,
            club_to_internal=club_to_internal or {},
            labels_path=labels_path,
            model_path=model_path,
            threshold=model_threshold,
            strict_model_classes=cluster_to_internal is not None,
        )
        return assignments, {
            "tracks_with_colour": len(track_colours),
            "tracks_assigned": len(assignments),
            "colour_outliers_corrected": 0,
            "prototype_separation_lab": None,
            "assignment_basis": (
                "TrackLab team clusters; colour prototypes unavailable"
                if cluster_to_internal is not None
                else "Existing TrackLab team labels; colour prototypes unavailable"
            ),
            "cluster_to_internal": cluster_to_internal,
            "identity_resolution": identity_evidence,
        }
    prototypes = {
        cluster: np.median(np.asarray(values), axis=0)
        for cluster, values in cluster_colours.items()
    }
    legacy_kit_mapping = cluster_to_internal is None
    if legacy_kit_mapping:
        arsenal_cluster = max(
            prototypes, key=lambda cluster: float(prototypes[cluster][1])
        )
        burnley_cluster = 1 - arsenal_cluster
        semantic_cluster = {arsenal_cluster: "left", burnley_cluster: "right"}
    else:
        semantic_cluster = dict(cluster_to_internal)

    assignments: dict[int, str] = dict(original_assignments)
    corrected = 0
    for track_id, cluster in track_clusters.items():
        assigned_cluster = cluster
        if track_roles.get(track_id) == "player" and track_id in track_colours:
            colour = track_colours[track_id]
            own_distance = float(np.linalg.norm(colour - prototypes[cluster]))
            other_cluster = 1 - cluster
            other_distance = float(np.linalg.norm(colour - prototypes[other_cluster]))
            if other_distance + 8.0 < own_distance and other_distance < own_distance * 0.60:
                assigned_cluster = other_cluster
                corrected += 1
        assignments[track_id] = semantic_cluster[assigned_cluster]

    for track_id, colour in track_colours.items():
        if track_id in assignments:
            continue
        nearest = min(prototypes, key=lambda cluster: float(np.linalg.norm(colour - prototypes[cluster])))
        assignments[track_id] = semantic_cluster[nearest]

    assignments, identity_evidence = resolve_team_assignments(
        assignments,
        detections,
        sequence_id=sequence_id or video.stem,
        club_to_internal=club_to_internal or {},
        labels_path=labels_path,
        model_path=model_path,
        threshold=model_threshold,
        strict_model_classes=cluster_to_internal is not None,
    )
    evidence = {
        "tracks_with_colour": len(track_colours),
        "tracks_assigned": len(assignments),
        "colour_outliers_corrected": corrected,
        "prototype_separation_lab": round(
            float(np.linalg.norm(prototypes[0] - prototypes[1])), 3
        ),
        "cluster_to_internal": semantic_cluster,
        "cluster_prototypes_lab_ab": {
            str(cluster): prototype.round(3).tolist()
            for cluster, prototype in prototypes.items()
        },
        "assignment_basis": (
            "Registry/explicit TrackLab cluster mapping + decisive Lab-chroma "
            "outlier correction"
            if not legacy_kit_mapping
            else "TrackLab track cluster + decisive Lab-chroma outlier correction"
        ),
        "identity_resolution": identity_evidence,
    }
    if legacy_kit_mapping:
        evidence.update(
            arsenal_cluster=arsenal_cluster,
            arsenal_prototype_lab_ab=prototypes[arsenal_cluster].round(3).tolist(),
            burnley_prototype_lab_ab=prototypes[burnley_cluster].round(3).tolist(),
        )
    return assignments, evidence


def apply_resolved_team_assignments(
    detections,
    assignments: dict[int, str],
    *,
    neutral_fail_closed: bool,
) -> dict:
    """Apply one track-level map and audit graph-eligible athlete tracks.

    For unreviewed neutral clips the persisted assignment map is the only
    authority shared with the overlay renderer. An athlete missing from that
    map is therefore excluded instead of silently retaining a raw TrackLab
    side that the overlay cannot reproduce.
    """

    athlete_rows = detections.role.isin(["player", "goalkeeper"])

    def assignment(track_id):
        try:
            return assignments.get(int(track_id), np.nan)
        except (TypeError, ValueError):
            return np.nan

    mapped = detections.loc[athlete_rows, "track_id"].map(assignment)
    missing_track_ids = set()
    for track_id, missing in zip(
        detections.loc[athlete_rows, "track_id"],
        mapped.isna(),
    ):
        if not missing:
            continue
        try:
            missing_track_ids.add(int(track_id))
        except (TypeError, ValueError):
            continue
    missing_track_ids = sorted(missing_track_ids)
    if neutral_fail_closed:
        detections.loc[athlete_rows, "team"] = mapped.fillna("ignore")
    else:
        detections.loc[athlete_rows, "team"] = mapped.fillna(
            detections.loc[athlete_rows, "team"]
        )

    graph_side_tracks = {
        int(track_id)
        for track_id in detections.loc[
            athlete_rows
            & detections.team.isin(["left", "right"]),
            "track_id",
        ]
    }
    persisted_side_tracks = {
        int(track_id)
        for track_id, team in assignments.items()
        if team in {"left", "right"}
    }
    unpersisted_graph_tracks = graph_side_tracks - persisted_side_tracks
    if neutral_fail_closed and unpersisted_graph_tracks:
        raise RuntimeError(
            "Neutral graph teams contain tracks absent from the persisted "
            f"assignment map: {sorted(unpersisted_graph_tracks)}"
        )
    return {
        "mode": (
            "persisted_map_fail_closed"
            if neutral_fail_closed
            else "resolved_map_with_tracklab_fallback"
        ),
        "persisted_side_tracks": len(persisted_side_tracks),
        "graph_side_tracks": len(graph_side_tracks),
        "unmapped_athlete_tracks_excluded": (
            missing_track_ids if neutral_fail_closed else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ball-confidence", type=float, default=0.03)
    parser.add_argument(
        "--recluster-teams",
        action="store_true",
        help="Replace outfield team labels with track-level shirt-colour clusters",
    )
    parser.add_argument("--left-team-name", default=None)
    parser.add_argument("--right-team-name", default=None)
    parser.add_argument("--left-team-direction", type=int, choices=(-1, 1), default=None)
    parser.add_argument("--right-team-direction", type=int, choices=(-1, 1), default=None)
    parser.add_argument(
        "--direction-provenance-json",
        default=None,
        help=(
            "Optional JSON metadata for explicit directions; direct manual "
            "directions default to calibrated confidence 1.0"
        ),
    )
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument(
        "--match-registry",
        type=Path,
        default=None,
        help="JSON registry containing a match-specific TrackLab cluster mapping",
    )
    parser.add_argument(
        "--match-id",
        default=None,
        help="Exact match key in --match-registry",
    )
    parser.add_argument(
        "--cluster-0-team-name",
        default=None,
        help="Reviewed club name for TrackLab team_cluster 0",
    )
    parser.add_argument(
        "--cluster-1-team-name",
        default=None,
        help="Reviewed club name for TrackLab team_cluster 1",
    )
    parser.add_argument(
        "--neutral-team-names",
        action="store_true",
        help="Map clusters to Team A/Team B without claiming club identity",
    )
    parser.add_argument(
        "--team-labels",
        type=Path,
        default=None,
        help="Optional sequence-scoped manual labels; generic matches default off",
    )
    parser.add_argument(
        "--team-model",
        type=Path,
        default=None,
        help="Optional match-specific identity model; generic matches default off",
    )
    parser.add_argument("--disable-team-labels", action="store_true")
    parser.add_argument("--disable-team-model", action="store_true")
    parser.add_argument("--team-model-threshold", type=float, default=None)
    args = parser.parse_args()

    if bool(args.match_registry) != bool(args.match_id):
        parser.error("--match-registry and --match-id must be supplied together")
    if bool(args.cluster_0_team_name) != bool(args.cluster_1_team_name):
        parser.error(
            "--cluster-0-team-name and --cluster-1-team-name must be supplied together"
        )
    if args.match_registry and args.cluster_0_team_name:
        parser.error(
            "Use either --match-registry or explicit cluster team names, not both"
        )
    if args.neutral_team_names and (
        args.match_registry or args.cluster_0_team_name
    ):
        parser.error(
            "--neutral-team-names conflicts with registry or explicit club names"
        )
    if args.disable_team_labels and args.team_labels is not None:
        parser.error("--disable-team-labels conflicts with --team-labels")
    if args.disable_team_model and args.team_model is not None:
        parser.error("--disable-team-model conflicts with --team-model")
    if (args.left_team_direction is None) != (
        args.right_team_direction is None
    ):
        parser.error(
            "--left-team-direction and --right-team-direction must be supplied "
            "together"
        )
    if (
        args.left_team_direction is not None
        and args.left_team_direction == args.right_team_direction
    ):
        parser.error("Team attack directions must be opposite")
    if (
        args.direction_provenance_json is not None
        and args.left_team_direction is None
    ):
        parser.error(
            "--direction-provenance-json requires explicit team directions"
        )
    direction_provenance = None
    if args.direction_provenance_json is not None:
        try:
            direction_provenance = json.loads(
                args.direction_provenance_json
            )
        except json.JSONDecodeError as error:
            parser.error(f"Invalid --direction-provenance-json: {error}")
        if not isinstance(direction_provenance, dict):
            parser.error("--direction-provenance-json must contain an object")

    generic_identity = bool(
        args.match_registry
        or args.cluster_0_team_name
        or args.neutral_team_names
    )
    if generic_identity and not args.recluster_teams:
        parser.error(
            "Registry, cluster-name, and neutral mappings require --recluster-teams"
        )
    if generic_identity and (args.left_team_name or args.right_team_name):
        parser.error(
            "Use --cluster-0-team-name/--cluster-1-team-name for generic matches; "
            "--left-team-name/--right-team-name are the legacy Burnley-Arsenal path"
        )

    if args.match_registry:
        team_config = load_match_team_config(
            args.match_registry,
            args.match_id,
            sequence_id=args.sequence_id or args.video.stem,
        )
    elif args.neutral_team_names:
        team_config = neutral_match_team_config(
            match_id=args.match_id,
            source="explicit_neutral_cli",
        )
    else:
        team_config = None
    if args.cluster_0_team_name:
        team_config = explicit_match_team_config(
            args.cluster_0_team_name,
            args.cluster_1_team_name,
            match_id=args.match_id,
        )

    effective_team_labels = None if args.disable_team_labels else args.team_labels
    effective_team_model = None if args.disable_team_model else args.team_model
    if not generic_identity:
        if not args.disable_team_labels and effective_team_labels is None:
            effective_team_labels = LEGACY_TEAM_LABELS
        if not args.disable_team_model and effective_team_model is None:
            effective_team_model = LEGACY_TEAM_MODEL

    with zipfile.ZipFile(args.state) as archive:
        detections = pickle.loads(archive.read("0.pkl"))
        images = pickle.loads(archive.read("0_image.pkl"))
    cluster_evidence = {}
    if args.recluster_teams:
        mapping_evidence = None
        tracklab_side_to_internal = None
        if team_config is not None:
            club_to_internal = team_config["club_to_internal"]
            cluster_to_internal = team_config["cluster_to_internal"]
            (
                raw_cluster_to_side,
                mapping_evidence,
            ) = infer_cluster_to_internal_from_tracklab_team(detections)
            if cluster_to_internal is None:
                cluster_to_internal = raw_cluster_to_side
            tracklab_side_to_internal = {
                raw_cluster_to_side[cluster]: internal
                for cluster, internal in cluster_to_internal.items()
            }
        else:
            club_to_internal = {
                name.lower(): internal
                for name, internal in (
                    (args.left_team_name, "left"),
                    (args.right_team_name, "right"),
                )
                if name
            }
            cluster_to_internal = None
        inferred, cluster_evidence = recluster_player_teams(
            args.video,
            detections,
            sequence_id=args.sequence_id,
            labels_path=effective_team_labels,
            model_path=effective_team_model,
            model_threshold=args.team_model_threshold,
            club_to_internal=club_to_internal,
            cluster_to_internal=cluster_to_internal,
            tracklab_side_to_internal=tracklab_side_to_internal,
        )
        if mapping_evidence is not None:
            cluster_evidence["semantic_cluster_alignment"] = mapping_evidence
        neutral_unreviewed = bool(
            team_config is not None
            and team_config["mapping_status"] == "unreviewed"
        )
        if neutral_unreviewed:
            # Persist the exact neutral per-track assignments used below so
            # review overlays cannot silently fall back to the raw, pre-
            # correction KMeans cluster.  The expansion pipeline explicitly
            # disables both identity inputs, which keeps these labels anonymous.
            cluster_evidence["neutral_corrected_track_assignments"] = {
                str(track_id): team
                for track_id, team in sorted(inferred.items())
                if team in {"left", "right"}
            }
            cluster_evidence["neutral_identity_inputs"] = {
                "manual_labels_enabled": effective_team_labels is not None,
                "identity_model_enabled": effective_team_model is not None,
            }
        assignment_application = apply_resolved_team_assignments(
            detections,
            inferred,
            neutral_fail_closed=neutral_unreviewed,
        )
        if neutral_unreviewed:
            cluster_evidence[
                "neutral_assignment_application"
            ] = assignment_application
        print(f"Assigned {len(inferred)} athlete tracks to teams")
    ball_by_frame = detect_balls(
        args.video, args.yolo, len(images), args.ball_confidence
    )
    params_by_frame = {
        int(row.frame): row.parameters for _, row in images.iterrows()
    }
    by_frame = {
        int(frame): rows for frame, rows in detections.groupby("image_id")
    }

    if args.left_team_direction is not None and args.right_team_direction is not None:
        direction = {
            "left": args.left_team_direction,
            "right": args.right_team_direction,
        }
        try:
            explicit_metadata = explicit_direction_metadata(
                direction,
                direction_provenance,
            )
        except ValueError as error:
            parser.error(str(error))
        direction_source = explicit_metadata["source"]
        direction_status = explicit_metadata["status"]
        direction_confident = explicit_metadata["confident"]
        direction_confidence = explicit_metadata["confidence"]
        direction_evidence = explicit_metadata["evidence"]
    else:
        direction_evidence = calibrate_detections(detections)
        direction_status = str(direction_evidence["status"])
        direction_confident = bool(direction_evidence["confident"])
        direction_confidence = float(direction_evidence["confidence"])
        calibrated = direction_evidence.get("directions")
        if calibrated is not None:
            direction = {
                team: int(calibrated[team]) for team in ("left", "right")
            }
            direction_source = "pitch_spatial_calibration"
        elif (
            team_config is not None
            and team_config["mapping_status"] == "unreviewed"
        ):
            # In the neutral path the internal labels intentionally retain
            # TrackLab's own-goal-side semantics. Keep a deterministic
            # canonical rotation for non-directional modelling, but expose the
            # abstention so no attacker-relative structured label (high press,
            # central screen, or either touchline trap) can be presented as
            # trusted evidence.
            direction = {"left": 1, "right": -1}
            direction_source = "tracklab_spatial_side_default_abstained"
        else:
            raise RuntimeError(
                "Could not confidently calibrate both team directions; "
                f"evidence={direction_evidence}"
            )
    team_names = (
        team_config["team_names"]
        if team_config is not None
        else {
            "left": args.left_team_name or "left",
            "right": args.right_team_name or "right",
        }
    )
    team_identity_status = (
        team_config["mapping_status"] if team_config is not None else "legacy"
    )
    team_identity_source = (
        team_config["source"]
        if team_config is not None
        else "legacy_burnley_arsenal_kit_rule"
    )

    graph_rows, masks, metadata = [], [], []
    previous: dict[int, tuple[int, np.ndarray]] = {}
    for frame, ball_detection in sorted(ball_by_frame.items()):
        parameters = params_by_frame.get(frame)
        if not parameters:
            continue
        ball = unproject(parameters, tuple(ball_detection["image_xy"]))
        rows = by_frame.get(frame)
        if ball is None or rows is None:
            continue
        players = []
        for _, row in rows.iterrows():
            if row.team not in ("left", "right") or not isinstance(row.bbox_pitch, dict):
                continue
            point = np.array(
                [
                    row.bbox_pitch["x_bottom_middle"],
                    row.bbox_pitch["y_bottom_middle"],
                ],
                dtype=np.float32,
            )
            if np.isfinite(point).all() and abs(point[0]) <= 54.5 and abs(point[1]) <= 36:
                players.append((row, point))
        team_counts = {
            team: sum(row.team == team for row, _ in players)
            for team in ("left", "right")
        }
        if len(players) < 6 or min(team_counts.values()) < 2:
            continue
        distances = np.array([np.linalg.norm(point - ball) for _, point in players])
        holder = int(distances.argmin())
        holder_distance = float(distances[holder])
        possession_team = players[holder][0].team
        holder_track_id = int(players[holder][0].track_id)
        rotation = direction[possession_team]
        players.sort(key=lambda item: (item[0].team != possession_team, item[0].track_id))
        features = np.zeros((23, 13), dtype=np.float32)
        mask = np.zeros(23, dtype=bool)
        for index, (row, point) in enumerate(players[:22]):
            rotated = point * rotation
            features[index, :2] = [
                (rotated[0] + PITCH_LENGTH / 2) / PITCH_LENGTH,
                (rotated[1] + PITCH_WIDTH / 2) / PITCH_WIDTH,
            ]
            track_id = int(row.track_id)
            if track_id in previous and frame > previous[track_id][0]:
                velocity = (
                    (point - previous[track_id][1]) / ((frame - previous[track_id][0]) / 25)
                ) * rotation
                features[index, 2:4] = np.clip(
                    velocity / [PITCH_LENGTH, PITCH_WIDTH], -0.2, 0.2
                )
            previous[track_id] = (frame, point)
            features[index, 4 if row.team == possession_team else 5] = 1
            features[index, 7 if row.role == "goalkeeper" else 8] = 1
            if track_id == holder_track_id and holder_distance <= 3:
                features[index, 12] = 1
            mask[index] = True
        ball_index = int(mask.sum())
        rotated_ball = ball * rotation
        features[ball_index, :2] = [
            (rotated_ball[0] + PITCH_LENGTH / 2) / PITCH_LENGTH,
            (rotated_ball[1] + PITCH_WIDTH / 2) / PITCH_WIDTH,
        ]
        features[ball_index, 6] = 1
        features[ball_index, 11] = 1
        mask[ball_index] = True
        graph_rows.append(features)
        masks.append(mask)
        metadata.append(
            {
                "sequence": args.sequence_id or args.video.stem,
                "frame": frame,
                "time_seconds": frame / 25,
                "possession_team": possession_team,
                "possession_club": team_names[possession_team],
                "pressing_club": team_names[
                    "right" if possession_team == "left" else "left"
                ],
                "team_identity_map": team_names,
                "team_identity_status": team_identity_status,
                "team_identity_source": team_identity_source,
                "team_identity_match_id": (
                    team_config["match_id"] if team_config is not None else None
                ),
                "attacking_direction_raw": rotation,
                "attacking_direction_label": (
                    "left_to_right" if rotation == 1 else "right_to_left"
                ),
                "direction_source": direction_source,
                "direction_status": direction_status,
                "direction_confident": direction_confident,
                "direction_confidence": direction_confidence,
                "direction_evidence": direction_evidence,
                "team_cluster_evidence": cluster_evidence,
                "ball_holder_distance_m": round(holder_distance, 3),
                "possession_confident": holder_distance <= 3,
                "ball_detection_confidence": round(ball_detection["confidence"], 5),
                "visible_nodes": int(mask.sum()),
                "source": "tracklab_gsr+yolo_coco_ball",
            }
        )
    if not graph_rows:
        raise RuntimeError("No classifiable frames with calibrated players and a detected ball")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=np.stack(graph_rows), masks=np.stack(masks)
    )
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata)
    )
    print(f"Wrote {len(graph_rows)} reconstructed graphs to {args.output}")


if __name__ == "__main__":
    main()
