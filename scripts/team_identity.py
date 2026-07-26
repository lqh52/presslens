"""Match-scoped utilities for player-track team and club identity.

The fallback team assignment remains outside this module. This layer provides
neutral/reviewed match mappings, human overrides, and a conservative linear
classifier over TrackLab ReID embeddings.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


INTERNAL_TEAMS = ("left", "right")
NEUTRAL_TEAM_NAMES = {"left": "Team A", "right": "Team B"}


def _canonical_club(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def neutral_match_team_config(
    *,
    match_id: str | None = None,
    source: str = "neutral_default",
) -> dict:
    """Return deterministic cluster identities without claiming club identity.

    ``left`` and ``right`` are the graph pipeline's two internal team slots.
    For an unreviewed match they retain TrackLab's per-clip semantic meaning:
    the team defending the raw pitch's left/right goal. TrackLab's numeric
    clusters are produced by a fresh KMeans fit for every clip, so they must
    never be assigned globally here.
    """
    return {
        "match_id": match_id,
        "mapping_status": "unreviewed",
        "source": source,
        "cluster_to_internal": None,
        "cluster_mapping_scope": "tracklab_team_per_sequence",
        "team_names": dict(NEUTRAL_TEAM_NAMES),
        "club_to_internal": {},
    }


def explicit_match_team_config(
    cluster_0_name: str,
    cluster_1_name: str,
    *,
    match_id: str | None = None,
    source: str = "explicit_cli",
) -> dict:
    """Create a reviewed team mapping from explicit TrackLab cluster names."""
    names = {
        "left": str(cluster_0_name).strip(),
        "right": str(cluster_1_name).strip(),
    }
    if not all(names.values()):
        raise ValueError("Both cluster team names must be non-empty")
    canonical = {_canonical_club(name): internal for internal, name in names.items()}
    if len(canonical) != 2:
        raise ValueError("Cluster 0 and cluster 1 must map to different clubs")
    return {
        "match_id": match_id,
        "mapping_status": "reviewed",
        "source": source,
        "cluster_to_internal": {0: "left", 1: "right"},
        "team_names": names,
        "club_to_internal": canonical,
    }


def _cluster_mapping(
    raw: object,
    *,
    context: str,
) -> dict[int, str]:
    try:
        mapping = {
            int(cluster): str(internal)
            for cluster, internal in raw.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{context} must map clusters 0/1 to left/right"
        ) from error
    if (
        set(mapping) != {0, 1}
        or set(mapping.values()) != set(INTERNAL_TEAMS)
    ):
        raise ValueError(
            f"{context} must be a bijection between clusters 0/1 and "
            "internal teams left/right"
        )
    return mapping


def infer_cluster_to_internal_from_tracklab_team(detections) -> tuple[dict[int, str], dict]:
    """Recover one clip's KMeans-to-spatial-team mapping.

    SoccerNet Game State Reconstruction fits team KMeans independently for
    every video. Its subsequent side-labeling module records the spatial
    own-goal side as ``detections.team``. This function robustly recovers that
    clip-local mapping and refuses mixed or missing clusters.
    """

    required = {"role", "team_cluster", "team"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(
            "TrackLab detections are missing columns required for per-sequence "
            f"team alignment: {sorted(missing)}"
        )
    players = detections[detections.role == "player"]
    votes: dict[int, dict[str, int]] = {}
    mapping: dict[int, str] = {}
    purities: dict[int, float] = {}
    for cluster in (0, 1):
        rows = players[players.team_cluster == cluster]
        counts = {
            team: int((rows.team == team).sum())
            for team in INTERNAL_TEAMS
        }
        total = sum(counts.values())
        votes[cluster] = counts
        if total == 0:
            raise ValueError(
                f"TrackLab cluster {cluster} has no left/right team-side votes"
            )
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if ranked[0][1] == ranked[1][1]:
            raise ValueError(
                f"TrackLab cluster {cluster} has tied team-side votes: {counts}"
            )
        mapping[cluster] = ranked[0][0]
        purities[cluster] = ranked[0][1] / total
    if set(mapping.values()) != set(INTERNAL_TEAMS):
        raise ValueError(
            "TrackLab clusters do not map bijectively to left/right for this "
            f"sequence: {mapping}; votes={votes}"
        )
    minimum_purity = min(purities.values())
    if minimum_purity < 0.8:
        raise ValueError(
            "TrackLab cluster-to-side mapping is ambiguous for this sequence: "
            f"purities={purities}; votes={votes}"
        )
    return mapping, {
        "source": "tracklab_team_mode_per_sequence",
        "cluster_to_internal": dict(mapping),
        "team_votes": {
            str(cluster): counts for cluster, counts in votes.items()
        },
        "purity": {
            str(cluster): round(value, 6)
            for cluster, value in purities.items()
        },
    }


def load_match_team_config(
    path: Path,
    match_id: str,
    *,
    sequence_id: str | None = None,
) -> dict:
    """Load one match's reviewed or neutral TrackLab cluster mapping.

    Unreviewed registry entries deliberately resolve to Team A/Team B even
    though the participating clubs are known. This prevents match metadata
    from being mistaken for a reviewed cluster-to-club assignment.
    """
    payload = json.loads(path.read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("matches"), dict):
        raise ValueError(f"Unsupported team identity registry: {path}")
    try:
        entry = payload["matches"][match_id]
    except KeyError as error:
        raise KeyError(f"Match {match_id!r} is not present in {path}") from error
    if not isinstance(entry, dict):
        raise ValueError(f"Registry entry for {match_id!r} must be an object")

    mapping_status = entry.get("mapping_status", "unreviewed")
    if mapping_status not in {"unreviewed", "reviewed"}:
        raise ValueError(
            f"Invalid mapping_status for {match_id!r}: {mapping_status!r}"
        )
    result = neutral_match_team_config(
        match_id=match_id,
        source=f"registry:{path}",
    )
    result["mapping_status"] = mapping_status
    if mapping_status == "unreviewed":
        return result

    sequence_mappings = entry.get("sequence_cluster_to_internal")
    if not isinstance(sequence_mappings, dict):
        raise ValueError(
            f"Reviewed registry entry {match_id!r} requires "
            "sequence_cluster_to_internal; TrackLab cluster IDs are clip-local "
            "and a global cluster_to_internal mapping is unsafe"
        )
    if not sequence_id:
        raise ValueError(
            f"Reviewed registry entry {match_id!r} requires sequence_id to "
            "select a clip-scoped cluster mapping"
        )
    try:
        sequence_mapping_raw = sequence_mappings[sequence_id]
    except KeyError as error:
        raise KeyError(
            f"Reviewed registry entry {match_id!r} has no clip-scoped mapping "
            f"for sequence {sequence_id!r}"
        ) from error
    cluster_to_internal = _cluster_mapping(
        sequence_mapping_raw,
        context=(
            f"sequence_cluster_to_internal[{sequence_id!r}] for {match_id!r}"
        ),
    )

    cluster_to_club_raw = entry.get("cluster_to_club")
    if not isinstance(cluster_to_club_raw, dict):
        raise ValueError(
            f"Reviewed registry entry {match_id!r} requires cluster_to_club"
        )
    try:
        cluster_to_club = {
            int(cluster): str(club).strip()
            for cluster, club in cluster_to_club_raw.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"cluster_to_club for {match_id!r} must map clusters 0/1 to clubs"
        ) from error
    if set(cluster_to_club) != {0, 1} or not all(cluster_to_club.values()):
        raise ValueError(
            f"cluster_to_club for {match_id!r} must name both clusters 0 and 1"
        )
    reference_mapping = _cluster_mapping(
        entry.get("cluster_to_internal"),
        context=f"cluster_to_internal for {match_id!r}",
    )
    team_names = {
        reference_mapping[cluster]: club
        for cluster, club in cluster_to_club.items()
    }
    club_to_internal = {
        _canonical_club(club): reference_mapping[cluster]
        for cluster, club in cluster_to_club.items()
    }
    if len(club_to_internal) != 2:
        raise ValueError(f"Reviewed registry entry {match_id!r} repeats a club")
    result.update(
        cluster_to_internal=cluster_to_internal,
        cluster_mapping_scope=f"sequence:{sequence_id}",
        team_names=team_names,
        club_to_internal=club_to_internal,
    )
    return result


def aggregate_track_embedding(rows, expected_dim: int | None = None) -> np.ndarray | None:
    """L2-normalize detections, robustly aggregate a track, then normalize again."""
    embeddings = []
    for value in rows.embeddings:
        try:
            embedding = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if (
            not len(embedding)
            or (expected_dim is not None and len(embedding) != expected_dim)
            or not np.isfinite(embedding).all()
        ):
            continue
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            continue
        embeddings.append(embedding / norm)
    if not embeddings:
        return None
    result = np.median(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(result))
    return (result / norm).astype(np.float32) if norm > 1e-8 else None


def read_human_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    records = payload.get("labels", payload)
    return {
        key: _canonical_club(value.get("label", value))
        for key, value in records.items()
    }


def load_linear_model(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        model = {key: artifact[key] for key in artifact.files}
    required = {"coef", "intercept", "mean", "scale", "classes", "threshold"}
    missing = required - model.keys()
    if missing:
        raise ValueError(f"Team identity artifact is missing {sorted(missing)}")
    model["classes"] = [str(value) for value in model["classes"].tolist()]
    model["threshold"] = float(np.asarray(model["threshold"]).reshape(-1)[0])
    return model


def predict_club(model: dict, embedding: np.ndarray) -> tuple[str, float]:
    mean = np.asarray(model["mean"], dtype=np.float32).reshape(-1)
    scale = np.asarray(model["scale"], dtype=np.float32).reshape(-1)
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError(
            f"Embedding/model dimensions disagree: {vector.shape}, "
            f"{mean.shape}, {scale.shape}"
        )
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    standardized = (vector - mean) / scale
    coef = np.asarray(model["coef"], dtype=np.float32)
    intercept = np.asarray(model["intercept"], dtype=np.float32).reshape(-1)
    classes = model["classes"]
    logits = coef @ standardized + intercept
    if len(classes) == 2 and len(logits) == 1:
        positive = float(1.0 / (1.0 + np.exp(-np.clip(logits[0], -30, 30))))
        probabilities = np.array([1.0 - positive, positive], dtype=np.float32)
    elif len(logits) == len(classes):
        shifted = logits - logits.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
    else:
        raise ValueError("Unsupported class/linear-head shape in team model")
    index = int(probabilities.argmax())
    return classes[index], float(probabilities[index])


def infer_nonparticipant_track_ids(
    detections,
    *,
    video_path: Path | None = None,
    model_path: Path | None,
    threshold: float | None = None,
    minimum_detections: int = 3,
    exclude_goalkeepers: bool = True,
) -> tuple[set[int], dict]:
    """Conservatively exclude tracks that should not appear in overlays.

    TrackLab's role head already removes explicit referees. This second gate
    handles referees, substitutes, fans, and goalkeepers that entered the
    outfield-player clusters. It uses the existing expert-trained ``ignore``
    class only as a rejection signal; club predictions are deliberately
    discarded so the function remains safe for unseen fixtures.
    """

    if minimum_detections <= 0:
        raise ValueError("minimum_detections must be positive")
    model = load_linear_model(model_path)
    cutoff = (
        float(threshold)
        if threshold is not None
        else float(model["threshold"]) if model is not None else None
    )
    if cutoff is not None and not 0 <= cutoff <= 1:
        raise ValueError("nonparticipant threshold must be between 0 and 1")
    if model is not None and "ignore" not in {
        _canonical_club(value) for value in model["classes"]
    }:
        raise ValueError("nonparticipant model has no ignore class")

    excluded: set[int] = set()
    reasons: dict[str, list[int]] = {
        "goalkeeper_role": [],
        "role_detection_nonplayer": [],
        "too_few_detections": [],
        "outside_pitch": [],
        "sideline_nonparticipant": [],
        "appearance_outlier": [],
        "expert_ignore_model": [],
    }
    model_scores: dict[str, float] = {}
    athletes = detections[detections.role.isin(["player", "goalkeeper"])]
    appearance_evidence: dict[str, object] = {}
    appearance_outliers: set[int] = set()
    video_height: float | None = None
    if video_path is not None:
        import cv2

        players = athletes[athletes.role == "player"]
        wanted_frames = set(map(int, players.image_id.unique()))
        samples: dict[int, list[np.ndarray]] = defaultdict(list)
        capture = cv2.VideoCapture(str(video_path))
        raw_video_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if raw_video_height > 0:
            video_height = raw_video_height
        frame = 0
        while frame <= max(wanted_frames, default=-1):
            ok, image = capture.read()
            if not ok:
                break
            if frame in wanted_frames:
                height, width = image.shape[:2]
                for _, row in players[players.image_id == frame].iterrows():
                    left, top, box_width, box_height = map(
                        float, row.bbox_ltwh
                    )
                    x1 = max(0, int(left + 0.2 * box_width))
                    x2 = min(width, int(left + 0.8 * box_width))
                    y1 = max(0, int(top + 0.12 * box_height))
                    y2 = min(height, int(top + 0.52 * box_height))
                    crop = image[y1:y2, x1:x2]
                    if crop.size < 30:
                        continue
                    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                    samples[int(row.track_id)].append(
                        np.median(
                            lab.reshape(-1, 3), axis=0
                        ).astype(np.float32)
                    )
            frame += 1
        capture.release()
        track_colours = {
            track_id: np.median(values, axis=0)
            for track_id, values in samples.items()
            if len(values) >= 2
        }
        track_clusters: dict[int, int] = {}
        for raw_track_id, rows in players.groupby("track_id"):
            clusters = rows.team_cluster.dropna()
            if len(clusters):
                track_clusters[int(raw_track_id)] = int(
                    clusters.mode().iloc[0]
                )
        prototypes = {}
        for cluster in (0, 1):
            values = [
                track_colours[track_id]
                for track_id, assigned in track_clusters.items()
                if assigned == cluster and track_id in track_colours
            ]
            if len(values) >= 3:
                prototypes[cluster] = np.median(
                    np.asarray(values), axis=0
                )
        thresholds = {}
        distances_by_track: dict[str, dict[str, float | int]] = {}
        if len(prototypes) == 2:
            own_distances_by_cluster: dict[int, list[float]] = {
                0: [],
                1: [],
            }
            for track_id, cluster in track_clusters.items():
                if track_id not in track_colours:
                    continue
                own_distances_by_cluster[cluster].append(
                    float(
                        np.linalg.norm(
                            track_colours[track_id] - prototypes[cluster]
                        )
                    )
                )
            for cluster, values in own_distances_by_cluster.items():
                median = float(np.median(values))
                mad = float(np.median(np.abs(np.asarray(values) - median)))
                thresholds[cluster] = max(
                    18.0,
                    median + 4.0 * max(1.0, 1.4826 * mad),
                )
            for track_id, cluster in track_clusters.items():
                if track_id not in track_colours:
                    continue
                distances = {
                    candidate: float(
                        np.linalg.norm(
                            track_colours[track_id] - prototype
                        )
                    )
                    for candidate, prototype in prototypes.items()
                }
                own_distance = distances[cluster]
                nearest_distance = min(distances.values())
                distances_by_track[str(track_id)] = {
                    "cluster": cluster,
                    "own_distance": round(own_distance, 3),
                    "nearest_distance": round(nearest_distance, 3),
                }
                if (
                    own_distance > thresholds[cluster]
                    and nearest_distance > 15.0
                ):
                    appearance_outliers.add(track_id)
        appearance_evidence = {
            "method": "full_lab_torso_track_outlier",
            "tracks_with_colour": len(track_colours),
            "prototypes_lab": {
                str(cluster): value.round(3).tolist()
                for cluster, value in prototypes.items()
            },
            "thresholds": {
                str(cluster): round(value, 3)
                for cluster, value in thresholds.items()
            },
            "distances": distances_by_track,
        }
    expected_dim = len(np.asarray(model["mean"]).reshape(-1)) if model else None
    for raw_track_id, rows in athletes.groupby("track_id"):
        track_id = int(raw_track_id)
        roles = rows.role.dropna()
        role = str(roles.mode().iloc[0]) if len(roles) else "player"
        if exclude_goalkeepers and role == "goalkeeper":
            excluded.add(track_id)
            reasons["goalkeeper_role"].append(track_id)
            continue
        if track_id in appearance_outliers:
            excluded.add(track_id)
            reasons["appearance_outlier"].append(track_id)
            continue
        if "role_detection" in rows:
            detected_roles = rows.role_detection.dropna().astype(str)
            if len(detected_roles):
                referee_count = int((detected_roles == "referee").sum())
                goalkeeper_fraction = float(
                    (detected_roles == "goalkeeper").mean()
                )
                if (
                    referee_count >= 2
                    and referee_count / len(detected_roles) >= 0.1
                ) or (
                    exclude_goalkeepers and goalkeeper_fraction >= 0.5
                ):
                    excluded.add(track_id)
                    reasons["role_detection_nonplayer"].append(track_id)
                    continue
        if len(rows) < minimum_detections:
            excluded.add(track_id)
            reasons["too_few_detections"].append(track_id)
            continue

        pitch_points = []
        for value in rows.bbox_pitch if "bbox_pitch" in rows else []:
            if not isinstance(value, dict):
                continue
            try:
                point = np.asarray(
                    [value["x_bottom_middle"], value["y_bottom_middle"]],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(point).all():
                pitch_points.append(point)
        if not pitch_points:
            excluded.add(track_id)
            reasons["outside_pitch"].append(track_id)
            continue
        points = np.asarray(pitch_points)
        if video_height is not None and "bbox_ltwh" in rows:
            bottoms = []
            for box in rows.bbox_ltwh:
                try:
                    bottoms.append(float(box[1]) + float(box[3]))
                except (TypeError, ValueError, IndexError):
                    continue
            if (
                bottoms
                and float(np.median(bottoms)) >= 0.94 * video_height
                and float(np.median(np.abs(points[:, 1]))) >= 27.0
            ):
                excluded.add(track_id)
                reasons["sideline_nonparticipant"].append(track_id)
                continue
        outside = (
            (np.abs(points[:, 0]) > 52.5)
            | (np.abs(points[:, 1]) > 34.0)
        )
        if float(outside.mean()) >= 0.5:
            excluded.add(track_id)
            reasons["outside_pitch"].append(track_id)
            continue

        if model is not None and role == "player":
            embedding = aggregate_track_embedding(rows, expected_dim)
            if embedding is not None:
                predicted, score = predict_club(model, embedding)
                if _canonical_club(predicted) == "ignore":
                    model_scores[str(track_id)] = round(float(score), 6)
                    if score >= float(cutoff):
                        excluded.add(track_id)
                        reasons["expert_ignore_model"].append(track_id)

    return excluded, {
        "mode": "role_pitch_duration_and_expert_ignore",
        "model_path": str(model_path) if model is not None else None,
        "model_threshold": cutoff,
        "exclude_goalkeepers": exclude_goalkeepers,
        "minimum_detections": minimum_detections,
        "excluded_track_ids": sorted(excluded),
        "reasons": {
            key: sorted(values)
            for key, values in reasons.items()
            if values
        },
        "ignore_model_scores": model_scores,
        "appearance": appearance_evidence,
    }


def resolve_team_assignments(
    base_assignments: dict[int, str],
    detections,
    *,
    sequence_id: str,
    club_to_internal: dict[str, str],
    labels_path: Path | None,
    model_path: Path | None,
    threshold: float | None = None,
    strict_model_classes: bool = False,
) -> tuple[dict[int, str], dict]:
    """Apply manual labels first, then confident model results, then fallback.

    ``ignore`` is an explicit rejection class for tracks that should not enter
    the tactical graph (for example referees, fans, substitutes, or a
    goalkeeper incorrectly assigned the outfield-player role).
    """
    assignments = dict(base_assignments)
    labels = read_human_labels(labels_path)
    model = load_linear_model(model_path)
    if model is not None and strict_model_classes:
        allowed = set(club_to_internal) | {"ignore"}
        unknown = {
            _canonical_club(value)
            for value in model["classes"]
            if _canonical_club(value) not in allowed
        }
        if unknown:
            raise ValueError(
                "Team identity model contains clubs outside this match: "
                f"{sorted(unknown)}"
            )
        if not club_to_internal:
            raise ValueError(
                "A team identity model requires a reviewed cluster-to-club mapping"
            )
    cutoff = (
        float(threshold)
        if threshold is not None
        else float(model["threshold"]) if model is not None else None
    )
    evidence = {
        "sequence_id": sequence_id,
        "manual_tracks": 0,
        "manual_excluded_tracks": 0,
        "model_tracks": 0,
        "model_excluded_tracks": 0,
        "model_abstained_tracks": 0,
        "fallback_tracks": 0,
        "model_threshold": cutoff,
        "model_path": str(model_path) if model is not None else None,
        "precedence": (
            "manual club/exclusion > model above threshold > appearance fallback"
        ),
    }
    athletes = detections[detections.role.isin(["player", "goalkeeper"])]
    expected_dim = len(np.asarray(model["mean"]).reshape(-1)) if model else None
    for track_id, rows in athletes.groupby("track_id"):
        track_id = int(track_id)
        key = f"{sequence_id}:{track_id}"
        human = labels.get(key)
        if human in club_to_internal:
            assignments[track_id] = club_to_internal[human]
            evidence["manual_tracks"] += 1
            continue
        if human == "ignore":
            assignments[track_id] = "ignore"
            evidence["manual_excluded_tracks"] += 1
            continue

        roles = rows.role.dropna()
        role = str(roles.mode().iloc[0]) if len(roles) else "player"
        if model is not None and role == "player":
            embedding = aggregate_track_embedding(rows, expected_dim)
            if embedding is not None:
                club, score = predict_club(model, embedding)
                club_key = _canonical_club(club)
                if club_key == "ignore" and score >= float(cutoff):
                    assignments[track_id] = "ignore"
                    evidence["model_excluded_tracks"] += 1
                    continue
                if club_key in club_to_internal and score >= float(cutoff):
                    assignments[track_id] = club_to_internal[club_key]
                    evidence["model_tracks"] += 1
                    continue
                evidence["model_abstained_tracks"] += 1
        if track_id in assignments:
            evidence["fallback_tracks"] += 1
    evidence["tracks_assigned"] = len(assignments)
    return assignments, evidence
