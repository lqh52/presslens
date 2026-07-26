"""Conservative raw-pitch attack-direction calibration.

The functions in this module operate on already reconstructed TrackLab
detections. They use pitch coordinates and goalkeeper/formation anchors only;
video-text retrieval, screen side, kit colour, and club names are deliberately
outside the calibration contract.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


TEAMS = ("left", "right")
PITCH_HALF_LENGTH = 52.5


def _value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _records(detections: Any) -> Iterable[Any]:
    if hasattr(detections, "itertuples"):
        return detections.itertuples(index=False)
    return detections


def _pitch_x(row: Any) -> float | None:
    pitch = _value(row, "bbox_pitch")
    if not isinstance(pitch, dict):
        return None
    try:
        value = float(pitch["x_bottom_middle"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) > PITCH_HALF_LENGTH + 2:
        return None
    return value


def _track_anchors(detections: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    samples: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    frames: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for index, row in enumerate(_records(detections)):
        team = str(_value(row, "team", ""))
        role = str(_value(row, "role", ""))
        if team not in TEAMS or role not in {"player", "goalkeeper"}:
            continue
        x = _pitch_x(row)
        if x is None:
            continue
        raw_track_id = _value(row, "track_id", index)
        try:
            track_id = int(raw_track_id)
        except (TypeError, ValueError):
            track_id = index
        key = (team, role, track_id)
        samples[key].append(x)
        raw_frame = _value(row, "image_id", index)
        try:
            frames[key].add(int(raw_frame))
        except (TypeError, ValueError):
            frames[key].add(index)

    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        team: {"player": [], "goalkeeper": []} for team in TEAMS
    }
    for (team, role, track_id), values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        result[team][role].append(
            {
                "track_id": track_id,
                "x": median,
                "mad_x": mad,
                "observations": len(values),
                "frames": len(frames[(team, role, track_id)]),
            }
        )
    return result


def _orientation_for_low_team(low_team: str) -> int:
    """Return the raw attack direction of internal team ``left``."""

    return 1 if low_team == "left" else -1


def _method(
    *,
    name: str,
    left_direction: int,
    confidence: float,
    base_weight: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return {
        "method": name,
        "left_direction": int(left_direction),
        "right_direction": int(-left_direction),
        "confidence": round(confidence, 6),
        "weight": round(base_weight * confidence, 6),
        **evidence,
    }


def calibrate_detections(
    detections: Any,
    *,
    minimum_player_tracks: int = 2,
    minimum_centroid_separation_m: float = 4.0,
    minimum_goalkeeper_frames: int = 3,
    minimum_goalkeeper_depth_m: float = 16.0,
    minimum_total_weight: float = 0.55,
    minimum_vote_confidence: float = 0.72,
) -> dict[str, Any]:
    """Calibrate the two internal teams' raw pitch directions for one clip.

    Each track contributes one robust median, preventing a long track from
    overwhelming shorter independent evidence. Goalkeeper anchors carry more
    weight than formation centroids. The two possible orientations vote
    against each other; disagreement is retained and may force abstention.
    """

    anchors = _track_anchors(detections)
    methods: list[dict[str, Any]] = []
    team_summary: dict[str, Any] = {}

    player_centroids: dict[str, float] = {}
    goalkeeper_centroids: dict[str, float] = {}
    goalkeeper_candidates = {
        team: [
            track
            for track in anchors[team]["goalkeeper"]
            if track["frames"] >= minimum_goalkeeper_frames
            and track["mad_x"] <= 12.0
            and abs(track["x"]) >= minimum_goalkeeper_depth_m
        ]
        for team in TEAMS
    }
    contested_goalkeeper_ids = (
        {
            track["track_id"]
            for track in goalkeeper_candidates["left"]
        }
        & {
            track["track_id"]
            for track in goalkeeper_candidates["right"]
        }
    )
    for team in TEAMS:
        player_tracks = anchors[team]["player"]
        goalkeeper_tracks = [
            track
            for track in goalkeeper_candidates[team]
            if track["track_id"] not in contested_goalkeeper_ids
        ]
        if player_tracks:
            player_centroids[team] = float(
                np.median([track["x"] for track in player_tracks])
            )
        if goalkeeper_tracks:
            goalkeeper_centroids[team] = float(
                np.median([track["x"] for track in goalkeeper_tracks])
            )
        team_summary[team] = {
            "player_tracks": len(player_tracks),
            "player_centroid_x": (
                round(player_centroids[team], 4)
                if team in player_centroids
                else None
            ),
            "goalkeeper_tracks": len(goalkeeper_tracks),
            "goalkeeper_centroid_x": (
                round(goalkeeper_centroids[team], 4)
                if team in goalkeeper_centroids
                else None
            ),
            "goalkeeper_track_evidence": goalkeeper_tracks,
            "contested_goalkeeper_track_ids": sorted(
                contested_goalkeeper_ids
            ),
        }

    if (
        len(player_centroids) == 2
        and min(
            len(anchors[team]["player"])
            for team in TEAMS
        )
        >= minimum_player_tracks
    ):
        separation = abs(
            player_centroids["left"] - player_centroids["right"]
        )
        if separation >= minimum_centroid_separation_m:
            low_team = min(player_centroids, key=player_centroids.get)
            separation_confidence = float(
                np.clip(
                    (separation - minimum_centroid_separation_m) / 16.0
                    + 0.45,
                    0.0,
                    0.9,
                )
            )
            support = min(
                len(anchors[team]["player"]) for team in TEAMS
            )
            support_factor = min(1.0, support / 5.0)
            methods.append(
                _method(
                    name="formation_track_median_order",
                    left_direction=_orientation_for_low_team(low_team),
                    confidence=separation_confidence * support_factor,
                    base_weight=1.0,
                    evidence={
                        "low_x_team": low_team,
                        "centroid_separation_m": round(separation, 4),
                    },
                )
            )

    if len(goalkeeper_centroids) == 2:
        separation = abs(
            goalkeeper_centroids["left"]
            - goalkeeper_centroids["right"]
        )
        low_team = min(goalkeeper_centroids, key=goalkeeper_centroids.get)
        low_x = goalkeeper_centroids[low_team]
        high_team = "right" if low_team == "left" else "left"
        high_x = goalkeeper_centroids[high_team]
        opposite_sides = low_x < 0 < high_x
        if (
            separation >= 2 * minimum_goalkeeper_depth_m
            and opposite_sides
            and min(abs(low_x), abs(high_x))
            >= minimum_goalkeeper_depth_m
        ):
            depth = min(abs(low_x), abs(high_x))
            confidence = float(
                np.clip(
                    0.62
                    + 0.18 * separation / (2 * PITCH_HALF_LENGTH)
                    + 0.20 * depth / PITCH_HALF_LENGTH,
                    0.0,
                    1.0,
                )
            )
            methods.append(
                _method(
                    name="paired_goalkeeper_anchors",
                    left_direction=_orientation_for_low_team(low_team),
                    confidence=confidence,
                    base_weight=3.0,
                    evidence={
                        "low_x_team": low_team,
                        "goalkeeper_separation_m": round(separation, 4),
                        "opposite_pitch_sides": True,
                    },
                )
            )
    elif len(goalkeeper_centroids) == 1:
        team, x = next(iter(goalkeeper_centroids.items()))
        if abs(x) >= minimum_goalkeeper_depth_m:
            team_direction = 1 if x < 0 else -1
            left_direction = (
                team_direction if team == "left" else -team_direction
            )
            confidence = float(
                np.clip(
                    0.55
                    + 0.35
                    * (abs(x) - minimum_goalkeeper_depth_m)
                    / (PITCH_HALF_LENGTH - minimum_goalkeeper_depth_m),
                    0.0,
                    0.9,
                )
            )
            methods.append(
                _method(
                    name="single_goalkeeper_anchor",
                    left_direction=left_direction,
                    confidence=confidence,
                    base_weight=1.5,
                    evidence={
                        "anchored_team": team,
                        "goalkeeper_x": round(x, 4),
                    },
                )
            )

    weights = {
        orientation: sum(
            float(method["weight"])
            for method in methods
            if method["left_direction"] == orientation
        )
        for orientation in (-1, 1)
    }
    total_weight = sum(weights.values())
    if total_weight < minimum_total_weight:
        return {
            "status": "abstained_insufficient_spatial_evidence",
            "confident": False,
            "confidence": 0.0,
            "directions": None,
            "orientation_weights": {
                str(key): round(value, 6)
                for key, value in weights.items()
            },
            "methods": methods,
            "teams": team_summary,
        }
    winner = max(weights, key=weights.get)
    agreement = weights[winner] / total_weight
    evidence_strength = min(1.0, total_weight)
    confidence = agreement * evidence_strength
    margin = abs(weights[1] - weights[-1]) / total_weight
    confident = confidence >= minimum_vote_confidence
    return {
        "status": (
            "calibrated"
            if confident
            else "abstained_conflicting_spatial_evidence"
        ),
        "confident": confident,
        "confidence": round(confidence, 6),
        "vote_agreement": round(agreement, 6),
        "evidence_strength": round(evidence_strength, 6),
        "orientation_margin": round(margin, 6),
        "directions": (
            {"left": int(winner), "right": int(-winner)}
            if confident
            else None
        ),
        "orientation_weights": {
            str(key): round(value, 6)
            for key, value in weights.items()
        },
        "methods": methods,
        "teams": team_summary,
    }


def aggregate_clip_calibrations(
    clips: list[dict[str, Any]],
    *,
    minimum_clips: int = 2,
    minimum_vote_confidence: float = 0.75,
) -> dict[str, Any]:
    """Aggregate independent clip calibrations without frame-count weighting."""

    usable = [
        clip
        for clip in clips
        if clip.get("confident")
        and isinstance(clip.get("directions"), dict)
        and clip["directions"].get("left") in (-1, 1)
    ]
    weights = {-1: 0.0, 1: 0.0}
    counts = {-1: 0, 1: 0}
    for clip in usable:
        orientation = int(clip["directions"]["left"])
        # Every clip gets at most one vote; confidence changes its weight only
        # modestly so one long or unusually clean excerpt cannot dominate.
        weight = 0.5 + 0.5 * float(clip.get("confidence", 0.0))
        weights[orientation] += weight
        counts[orientation] += 1
    total = sum(weights.values())
    if len(usable) < minimum_clips or total <= 1e-9:
        return {
            "status": "abstained_insufficient_clips",
            "confident": False,
            "confidence": 0.0,
            "directions": None,
            "clips_available": len(clips),
            "clips_usable": len(usable),
            "orientation_clip_counts": {
                str(key): value for key, value in counts.items()
            },
            "orientation_weights": {
                str(key): round(value, 6)
                for key, value in weights.items()
            },
        }
    winner = max(weights, key=weights.get)
    confidence = weights[winner] / total
    confident = (
        confidence >= minimum_vote_confidence
        and counts[winner] >= minimum_clips
    )
    return {
        "status": (
            "calibrated"
            if confident
            else "abstained_conflicting_clips"
        ),
        "confident": confident,
        "confidence": round(confidence, 6),
        "orientation_margin": round(
            abs(weights[1] - weights[-1]) / total, 6
        ),
        "directions": (
            {"left": int(winner), "right": int(-winner)}
            if confident
            else None
        ),
        "clips_available": len(clips),
        "clips_usable": len(usable),
        "orientation_clip_counts": {
            str(key): value for key, value in counts.items()
        },
        "orientation_weights": {
            str(key): round(value, 6)
            for key, value in weights.items()
        },
    }
