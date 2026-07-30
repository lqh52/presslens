#!/usr/bin/env python3
"""Build a compact human-review set for club identity on demo player tracks.

Run this script with the SoccerNet Game State Reconstruction environment because
the TrackLab state pickle depends on that environment's pandas/TrackLab stack.
The resulting crops and manifest stay under the git-ignored ``data/review`` tree.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from convert_tracklab_state import recluster_player_teams
from team_identity import (
    explicit_match_team_config,
    infer_cluster_to_internal_from_tracklab_team,
    load_match_team_config,
    neutral_match_team_config,
)


CLUB_BY_INTERNAL_TEAM = {"left": "arsenal", "right": "burnley"}


def clip_half(video_id: str, clip: dict) -> int:
    if clip.get("half") is not None:
        half = int(clip["half"])
    else:
        match = re.search(r"(?:^|[-_])h([12])(?:[-_]|$)", video_id, re.IGNORECASE)
        if match is None:
            match = re.match(r"h([12])", video_id, re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"Cannot infer half for {video_id}; add half: 1 or 2 to its clip"
            )
        half = int(match.group(1))
    if half not in {1, 2}:
        raise ValueError(f"Invalid half for {video_id}: {half}")
    return half


def latest_state(root: Path, video_id: str) -> Path:
    pattern = f"multi-{video_id}/**/states/multi-{video_id}.pklz"
    paths = [path for path in root.glob(pattern) if path.stat().st_size]
    if not paths:
        raise FileNotFoundError(f"No non-empty TrackLab state matching {pattern}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_detections(state_path: Path):
    with zipfile.ZipFile(state_path) as archive:
        return pickle.loads(archive.read("0.pkl"))


def quality_rows(rows):
    mask = []
    for _, row in rows.iterrows():
        _, _, width, height = map(float, row.bbox_ltwh)
        mask.append(
            float(row.get("bbox_conf", 1.0)) >= 0.45
            and width >= 12
            and height >= 40
        )
    return rows[np.asarray(mask, dtype=bool)]


def sample_rows(rows, count: int) -> list:
    """Choose high-quality detections while preserving temporal diversity."""
    candidates = []
    for index, row in rows.iterrows():
        left, top, width, height = map(float, row.bbox_ltwh)
        confidence = float(row.get("bbox_conf", 1.0))
        area = width * height
        candidates.append(
            (int(row.image_id), confidence * np.sqrt(area), index, row)
        )
    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0])
    chosen = []
    frame_values = np.asarray([item[0] for item in candidates])
    targets = np.linspace(frame_values.min(), frame_values.max(), count)
    for target in targets:
        unused = [item for item in candidates if item[2] not in {x[2] for x in chosen}]
        if not unused:
            break
        scale = max(1.0, float(frame_values.max() - frame_values.min()))
        pick = max(
            unused,
            key=lambda item: item[1] / max(value[1] for value in unused)
            - 0.55 * abs(item[0] - target) / scale,
        )
        chosen.append(pick)

    chosen.sort(key=lambda item: item[0])
    return [item[3] for item in chosen]


def read_frames(video_path: Path, frame_numbers: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    images: dict[int, np.ndarray] = {}
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    for frame_number in sorted(frame_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if ok:
            images[frame_number] = image
    capture.release()
    return images


def player_tile(image: np.ndarray, bbox: np.ndarray, frame: int) -> np.ndarray:
    left, top, width, height = map(float, bbox)
    image_height, image_width = image.shape[:2]
    x1 = max(0, int(left - 0.28 * width))
    x2 = min(image_width, int(left + 1.28 * width))
    y1 = max(0, int(top - 0.12 * height))
    y2 = min(image_height, int(top + 1.08 * height))
    crop = image[y1:y2, x1:x2]
    canvas = np.full((300, 220, 3), 24, dtype=np.uint8)
    if crop.size:
        scale = min(210 / crop.shape[1], 266 / crop.shape[0])
        resized = cv2.resize(
            crop,
            (
                max(1, int(crop.shape[1] * scale)),
                max(1, int(crop.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        x = (canvas.shape[1] - resized.shape[1]) // 2
        y = 6 + (266 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 274), (219, 299), (13, 17, 15), -1)
    cv2.putText(
        canvas,
        f"frame {frame}",
        (9, 292),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 228, 222),
        1,
        cv2.LINE_AA,
    )
    return canvas


def contact_sheet(
    sample_rows_for_track: list,
    images: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, list[int]]:
    tiles = []
    frames = []
    for row in sample_rows_for_track:
        frame = int(row.image_id)
        image = images.get(frame)
        if image is None:
            continue
        tiles.append(player_tile(image, row.bbox_ltwh, frame))
        frames.append(frame)
    if not tiles:
        return None, []
    while len(tiles) < 3:
        tiles.append(np.full_like(tiles[0], 24))
    return np.concatenate(tiles[:3], axis=1), frames


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-manifest",
        type=Path,
        default=Path("public/demo/manifest.json"),
    )
    parser.add_argument(
        "--video-directory",
        type=Path,
        default=Path("data/raw/gsr-multi-input"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("third_party/sn-gamestate/outputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/review/team_tracks"),
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=50,
        help="Review tracks present this many frames around the demo evidence frame",
    )
    parser.add_argument("--samples-per-track", type=int, default=3)
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=6,
        help="Maximum review tracks sampled from each learned cluster per video",
    )
    parser.add_argument(
        "--match-registry",
        type=Path,
        default=None,
        help="Team identity registry; clips may provide matchId/match_id",
    )
    parser.add_argument(
        "--match-id",
        default=None,
        help="Registry match key shared by every clip unless a clip overrides it",
    )
    parser.add_argument("--cluster-0-team-name", default=None)
    parser.add_argument("--cluster-1-team-name", default=None)
    parser.add_argument("--neutral-team-names", action="store_true")
    parser.add_argument(
        "--team-labels",
        type=Path,
        default=None,
        help="Optional sequence-scoped labels; generic matches default off",
    )
    parser.add_argument(
        "--team-model",
        type=Path,
        default=None,
        help="Optional match-specific model; generic matches default off",
    )
    parser.add_argument("--disable-team-labels", action="store_true")
    parser.add_argument("--disable-team-model", action="store_true")
    parser.add_argument("--team-model-threshold", type=float, default=None)
    args = parser.parse_args()
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
    if args.match_id and not args.match_registry:
        parser.error("--match-id requires --match-registry")

    demo = json.loads(args.demo_manifest.read_text())
    clips_by_video = {item["videoId"]: item for item in demo["clips"]}
    crops_path = args.output / "crops"
    crops_path.mkdir(parents=True, exist_ok=True)
    items = []
    match_configs: dict[tuple[str, str], dict] = {}

    for video_id, clip in sorted(clips_by_video.items()):
        centre = int(clip["frame"])
        start, end = max(0, centre - args.radius), centre + args.radius
        video_path = args.video_directory / f"{video_id}.mp4"
        state_path = latest_state(args.state_root, video_id)
        detections = load_detections(state_path)
        clip_match_id = (
            clip.get("matchId")
            or clip.get("match_id")
            or args.match_id
        )
        if args.match_registry:
            if not clip_match_id:
                raise ValueError(
                    f"{video_id} has no matchId/match_id and --match-id was not set"
                )
            config_key = (clip_match_id, video_id)
            if config_key not in match_configs:
                match_configs[config_key] = load_match_team_config(
                    args.match_registry,
                    clip_match_id,
                    sequence_id=video_id,
                )
            config = match_configs[config_key]
        elif args.cluster_0_team_name:
            config = explicit_match_team_config(
                args.cluster_0_team_name,
                args.cluster_1_team_name,
                match_id=clip_match_id,
            )
        elif args.neutral_team_names:
            config = neutral_match_team_config(
                match_id=clip_match_id,
                source="explicit_neutral_cli",
            )
        else:
            config = None
        if config is not None:
            match_configs.setdefault(
                (clip_match_id or "__all_clips__", video_id),
                config,
            )

        generic_identity = config is not None
        labels_path = (
            None
            if args.disable_team_labels
            else args.team_labels
            if generic_identity
            else args.team_labels or Path("data/review/team_tracks/labels.json")
        )
        model_path = (
            None
            if args.disable_team_model
            else args.team_model
            if generic_identity
            else args.team_model
            or Path("models/team_identity_burnley_arsenal.npz")
        )
        cluster_to_internal = (
            config["cluster_to_internal"] if config is not None else None
        )
        tracklab_side_to_internal = None
        if config is not None:
            (
                raw_cluster_to_side,
                _,
            ) = infer_cluster_to_internal_from_tracklab_team(detections)
            if cluster_to_internal is None:
                cluster_to_internal = raw_cluster_to_side
            tracklab_side_to_internal = {
                raw_cluster_to_side[cluster]: internal
                for cluster, internal in cluster_to_internal.items()
            }
        assignments, evidence = recluster_player_teams(
            video_path,
            detections,
            sequence_id=video_id,
            labels_path=labels_path,
            model_path=model_path,
            model_threshold=args.team_model_threshold,
            club_to_internal=(
                config["club_to_internal"] if config is not None else None
            ),
            cluster_to_internal=cluster_to_internal,
            tracklab_side_to_internal=tracklab_side_to_internal,
        )
        athletes = detections[
            (detections.role == "player")
            & detections.image_id.between(start, end)
        ]
        rows_by_track = {}
        selected_by_track = {}
        wanted_frames: set[int] = set()
        candidates_by_cluster: dict[int, list[tuple[tuple[int, float], int, object]]] = {
            0: [],
            1: [],
        }
        for track_id, rows in athletes.groupby("track_id"):
            quality = quality_rows(rows)
            if len(quality) < 5:
                continue
            integer_track_id = int(track_id)
            cluster_values = quality.team_cluster.dropna()
            if not len(cluster_values):
                continue
            cluster = int(cluster_values.mode().iloc[0])
            areas = [
                float(row.bbox_ltwh[2] * row.bbox_ltwh[3])
                for _, row in quality.iterrows()
            ]
            candidates_by_cluster.setdefault(cluster, []).append(
                ((len(quality), float(np.median(areas))), integer_track_id, quality)
            )

        for cluster_candidates in candidates_by_cluster.values():
            cluster_candidates.sort(key=lambda item: item[0], reverse=True)
            for _, track_id, quality in cluster_candidates[: args.max_per_cluster]:
                selected = sample_rows(quality, args.samples_per_track)
                if not selected:
                    continue
                rows_by_track[track_id] = quality
                selected_by_track[track_id] = selected
                wanted_frames.update(int(row.image_id) for row in selected)

        images = read_frames(video_path, wanted_frames)
        for track_id, selected in sorted(selected_by_track.items()):
            sheet, sample_frames = contact_sheet(selected, images)
            if sheet is None:
                continue
            rows = rows_by_track[track_id]
            filename = f"{video_id}-track-{track_id}.jpg"
            cv2.imwrite(
                str(crops_path / filename),
                sheet,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            prediction = (
                config["team_names"].get(assignments.get(track_id))
                if config is not None
                else CLUB_BY_INTERNAL_TEAM.get(assignments.get(track_id))
            )
            cluster_values = rows.team_cluster.dropna()
            cluster = (
                int(cluster_values.mode().iloc[0]) if len(cluster_values) else None
            )
            role_values = rows.role.dropna()
            role = str(role_values.mode().iloc[0]) if len(role_values) else "player"
            items.append(
                {
                    "key": f"{video_id}:{track_id}",
                    "video_id": video_id,
                    "match_id": clip_match_id,
                    "half": clip_half(video_id, clip),
                    "track_id": track_id,
                    "role": role,
                    "crop": filename,
                    "sample_frames": sample_frames,
                    "detections_in_window": int(len(rows)),
                    "current_prediction": prediction,
                    "mapping_status": (
                        config["mapping_status"] if config is not None else "legacy"
                    ),
                    "team_cluster": cluster,
                    "prototype_separation_lab": evidence.get(
                        "prototype_separation_lab"
                    ),
                }
            )
        print(f"{video_id}: {len(selected_by_track)} reviewable athlete tracks")

    identity_configs = {}
    for (match_id, _sequence_id), config in match_configs.items():
        identity_configs.setdefault(
            match_id,
            {
                "mapping_status": config["mapping_status"],
                "team_names": config["team_names"],
                "source": config["source"],
            },
        )
    if len(identity_configs) > 1:
        match_label = "Multiple SoccerNet matches"
    elif identity_configs:
        match_label = " / ".join(
            next(iter(identity_configs.values()))["team_names"].values()
        )
    else:
        match_label = "Burnley vs Arsenal"
    if identity_configs:
        label_options = sorted(
            {
                name.lower()
                for config in identity_configs.values()
                for name in config["team_names"].values()
            }
        ) + ["ignore"]
    else:
        label_options = ["arsenal", "burnley", "ignore"]
    manifest = {
        "match": match_label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            f"Player tracks within ±{args.radius} frames of each visible demo excerpt"
        ),
        "labels": label_options,
        "team_identity": identity_configs,
        "items": items,
    }
    write_json_atomic(args.output / "manifest.json", manifest)
    print(f"Wrote {len(items)} review items to {args.output}")


if __name__ == "__main__":
    main()
