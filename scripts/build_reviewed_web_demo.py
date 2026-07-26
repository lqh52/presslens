#!/usr/bin/env python3
"""Export reviewed tactical clips into the local PressLens web app."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_COPY = {
    "high_press": (
        "High press",
        "Several defenders compress the ball area during the build-up",
        [
            "dense pressure",
            "defensive-third build-up",
            "players converging on ball",
        ],
    ),
    "central_screen": (
        "Central screen",
        "The defending shape occupies central progression lanes ahead of the ball",
        [
            "central corridor protected",
            "forward lanes screened",
            "compact central shape",
        ],
    ),
    "trap_left": (
        "Left touchline trap",
        "Pressure is concentrated around the left touchline with defenders positioned inside the ball",
        ["ball near left touchline", "inside route constrained", "local pressure"],
    ),
    "trap_right": (
        "Right touchline trap",
        "Pressure is concentrated around the right touchline with defenders positioned inside the ball",
        ["ball near right touchline", "inside route constrained", "local pressure"],
    ),
    "unstructured": (
        "No local pressure",
        "The reconstructed state shows limited coordinated pressure in the immediate ball area",
        ["nearest defender distant", "low local density", "press structure unclear"],
    ),
}


def canonical_frame(
    features: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> Image.Image:
    image = Image.new("RGB", (630, 408), "#214b38")
    draw = ImageDraw.Draw(image)
    line = (190, 214, 198)
    draw.rectangle((6, 6, 624, 402), outline=line, width=1)
    draw.line((315, 6, 315, 402), fill=line, width=1)
    draw.ellipse((260, 149, 370, 259), outline=line, width=1)
    nodes = []
    ball = None
    for node, visible in zip(features, mask):
        if not visible:
            continue
        point = (
            int(np.clip(node[0], 0, 1) * 618 + 6),
            int(np.clip(node[1], 0, 1) * 396 + 6),
        )
        if node[6] > 0.5:
            ball = point
        else:
            nodes.append(
                (
                    "build" if node[4] > 0.5 else "press",
                    np.asarray(node[:2]),
                    point,
                )
            )
    edges = set()
    for left, (team, xy, _) in enumerate(nodes):
        neighbours = sorted(
            (
                float(np.linalg.norm((xy - other_xy) * [105, 68])),
                right,
            )
            for right, (other_team, other_xy, _) in enumerate(nodes)
            if right != left and other_team == team
        )
        for distance, right in neighbours[:2]:
            if distance <= 22:
                edges.add(tuple(sorted((left, right))))
    for left, right in edges:
        colour = "#dcebe1" if nodes[left][0] == "build" else "#ff745f"
        draw.line(
            (nodes[left][2], nodes[right][2]),
            fill=colour,
            width=1,
        )
    for team, _, point in nodes:
        colour = "#e8f0eb" if team == "build" else "#ff624c"
        draw.ellipse(
            (
                point[0] - 5,
                point[1] - 5,
                point[0] + 5,
                point[1] + 5,
            ),
            fill=colour,
        )
    if ball:
        draw.ellipse(
            (ball[0] - 4, ball[1] - 4, ball[0] + 4, ball[1] + 4),
            fill="#d9f45d",
        )
    draw.rectangle((14, 14, 190, 39), fill="#0b1711")
    draw.text(
        (23, 21),
        label.replace("_", " ").upper(),
        fill="#f2f6f3",
        font=ImageFont.load_default(),
    )
    return image


def align_canonical_orientation(
    features: np.ndarray,
    frame_direction: int,
    locked_direction: int,
) -> np.ndarray:
    if frame_direction == locked_direction:
        return features
    aligned = features.copy()
    aligned[:, 0:2] = 1.0 - aligned[:, 0:2]
    aligned[:, 2:4] *= -1.0
    return aligned


def render_canonical_assets(
    features: np.ndarray,
    masks: np.ndarray,
    predictions: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    output: Path,
    representative_index: int,
) -> Path:
    locked_direction = int(
        predictions[representative_index].get(
            "attacking_direction_raw",
            1,
        )
    )

    def oriented(index: int) -> np.ndarray:
        frame_direction = int(
            predictions[index].get(
                "attacking_direction_raw",
                locked_direction,
            )
        )
        return align_canonical_orientation(
            features[index],
            frame_direction,
            locked_direction,
        )

    available = [
        index
        for index, row in enumerate(predictions)
        if start_frame <= int(row["frame"]) < end_frame
    ]
    if not available:
        available = [representative_index]
    representative = canonical_frame(
        oriented(representative_index),
        masks[representative_index],
        predictions[representative_index]["predicted_situation"],
    )
    representative.save(output.with_suffix(".png"))
    video_output = output.with_suffix(".mp4")
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
            "rgb24",
            "-s",
            "630x408",
            "-r",
            "25",
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
            str(video_output),
        ],
        stdin=subprocess.PIPE,
    )
    for source_frame in range(start_frame, end_frame):
        nearest = min(
            available,
            key=lambda index: abs(
                int(predictions[index]["frame"]) - source_frame
            ),
        )
        image = canonical_frame(
            oriented(nearest),
            masks[nearest],
            predictions[nearest]["predicted_situation"],
        )
        assert encoder.stdin is not None
        encoder.stdin.write(
            np.asarray(image, dtype=np.uint8).tobytes()
        )
    assert encoder.stdin is not None
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(
            f"ffmpeg failed while writing {video_output}"
        )
    return video_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"Empty JSONL artifact: {path}")
    return rows


def atomic_json(path: Path, payload: Any) -> None:
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


def remove_unreferenced_web_assets(
    output: Path,
    manifest: dict[str, Any],
) -> None:
    retained = {
        "manifest.json",
        "search-index.json",
    }
    for clip in manifest["clips"]:
        for field in (
            "video",
            "canonicalImage",
            "canonicalVideo",
            "thumbnail",
        ):
            retained.add(
                str(clip[field]).removeprefix("/demo/")
            )
    removed = 0
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file() and str(path.relative_to(output)) not in retained:
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f"Removed {removed} obsolete web assets from {output}")


def graph_nodes(
    features: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    players = []
    ball = None
    for node, visible in zip(features, mask):
        if not visible:
            continue
        point = {
            "x": round(float(np.clip(node[0], 0, 1) * 100), 3),
            "y": round(float(np.clip(node[1], 0, 1) * 68), 3),
            "dx": round(float(node[2] * 100), 3),
            "dy": round(float(node[3] * 68), 3),
        }
        if node[6] > 0.5:
            ball = {"x": point["x"], "y": point["y"]}
            continue
        point.update(
            team="build" if node[4] > 0.5 else "press",
            role="goalkeeper" if node[7] > 0.5 else "player",
            controlsBall=bool(node[12] > 0.5),
        )
        players.append(point)
    return players, ball


def append_legacy_accepted_clips(
    clips: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    manifest_path: Path,
    review_path: Path,
    output: Path,
) -> None:
    """Preserve the expert-approved Burnley set in every expanded rebuild."""
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    legacy = json.loads(manifest_path.read_text())
    removed_video_ids = {
        str(row["video_id"])
        for row in json.loads(review_path.read_text())
        if row.get("demo_action") == "remove"
    }
    assets_root = manifest_path.parent
    existing_ids = {str(clip["id"]) for clip in clips}
    accepted = [
        clip
        for clip in legacy["clips"]
        if clip.get("phase") == "expert_accepted"
        and str(clip["videoId"]) not in removed_video_ids
        and str(clip["id"]) not in existing_ids
    ]
    for clip in accepted:
        referenced = [
            clip["video"],
            clip["canonicalImage"],
            clip["canonicalVideo"],
            clip["thumbnail"],
        ]
        for public_path in referenced:
            relative = Path(str(public_path).removeprefix("/demo/"))
            source = assets_root / relative
            destination = output / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        preserved = dict(clip)
        half, start = str(clip["videoId"]).split("-", 1)
        public_id = f"bur-ars-20150411-{half}-{int(start):04d}"
        preserved["id"] = public_id
        preserved["videoId"] = public_id
        preserved["match"] = (
            "2015-04-11 - 19-30 Burnley 0 - 1 Arsenal"
        )
        preserved["reviewDecision"] = "include"
        preserved["labelSource"] = "expert_review"
        clips.append(preserved)

    accepted_video_ids = {str(clip["videoId"]) for clip in accepted}
    for video in legacy["videos"]:
        if str(video["id"]) not in accepted_video_ids:
            continue
        half, start = str(video["id"]).split("-", 1)
        preserved_video = dict(video)
        preserved_video["id"] = (
            f"bur-ars-20150411-{half}-{int(start):04d}"
        )
        videos.append(preserved_video)
    print(
        f"Preserved {len(accepted)} previously accepted Burnley clips "
        f"from {manifest_path}"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    review_manifest = json.loads(args.review_manifest.read_text())
    annotations = json.loads(args.annotations.read_text()).get(
        "annotations", {}
    )
    accepted = [
        item
        for item in review_manifest["items"]
        if annotations.get(item["id"], {}).get("decision") == "accept"
    ]
    if not accepted:
        raise RuntimeError("No accepted review clips were found")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    clips = []
    videos = []
    for item in accepted:
        clip_id = str(item["id"])
        source_video = (
            args.review_manifest.parent / str(item["video"])
        ).resolve()
        graph_path = Path(item["provenance"]["graph"])
        predictions_path = Path(item["provenance"]["predictions"])
        weak_path = graph_path.with_name(f"{graph_path.stem}-weak.jsonl")
        for required in (
            source_video,
            graph_path,
            predictions_path,
            weak_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

        public_video_name = f"video-{clip_id}-annotated.mp4"
        public_video = output / public_video_name
        shutil.copy2(source_video, public_video)
        predictions = read_jsonl(predictions_path)
        weak_rows = read_jsonl(weak_path)
        representative_frame = int(item["representative_frame"])
        representative_index = next(
            (
                index
                for index, row in enumerate(predictions)
                if int(row["frame"]) == representative_frame
            ),
            None,
        )
        if representative_index is None:
            raise RuntimeError(
                f"{clip_id}: representative frame is absent from predictions"
            )
        graph = np.load(graph_path)
        canonical_base = output / f"canonical-{clip_id}"
        canonical_video = render_canonical_assets(
            graph["features"],
            graph["masks"],
            predictions,
            int(item["excerpt_start_frame"]),
            int(item["excerpt_end_frame"]),
            canonical_base,
            representative_index,
        )
        canonical_image = canonical_base.with_suffix(".png")

        local_frame = representative_frame - int(
            item["excerpt_start_frame"]
        )
        capture = cv2.VideoCapture(str(public_video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ok, poster = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"{clip_id}: cannot extract web poster")
        poster_name = f"{clip_id}-frame-{representative_frame:04d}.jpg"
        cv2.imwrite(
            str(frames_dir / poster_name),
            poster,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

        prediction = predictions[representative_index]
        weak = weak_rows[representative_index]
        situation = str(item["model_label"])
        title, description, tags = LABEL_COPY[situation]
        players, ball = graph_nodes(
            graph["features"][representative_index],
            graph["masks"][representative_index],
        )
        confidence = round(
            float(item["classification_confidence"]) * 100, 1
        )
        evidence = [
            (
                f"Frame agreement · {item['majority_frames']} / "
                f"{item['valid_graph_frames']} reliable frames agree"
            ),
            (
                "Direction confidence "
                f"{float(item['direction_confidence']) * 100:.1f}%"
            ),
            (
                f"{weak.get('descriptors', {}).get('press_within_12m', 0)} "
                "defenders within 12 m"
            ),
            f"Geometric rule: {weak.get('weak_label', 'abstain').replace('_', ' ')}",
        ]
        clips.append(
            {
                "id": clip_id,
                "videoId": clip_id,
                "video": f"/demo/{public_video_name}",
                "match": item["match"],
                "canonicalImage": f"/demo/{canonical_image.name}",
                "canonicalVideo": f"/demo/{canonical_video.name}",
                "minute": item["match_clock"],
                "half": int(item["half"]),
                "timeSeconds": round(local_frame / args.fps, 3),
                "frame": representative_frame,
                "situation": situation,
                "title": title,
                "confidence": confidence,
                "majorityFrames": int(item["majority_frames"]),
                "validFrames": int(item["valid_graph_frames"]),
                "orientationValidated": bool(item["direction_usable"]),
                "reviewDecision": "include",
                "labelSource": "expert_review",
                "phase": "expert_accepted",
                "visibleNodes": int(prediction["visible_nodes"]),
                "possessionConfident": bool(
                    prediction["possession_confident"]
                ),
                "ballConfidence": round(
                    float(prediction["ball_detection_confidence"]) * 100,
                    1,
                ),
                "possessionClub": prediction.get(
                    "possession_club", "Team A"
                ),
                "pressingClub": prediction.get("pressing_club", "Team B"),
                "attackDirection": prediction.get(
                    "attacking_direction_label", "undetermined"
                ),
                "directionSource": prediction.get(
                    "direction_source", "undetermined"
                ),
                "directionConfidence": round(
                    float(item["direction_confidence"]) * 100, 1
                ),
                "teamIdentityMap": prediction.get("team_identity_map", {}),
                "ballHolderDistanceM": prediction.get(
                    "ball_holder_distance_m"
                ),
                "description": description,
                "evidence": evidence,
                "tags": tags + ["reviewed", "real video"],
                "probabilities": prediction["probabilities"],
                "weakLabel": weak.get("weak_label", "abstain"),
                "weakRule": weak.get("weak_rule", "unavailable"),
                "thumbnail": f"/demo/frames/{poster_name}",
                "players": players,
                "ball": ball,
                "overlayTrackFilter": item.get("overlay_track_filter"),
            }
        )
        videos.append(
            {
                "id": clip_id,
                "half": int(item["half"]),
                "startSeconds": float(item["source_start_seconds"]),
                "path": f"/demo/{public_video_name}",
            }
        )
        print(f"{clip_id}: exported reviewed {situation}")

    append_legacy_accepted_clips(
        clips,
        videos,
        args.legacy_manifest,
        args.legacy_review,
        output,
    )
    manifest = {
        "name": "Tactical retrieval.",
        "source": "Local SoccerNet research video",
        "count": len(clips),
        "videoCount": len(videos),
        "matchCount": len({clip["match"] for clip in clips}),
        "reviewStatus": "expert_accepted",
        "videos": videos,
        "clips": clips,
    }
    atomic_json(output / "manifest.json", manifest)
    remove_unreferenced_web_assets(output, manifest)
    print(f"Wrote {len(clips)} accepted clips to {output}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=Path("data/review/expanded_tactical/manifest.json"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/review/expanded_tactical/annotations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/demo"),
    )
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        default=Path("data/review/burnley_approved_web/manifest.json"),
        help="Previously accepted Burnley clips that must survive rebuilds",
    )
    parser.add_argument(
        "--legacy-review",
        type=Path,
        default=Path("data/annotations/demo_video_review.json"),
        help="Review decisions controlling preserved Burnley clips",
    )
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    for required in (
        args.review_manifest,
        args.annotations,
        args.legacy_manifest,
        args.legacy_review,
    ):
        if not required.is_file():
            parser.error(f"Required review artifact is missing: {required}")
    return args


if __name__ == "__main__":
    build(parse_args())
