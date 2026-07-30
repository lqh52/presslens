#!/usr/bin/env python3
"""Build the existing PressLens product catalogue from curated pressing clips."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.build_reviewed_web_demo import atomic_json, graph_nodes
    from scripts.render_presslens_media import render_clip
except ModuleNotFoundError:
    from build_reviewed_web_demo import atomic_json, graph_nodes
    from render_presslens_media import render_clip


COPY = {
    "high_press_wing": (
        "High press — wing",
        "The pressing team engages high and channels pressure toward a flank.",
        ["high press", "wing pressure", "wide pressing lane"],
    ),
    "high_press_central": (
        "High press — central",
        "The pressing team engages high through the central build-up lanes.",
        ["high press", "central pressure", "build-up disruption"],
    ),
    "medium_press": (
        "Medium press",
        "The defending team engages around the middle third in a coordinated shape.",
        ["medium block", "middle-third pressure", "compact shape"],
    ),
    "low_block": (
        "Low block",
        "The defending team stays compact near its own penalty area with most of its shape behind the ball.",
        ["low block", "deep compact shape", "defensive-third pressure"],
    ),
}


def source_config(root: Path, source: str) -> tuple[Path, Path, Path]:
    if source == "published-tracking-review":
        return (
            root / "artifacts/published-tracking-review",
            root / "artifacts/skillcorner-pressing-inference/reviewed-8-4class.json",
            root / "artifacts/published-tracking-review/real-video-graphs.npz",
        )
    if source == "tactical-coverage-review":
        return (
            root / "artifacts/tactical-coverage-review",
            root / "artifacts/tactical-coverage-review/pressing-review-predictions.json",
            root / "artifacts/tactical-coverage-review/pressing-review-graphs.npz",
        )
    raise ValueError(f"Unknown catalogue source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/annotations/product_pressing_catalog.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("public/demo"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    catalogue = json.loads(args.catalog.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)

    loaded: dict[str, tuple[Path, dict, np.lib.npyio.NpzFile, list[dict]]] = {}
    clips, videos, retained_files = [], [], {"manifest.json", "search-index.json"}
    for selected in catalogue["clips"]:
        source = selected["source"]
        if source not in loaded:
            base, prediction_path, graph_path = source_config(root, source)
            predictions = {
                row["clip_id"]: row
                for row in json.loads(prediction_path.read_text())["clips"]
            }
            metadata_path = graph_path.with_suffix(".jsonl")
            metadata = [
                json.loads(line)
                for line in metadata_path.read_text().splitlines()
                if line.strip()
            ]
            loaded[source] = (base, predictions, np.load(graph_path), metadata)
        base, predictions, graph, metadata = loaded[source]
        clip_id, label = selected["clip_id"], selected["label"]
        prediction = predictions[clip_id]
        representative = max(
            prediction["windows"],
            key=lambda row: row["probabilities"].get(label, 0.0),
        )
        frame = int(representative["frame"])
        candidates = [
            (index, row)
            for index, row in enumerate(metadata)
            if row["clip_id"] == clip_id
        ]
        graph_index, graph_row = min(
            candidates, key=lambda item: abs(int(item[1]["frame"]) - frame)
        )
        public_id = selected["id"].replace("::", "--")
        broadcast_name = f"video-{public_id}-v1.mp4"
        canonical_name = f"canonical-{public_id}-v1.mp4"
        reviewed_labels_path = base / "track-labels.json"
        reviewed_labels = (
            json.loads(reviewed_labels_path.read_text()).get("labels", {})
            if reviewed_labels_path.exists()
            else {}
        )
        render_clip(
            base,
            args.output,
            clip_id,
            reviewed_labels,
            (broadcast_name, canonical_name),
        )
        capture = cv2.VideoCapture(str(args.output / broadcast_name))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, poster = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        duration_seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        capture.release()
        if not ok:
            raise RuntimeError(f"{public_id}: cannot extract representative frame")
        poster_name = f"frames/{public_id}-frame-{frame:04d}.jpg"
        cv2.imwrite(str(args.output / poster_name), poster)
        canonical_image = f"canonical-{public_id}.png"
        canonical_capture = cv2.VideoCapture(str(args.output / canonical_name))
        canonical_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, canonical = canonical_capture.read()
        canonical_capture.release()
        if not ok:
            raise RuntimeError(f"{public_id}: cannot extract canonical frame")
        cv2.imwrite(str(args.output / canonical_image), canonical)

        title, description, tags = COPY[label]
        players, ball = graph_nodes(
            graph["features"][graph_index],
            graph["masks"][graph_index],
        )
        probability = representative["probabilities"][label]
        clip = {
            "id": public_id,
            "videoId": public_id,
            "video": f"/demo/{broadcast_name}",
            "match": clip_id,
            "canonicalImage": f"/demo/{canonical_image}",
            "canonicalVideo": f"/demo/{canonical_name}",
            "minute": f"{frame / fps / 60:.1f} min",
            "half": 2 if "-h2-" in clip_id else 1,
            "timeSeconds": round(frame / fps, 3),
            "durationSeconds": round(duration_seconds, 1),
            "frame": frame,
            "situation": label,
            "title": title,
            "confidence": round(float(probability) * 100, 1),
            "majorityFrames": int(prediction["summary"]["windows"]),
            "validFrames": int(prediction["summary"]["windows"]),
            "orientationValidated": bool(graph_row.get("direction_confident", False)),
            "reviewDecision": "include",
            "labelSource": "expert_review",
            "phase": selected["quality"],
            "visibleNodes": int(graph_row["visible_nodes"]),
            "possessionConfident": bool(graph_row.get("possession_confident", False)),
            "ballConfidence": 100.0 if graph_row.get("possession_confident") else 0.0,
            "possessionClub": "Possession team",
            "pressingClub": "Defending team",
            "attackDirection": (
                "left_to_right"
                if int(graph_row.get("attacking_direction_raw", 1)) == 1
                else "right_to_left"
            ),
            "directionSource": "canonical graph calibration",
            "directionConfidence": round(
                float(graph_row.get("direction_confidence", 0.0)) * 100, 1
            ),
            "teamIdentityMap": {
                "possession": "Possession team",
                "pressing": "Defending team",
            },
            "ballHolderDistanceM": graph_row.get("ball_holder_distance_m"),
            "description": description,
            "evidence": [
                f"Model confidence · {float(probability) * 100:.1f}%",
                f"Visible graph nodes · {graph_row['visible_nodes']}",
            ],
            "tags": tags + [selected["quality"], "reviewed", "real video"],
            "probabilities": representative["probabilities"],
            "weakLabel": label,
            "weakRule": "human-reviewed graph classification",
            "thumbnail": f"/demo/{poster_name}",
            "players": players,
            "ball": ball,
            "overlayTrackFilter": None,
        }
        clips.append(clip)
        videos.append(
            {
                "id": public_id,
                "half": clip["half"],
                "startSeconds": 0.0,
                "path": clip["video"],
            }
        )
        retained_files.update(
            {broadcast_name, canonical_name, canonical_image, poster_name}
        )
        print(f"{public_id}: {label} ({selected['quality']})")

    manifest = {
        "name": "Tactical retrieval.",
        "source": "Human-reviewed reconstructed match video",
        "count": len(clips),
        "videoCount": len(videos),
        "matchCount": len({row["match"] for row in clips}),
        "reviewStatus": "expert_curated_product",
        "videos": videos,
        "clips": clips,
    }
    atomic_json(args.output / "manifest.json", manifest)
    for path in sorted(args.output.rglob("*"), reverse=True):
        relative = str(path.relative_to(args.output))
        if path.is_file() and relative not in retained_files:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f"Wrote {len(clips)} product clips to {args.output}")


if __name__ == "__main__":
    main()
