#!/usr/bin/env python3
"""Infer SkillCorner pressing labels on reviewed projected-video graphs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from scripts.train_skillcorner_pressing_stg import (
        SpatiotemporalGraphNet,
        temporal_windows,
    )
except ModuleNotFoundError:
    from train_skillcorner_pressing_stg import (
        SpatiotemporalGraphNet,
        temporal_windows,
    )


MEDIA_NAMES = {
    "ars-che-20160924-h2-0058": (
        "video-ars-che-20160924-h2-0058-annotated.mp4",
        "canonical-ars-che-20160924-h2-0058.mp4",
    ),
    "ars-che-20160924-h2-0067": (
        "video-ars-che-20160924-h2-0067-annotated.mp4",
        "canonical-ars-che-20160924-h2-0067.mp4",
    ),
    "bur-ars-20150411-h1-0128": (
        "video-h1-128-annotated.mp4",
        "canonical-h1-128.mp4",
    ),
    "bur-ars-20150411-h1-0203": (
        "video-h1-203-annotated.mp4",
        "canonical-h1-203.mp4",
    ),
    "bur-ars-20150411-h1-0833": (
        "video-h1-833-annotated.mp4",
        "canonical-h1-833.mp4",
    ),
    "bur-ars-20150411-h1-1673": (
        "video-h1-1673-annotated.mp4",
        "canonical-h1-1673.mp4",
    ),
    "hul-ars-20160917-h1-0009": (
        "video-hul-ars-20160917-h1-0009-annotated.mp4",
        "canonical-hul-ars-20160917-h1-0009.mp4",
    ),
    "lei-ars-20150926-h1-0093-published": (
        "video-lei-ars-20150926-h1-0093-annotated.mp4",
        "canonical-lei-ars-20150926-h1-0093.mp4",
    ),
}


def read_video_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        return np.full((680, 1050, 3), (13, 27, 20), np.uint8)
    return image


def preview(
    clip_id: str,
    frame: int,
    prediction: dict,
    media_dir: Path,
) -> np.ndarray:
    broadcast_name, canonical_name = MEDIA_NAMES[clip_id]
    broadcast = read_video_frame(media_dir / broadcast_name, frame)
    canonical = read_video_frame(media_dir / canonical_name, frame)
    broadcast = cv2.resize(broadcast, (525, 340))
    canonical = cv2.resize(canonical, (525, 340))
    body = np.hstack([broadcast, canonical])
    header = np.full((82, 1050, 3), (13, 27, 20), np.uint8)
    label = prediction["label"].replace("_", " ").upper()
    cv2.putText(
        header,
        f"{label}  {prediction['confidence']:.0%}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (106, 240, 255),
        2,
        cv2.LINE_AA,
    )
    detail = (
        f"{clip_id} | frame {frame} | "
        + "  ".join(
            f"{name.replace('_', ' ')} {value:.0%}"
            for name, value in prediction["probabilities"].items()
        )
    )
    cv2.putText(
        header,
        detail,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 238, 229),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Write predictions without rendering clip-specific preview media.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    model = SpatiotemporalGraphNet(
        checkpoint["feature_dim"],
        len(labels),
        width=checkpoint["width"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    payload = np.load(args.graphs)
    rows = [
        json.loads(line)
        for line in args.metadata.read_text().splitlines()
        if line.strip()
    ]
    by_clip: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_clip[row["clip_id"]].append(index)

    output_clips = []
    preview_images = []
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in sorted(by_clip):
        predictions = []
        for window in temporal_windows(rows, by_clip[clip_id]):
            features = torch.from_numpy(
                payload["features"][window][None].astype(np.float32)
            )
            masks = torch.from_numpy(payload["masks"][window][None])
            with torch.inference_mode():
                probabilities = model(features, masks).softmax(1)[0].numpy()
            winner = int(probabilities.argmax())
            center = window[len(window) // 2]
            predictions.append(
                {
                    "frame": int(rows[center]["frame"]),
                    "source_frames": [int(rows[index]["frame"]) for index in window],
                    "label": labels[winner],
                    "confidence": round(float(probabilities[winner]), 4),
                    "probabilities": {
                        label: round(float(probabilities[index]), 4)
                        for index, label in enumerate(labels)
                    },
                    "visible_nodes": [
                        int(rows[index]["visible_nodes"]) for index in window
                    ],
                }
            )
        if not predictions:
            continue
        mean_probabilities = {
            label: float(
                np.mean(
                    [
                        prediction["probabilities"][label]
                        for prediction in predictions
                    ]
                )
            )
            for label in labels
        }
        dominant = max(mean_probabilities, key=mean_probabilities.get)
        representative = max(
            predictions,
            key=lambda prediction: prediction["probabilities"][dominant],
        )
        clip_result = {
            "clip_id": clip_id,
            "summary": {
                "label": dominant,
                "confidence": round(mean_probabilities[dominant], 4),
                "probabilities": {
                    label: round(value, 4)
                    for label, value in mean_probabilities.items()
                },
                "windows": len(predictions),
            },
            "windows": predictions,
        }
        output_clips.append(clip_result)
        if not args.skip_previews:
            image = preview(
                clip_id,
                representative["frame"],
                representative,
                args.media_dir,
            )
            image_path = args.preview_dir / f"{clip_id}.png"
            cv2.imwrite(str(image_path), image)
            preview_images.append(image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "method": "spatiotemporal_graph_only",
        "vision_features": False,
        "model": str(args.model),
        "labels": labels,
        "clips": output_clips,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if preview_images:
        cells = [cv2.resize(image, (630, 253)) for image in preview_images]
        if len(cells) % 2:
            cells.append(np.full_like(cells[0], (13, 27, 20)))
        contact = np.vstack(
            [np.hstack(cells[index : index + 2]) for index in range(0, len(cells), 2)]
        )
        contact_path = args.preview_dir / "contact-sheet.png"
        cv2.imwrite(str(contact_path), contact)
        print(f"Wrote {len(output_clips)} clips to {args.output} and {contact_path}")
    else:
        print(f"Wrote {len(output_clips)} clips to {args.output}")


if __name__ == "__main__":
    main()
