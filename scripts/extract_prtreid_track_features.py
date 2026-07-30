#!/usr/bin/env python3
"""Extract track-level SoccerNet PRTReID embeddings from tracking results.

Run this script with third_party/sn-gamestate/.venv/bin/python. It loads the
exact PRTReID checkpoint/configuration vendored with SoccerNet-GSR and writes
one compact JSON feature cache per tracking result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def select_samples(
    frames: list[dict[str, Any]], max_samples: int
) -> dict[int, list[tuple[int, list[float], float]]]:
    candidates: dict[int, list[tuple[int, list[float], float]]] = defaultdict(list)
    for frame in frames:
        frame_index = int(frame["frame"])
        for detection in frame["detections"]:
            if detection.get("track_id") is None:
                continue
            box = [float(value) for value in detection["bbox"]]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            score = float(detection.get("confidence", 0.0)) * np.sqrt(area)
            candidates[int(detection["track_id"])].append((frame_index, box, score))

    selected: dict[int, list[tuple[int, list[float], float]]] = {}
    for track_id, rows in candidates.items():
        # Prefer clear, large crops but retain temporal diversity.
        ranked = sorted(rows, key=lambda row: row[2], reverse=True)
        chosen: list[tuple[int, list[float], float]] = []
        minimum_gap = 3
        for row in ranked:
            if all(abs(row[0] - previous[0]) >= minimum_gap for previous in chosen):
                chosen.append(row)
            if len(chosen) == max_samples:
                break
        selected[track_id] = sorted(chosen, key=lambda row: row[0])
    return selected


def read_crops(
    video_path: Path,
    samples: dict[int, list[tuple[int, list[float], float]]],
) -> list[tuple[int, int, np.ndarray]]:
    by_frame: dict[int, list[tuple[int, int, list[float]]]] = defaultdict(list)
    for track_id, rows in samples.items():
        for sample_index, (frame_index, box, _) in enumerate(rows):
            by_frame[frame_index].append((track_id, sample_index, box))

    capture = cv2.VideoCapture(str(video_path))
    crops: list[tuple[int, int, np.ndarray]] = []
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        for track_id, sample_index, box in by_frame.get(frame_index, []):
            left = max(0, min(width - 1, int(np.floor(box[0]))))
            top = max(0, min(height - 1, int(np.floor(box[1]))))
            right = max(left + 1, min(width, int(np.ceil(box[2]))))
            bottom = max(top + 1, min(height, int(np.ceil(box[3]))))
            crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            if crop.size:
                crops.append((track_id, sample_index, crop))
        frame_index += 1
    capture.release()
    return crops


def load_model(gsr_root: Path, config_path: Path, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    sys.path.insert(0, str(gsr_root))
    cfg = OmegaConf.load(config_path)
    cfg.project_dir = str(gsr_root)
    cfg.model_dir = str(gsr_root / "pretrained_models")
    cfg.data_dir = str(gsr_root / "data")
    cfg.modules.reid.save_path = tempfile.mkdtemp(prefix="prtreid-")
    cfg.modules.reid.job_id = int(time.time())
    tracking_dataset = instantiate(cfg.dataset)
    # ExternalVideo does not expose these fields, while the ReID registration
    # API expects them on every TrackLab dataset.
    tracking_dataset.name = "SoccerNet"
    tracking_dataset.nickname = "sn"
    return instantiate(
        cfg.modules.reid,
        tracking_dataset=tracking_dataset,
        device=device,
    )


def extract_one(
    model,
    result_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    max_samples: int,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text())
    samples = select_samples(payload["frames"], max_samples)
    crops = read_crops(Path(payload["clip_path"]), samples)
    per_track_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    per_track_roles: dict[int, list[str]] = defaultdict(list)
    per_track_role_scores: dict[int, list[float]] = defaultdict(list)

    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        images = [torch.from_numpy(row[2]) for row in batch]
        table = pd.DataFrame(index=range(len(batch)))
        prediction = model.process(
            {"img": images},
            table,
            pd.DataFrame(index=table.index),
        )
        for row, embedding, role, role_score in zip(
            batch,
            prediction["embeddings"],
            prediction["role_detection"],
            prediction["role_confidence"],
        ):
            track_id = row[0]
            per_track_embeddings[track_id].append(
                l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
            )
            per_track_roles[track_id].append(str(role))
            per_track_role_scores[track_id].append(float(role_score))

    tracks = []
    for track_id in sorted(per_track_embeddings):
        vectors = np.stack(per_track_embeddings[track_id])
        embedding = l2_normalize(np.median(vectors, axis=0))
        role_votes = Counter(per_track_roles[track_id])
        tracks.append(
            {
                "track_id": track_id,
                "embedding": embedding.round(7).tolist(),
                "samples": len(vectors),
                "prt_role_vote": role_votes.most_common(1)[0][0],
                "prt_role_votes": dict(role_votes),
                "mean_prt_role_score": round(
                    float(np.mean(per_track_role_scores[track_id])), 6
                ),
            }
        )
    output = {
        "schema_version": 1,
        "clip_id": payload["clip_id"],
        "source_result": str(result_path),
        "model": "SoccerNet PRTReID baseline (BPBReID HRNet32, 256-D global)",
        "configuration": {
            "max_samples_per_track": max_samples,
            "aggregation": "median of per-crop L2-normalized embeddings, then L2",
        },
        "tracks": tracks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gsr-root", type=Path, default=Path("third_party/sn-gamestate")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    gsr_root = args.gsr_root.resolve()
    config_path = args.config
    if config_path is None:
        configs = sorted(gsr_root.glob("outputs/*/*/*/configs/config.yaml"))
        if not configs:
            raise RuntimeError("No resolved SoccerNet-GSR config found")
        config_path = configs[0]
    model = load_model(gsr_root, config_path.resolve(), args.device)
    for result_path in sorted(args.results_dir.glob("*.json")):
        destination = args.output_dir / result_path.name
        if args.skip_existing and destination.exists():
            continue
        output = extract_one(
            model,
            result_path,
            destination,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        print(f"{output['clip_id']}: {len(output['tracks'])} embedded tracks")


if __name__ == "__main__":
    main()
