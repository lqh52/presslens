#!/usr/bin/env python3
"""Extract track-level DINO body/torso appearance features.

The model is configurable. The default is the public DINOv2-small checkpoint;
switch --model to an accessible DINOv3 checkpoint without changing the cache
schema or downstream review controls.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from extract_prtreid_track_features import (
    l2_normalize,
    read_crops,
    select_samples,
)


def torso_from_body(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return image[
        int(0.06 * height) : max(int(0.58 * height), 1),
        int(0.14 * width) : max(int(0.86 * width), 1),
    ]


@torch.inference_mode()
def extract_one(
    processor,
    model,
    model_name: str,
    result_path: Path,
    output_path: Path,
    *,
    device: str,
    batch_size: int,
    max_samples: int,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text())
    samples = select_samples(payload["frames"], max_samples)
    crops = read_crops(Path(payload["clip_path"]), samples)
    per_track: dict[int, list[np.ndarray]] = defaultdict(list)

    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        bodies = [row[2] for row in batch]
        torsos = [torso_from_body(image) for image in bodies]
        inputs = processor(images=bodies + torsos, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            output = model(**inputs).last_hidden_state[:, 0]
        vectors = output.float().cpu().numpy()
        count = len(batch)
        for row, body, torso in zip(batch, vectors[:count], vectors[count:]):
            per_track[row[0]].append(
                l2_normalize(
                    np.concatenate([l2_normalize(body), l2_normalize(torso)])
                )
            )

    tracks = []
    for track_id in sorted(per_track):
        vectors = np.stack(per_track[track_id])
        embedding = l2_normalize(np.median(vectors, axis=0))
        tracks.append(
            {
                "track_id": track_id,
                "embedding": embedding.round(7).tolist(),
                "samples": len(vectors),
            }
        )
    output = {
        "schema_version": 1,
        "clip_id": payload["clip_id"],
        "source_result": str(result_path),
        "model": model_name,
        "configuration": {
            "features": "concatenated CLS body and upper-torso embeddings",
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
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    for result_path in sorted(args.results_dir.glob("*.json")):
        destination = args.output_dir / result_path.name
        if args.skip_existing and destination.exists():
            continue
        output = extract_one(
            processor,
            model,
            args.model,
            result_path,
            destination,
            device=args.device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        print(f"{output['clip_id']}: {len(output['tracks'])} DINO tracks")


if __name__ == "__main__":
    main()
