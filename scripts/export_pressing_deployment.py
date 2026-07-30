#!/usr/bin/env python3
"""Package the reviewed pressing graph model for a deployment handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.train_skillcorner_pressing_stg import SpatiotemporalGraphNet
except ModuleNotFoundError:
    from train_skillcorner_pressing_stg import SpatiotemporalGraphNet


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--review-labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--smoke-graphs", type=Path, required=True)
    parser.add_argument("--product-catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpatiotemporalGraphNet(
        checkpoint["feature_dim"],
        len(checkpoint["labels"]),
        width=checkpoint["width"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    graphs = np.load(args.smoke_graphs)
    features = torch.from_numpy(graphs["features"][:5][None].astype(np.float32))
    masks = torch.from_numpy(graphs["masks"][:5][None].astype(bool))
    traced = torch.jit.trace(model, (features, masks))
    scripted_path = args.output / "pressing-graph-model.torchscript.pt"
    traced.save(str(scripted_path))
    with torch.inference_mode():
        eager = model(features, masks)
        deployed = torch.jit.load(str(scripted_path))(features, masks)
    maximum_difference = float((eager - deployed).abs().max())
    if maximum_difference > 1e-5:
        raise RuntimeError(
            f"TorchScript smoke test differs from eager model: {maximum_difference}"
        )

    checkpoint_path = args.output / "pressing-graph-checkpoint.pt"
    shutil.copy2(args.checkpoint, checkpoint_path)
    metrics_source = args.checkpoint.with_suffix(".metrics.json")
    metrics_path = args.output / "training-metrics.json"
    shutil.copy2(metrics_source, metrics_path)

    reviewed = json.loads(args.review_labels.read_text())["labels"]
    predicted = {
        row["clip_id"]: row
        for row in json.loads(args.predictions.read_text())["clips"]
    }
    retained, excluded = [], []
    for clip_id, row in reviewed.items():
        if row["label"] == "exclude":
            excluded.append(
                {
                    "clip_id": clip_id,
                    "reason": row.get("notes") or "human unsure/excluded",
                }
            )
            continue
        summary = predicted[clip_id]["summary"]
        retained.append(
            {
                "clip_id": clip_id,
                "human_label": row["label"],
                "model_label": summary["label"],
                "confidence": summary["confidence"],
                "probabilities": summary["probabilities"],
                "windows": summary["windows"],
                "notes": row.get("notes", ""),
            }
        )
    retained_path = args.output / "retained-review.json"
    excluded_path = args.output / "excluded-review.json"
    retained_path.write_text(
        json.dumps({"schema_version": 1, "clips": retained}, indent=2) + "\n"
    )
    excluded_path.write_text(
        json.dumps({"schema_version": 1, "clips": excluded}, indent=2) + "\n"
    )

    schema_path = args.output / "input-schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": {
                    "features": {
                        "dtype": "float32",
                        "shape": ["batch", 5, 23, 13],
                    },
                    "masks": {
                        "dtype": "bool",
                        "shape": ["batch", 5, 23],
                    },
                },
                "feature_order": [
                    "x_normalized",
                    "y_normalized",
                    "vx_normalized",
                    "vy_normalized",
                    "possession_team",
                    "pressing_team",
                    "ball_team",
                    "goalkeeper_role",
                    "player_role",
                    "referee_role",
                    "other_role",
                    "ball_role",
                    "ball_control",
                ],
                "labels": checkpoint["labels"],
                "coordinate_contract": (
                    "Possession team attacks toward increasing normalized x."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    smoke_path = args.output / "smoke-input.npz"
    np.savez_compressed(
        smoke_path,
        features=features.numpy(),
        masks=masks.numpy(),
        expected_logits=eager.numpy(),
    )

    files = [
        scripted_path,
        checkpoint_path,
        metrics_path,
        retained_path,
        excluded_path,
        schema_path,
        smoke_path,
    ]
    product_catalog = None
    if args.product_catalog:
        product_catalog = json.loads(args.product_catalog.read_text())
        product_path = args.output / "product-catalog.json"
        shutil.copy2(args.product_catalog, product_path)
        files.append(product_path)
    manifest = {
        "schema_version": 1,
        "package": "presslens-pressing-graph-4class",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "PyTorch TorchScript; CPU or CUDA",
        "labels": checkpoint["labels"],
        "quality_contract": {
            "minimum_player_nodes_per_frame": 6,
            "possession_confidence_minimum": 0.45,
            "ball_holder_distance_maximum_m": 5.0,
            "sequence_frames": 5,
            "calibration": (
                "Only observed or RANSAC-validated feature-transported "
                "homographies; unreliable frames are omitted."
            ),
            "abstain_when": (
                "No valid five-frame temporal window survives projection, "
                "identity, ball, and possession gates."
            ),
        },
        "review": {
            "retained_clips": len(retained),
            "excluded_clips": len(excluded),
            "retained_model_agreement": (
                sum(row["human_label"] == row["model_label"] for row in retained)
                / max(len(retained), 1)
            ),
            "note": "Retained agreement is on training seeds, not held-out accuracy.",
        },
        "product_catalog": (
            {
                "clips": len(product_catalog["clips"]),
                "quality_counts": {
                    quality: sum(
                        row["quality"] == quality
                        for row in product_catalog["clips"]
                    )
                    for quality in ("best", "good")
                },
                "note": (
                    "Product examples merge both review rounds and are not "
                    "presented as train/test partitions."
                ),
            }
            if product_catalog
            else None
        ),
        "torchscript_smoke_maximum_logit_difference": maximum_difference,
        "files": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Packaged {len(retained)} retained and {len(excluded)} excluded clips "
        f"at {args.output}"
    )


if __name__ == "__main__":
    main()
