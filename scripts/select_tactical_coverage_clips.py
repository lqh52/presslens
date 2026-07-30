#!/usr/bin/env python3
"""Use existing canonical graphs only to select clips for balanced reprocessing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

try:
    from .train_graph_classifier import TacticalGraphNet
except ImportError:
    from train_graph_classifier import TacticalGraphNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = TacticalGraphNet(checkpoint["feature_dim"], len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    ranked = {label: [] for label in labels}
    with torch.inference_mode():
        for graph_path in sorted(args.graphs_dir.glob("*.npz")):
            if graph_path.stem.endswith("-weak"):
                continue
            metadata_path = graph_path.with_suffix(".jsonl")
            if not metadata_path.exists():
                continue
            payload = np.load(graph_path)
            metadata = [
                json.loads(line) for line in metadata_path.read_text().splitlines()
            ]
            features = torch.from_numpy(payload["features"]).to(args.device)
            masks = torch.from_numpy(payload["masks"]).to(args.device)
            probability = model(features, masks).softmax(1).cpu().numpy()
            accepted = []
            for row, values in zip(metadata, probability):
                prediction = int(values.argmax())
                confidence = float(values[prediction])
                if (
                    row.get("possession_confident")
                    and row.get("direction_confident")
                    and confidence >= 0.65
                ):
                    accepted.append((labels[prediction], confidence))
            counts = Counter(label for label, _ in accepted)
            for label, count in counts.items():
                values = [score for name, score in accepted if name == label]
                ranked[label].append(
                    {
                        "id": graph_path.stem,
                        "predicted_class": label,
                        "class_frames": count,
                        "accepted_frames": len(accepted),
                        "coverage_score": round(
                            count / max(len(metadata), 1) * float(np.mean(values)), 6
                        ),
                    }
                )
    source = {
        row["id"]: row
        for row in json.loads(args.source_manifest.read_text())["clips"]
    }
    selected, used = [], set()
    for label in labels:
        choices = sorted(
            ranked[label],
            key=lambda row: (row["coverage_score"], row["class_frames"]),
            reverse=True,
        )
        for candidate in choices:
            if candidate["id"] in used or candidate["id"] not in source:
                continue
            selected.append({**source[candidate["id"]], "triage": candidate})
            used.add(candidate["id"])
            if sum(row["triage"]["predicted_class"] == label for row in selected) >= args.per_class:
                break
    result = {
        "schema_version": 1,
        "selection_method": "old GSR graph triage only; final pipeline must be rerun",
        "model": str(args.model),
        "clips": selected,
        "availability": {
            label: len(ranked[label]) for label in labels
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Selected {len(selected)} clips: "
        + ", ".join(
            f"{label}={sum(row['triage']['predicted_class'] == label for row in selected)}"
            for label in labels
        )
    )


if __name__ == "__main__":
    main()
