#!/usr/bin/env python3
"""Apply a synthetic tactical graph model to converted SoccerNet-GSR frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_graph_classifier import TacticalGraphNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/tactical_graph_net.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = TacticalGraphNet(checkpoint["feature_dim"], len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    payload = np.load(args.graphs)
    features, masks = payload["features"], payload["masks"]
    metadata_path = args.graphs.with_suffix(".jsonl")
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    rows = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            x = torch.from_numpy(features[start : start + args.batch_size]).to(device)
            mask = torch.from_numpy(masks[start : start + args.batch_size]).to(device)
            probabilities = model(x, mask).softmax(1).cpu().numpy()
            for meta, probability in zip(metadata[start:], probabilities):
                prediction = int(probability.argmax())
                rows.append(
                    {
                        **meta,
                        "predicted_situation": labels[prediction],
                        "confidence": round(float(probability[prediction]), 6),
                        "probabilities": {
                            label: round(float(value), 6)
                            for label, value in zip(labels, probability)
                        },
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
