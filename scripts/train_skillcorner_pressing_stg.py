#!/usr/bin/env python3
"""Train a fixture-split spatiotemporal graph classifier for pressing phases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.train_graph_classifier import GraphBlock
except ModuleNotFoundError:
    from train_graph_classifier import GraphBlock


class SpatiotemporalGraphNet(nn.Module):
    def __init__(
        self, feature_dim: int, classes: int, width: int = 80
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.blocks = nn.ModuleList([GraphBlock(width), GraphBlock(width)])
        self.temporal = nn.GRU(
            width * 2,
            width,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(width, classes),
        )

    def encode_frames(
        self, features: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        batch, time, nodes, feature_dim = features.shape
        flat_features = features.reshape(batch * time, nodes, feature_dim)
        flat_masks = masks.reshape(batch * time, nodes)
        h = self.input(flat_features) * flat_masks[..., None]
        xy = flat_features[..., :2]
        for block in self.blocks:
            h = block(h, xy, flat_masks)
        possession = flat_features[..., 4] * flat_masks
        pressing = flat_features[..., 5] * flat_masks
        pooled = torch.cat(
            [
                (h * possession[..., None]).sum(1)
                / possession.sum(1, keepdim=True).clamp_min(1),
                (h * pressing[..., None]).sum(1)
                / pressing.sum(1, keepdim=True).clamp_min(1),
            ],
            dim=-1,
        )
        return pooled.reshape(batch, time, -1)

    def forward(
        self, features: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        sequence = self.encode_frames(features, masks)
        encoded, _ = self.temporal(sequence)
        return self.head(encoded.mean(1))


def load_sequences(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, masks, labels = [], [], []
    for path in paths:
        payload = np.load(path)
        sequence_ids = payload["sequence_index"]
        for sequence_id in np.unique(sequence_ids):
            indices = np.flatnonzero(sequence_ids == sequence_id)
            if len(indices) != 5:
                continue
            order = indices[np.argsort(payload["frame_index"][indices])]
            sequence_labels = payload["labels"][order]
            if not np.all(sequence_labels == sequence_labels[0]):
                raise ValueError(f"Inconsistent sequence labels in {path}")
            features.append(payload["features"][order])
            masks.append(payload["masks"][order])
            labels.append(sequence_labels[0])
    return (
        np.stack(features).astype(np.float32),
        np.stack(masks).astype(bool),
        np.asarray(labels, dtype=np.int64),
    )


def temporal_windows(rows: list[dict], indices: list[int]) -> list[list[int]]:
    ordered = sorted(indices, key=lambda index: int(rows[index]["frame"]))
    frame_values = np.asarray(
        [int(rows[index]["frame"]) for index in ordered], dtype=np.int64
    )
    low, high = int(frame_values[0]), int(frame_values[-1])
    if high - low >= 50:
        centers = list(range(low + 25, high - 24, 10))
        if high - 25 not in centers:
            centers.append(high - 25)
        targets_by_window = [
            [center + offset for offset in (-25, -12, 0, 12, 25)]
            for center in centers
        ]
    else:
        targets_by_window = [
            np.linspace(low, high, 5).round().astype(int).tolist()
        ]
    windows = []
    for targets in targets_by_window:
        selected = [
            ordered[int(np.abs(frame_values - target).argmin())]
            for target in targets
        ]
        if len(set(selected)) == 5:
            windows.append(selected)
    return windows


def load_review_seeds(
    graph_path: Path,
    metadata_path: Path,
    seeds_path: Path,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(graph_path)
    rows = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    reviewed = json.loads(seeds_path.read_text())["labels"]
    by_clip: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["clip_id"] in reviewed:
            by_clip[row["clip_id"]].append(index)
    features, masks, labels = [], [], []
    for clip_id, indices in by_clip.items():
        for window in temporal_windows(rows, indices):
            features.append(payload["features"][window])
            masks.append(payload["masks"][window])
            labels.append(class_names.index(reviewed[clip_id]))
    return (
        np.stack(features).astype(np.float32),
        np.stack(masks).astype(bool),
        np.asarray(labels, dtype=np.int64),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: int,
) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = total = 0
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    with torch.inference_mode():
        for features, masks, labels in loader:
            predictions = model(
                features.to(device), masks.to(device)
            ).argmax(1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
            for truth, guess in zip(labels, predictions):
                confusion[truth, guess] += 1
    return correct / max(total, 1), confusion.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--valid-fixture", default="1996435")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--review-graphs", type=Path)
    parser.add_argument("--review-metadata", type=Path)
    parser.add_argument("--review-seeds", type=Path)
    parser.add_argument("--seed-repeat", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(args.data_dir.glob("*.npz"))
    valid_paths = [path for path in paths if path.stem == args.valid_fixture]
    train_paths = [path for path in paths if path.stem != args.valid_fixture]
    if not valid_paths or not train_paths:
        raise ValueError("Need both training and held-out fixture files")
    train_x, train_m, train_y = load_sequences(train_paths)
    valid_x, valid_m, valid_y = load_sequences(valid_paths)
    class_names = np.load(paths[0])["label_names"].tolist()
    reviewed_sequences = 0
    if args.review_seeds:
        if not args.review_graphs or not args.review_metadata:
            raise ValueError(
                "--review-seeds requires --review-graphs and --review-metadata"
            )
        seed_x, seed_m, seed_y = load_review_seeds(
            args.review_graphs,
            args.review_metadata,
            args.review_seeds,
            class_names,
        )
        reviewed_sequences = len(seed_y)
        train_x = np.concatenate(
            [train_x, np.repeat(seed_x, args.seed_repeat, axis=0)]
        )
        train_m = np.concatenate(
            [train_m, np.repeat(seed_m, args.seed_repeat, axis=0)]
        )
        train_y = np.concatenate(
            [train_y, np.repeat(seed_y, args.seed_repeat, axis=0)]
        )

    train_data = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_m),
        torch.from_numpy(train_y),
    )
    valid_data = TensorDataset(
        torch.from_numpy(valid_x),
        torch.from_numpy(valid_m),
        torch.from_numpy(valid_y),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_data, args.batch_size * 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatiotemporalGraphNet(
        train_x.shape[-1], len(class_names)
    ).to(device)
    counts = np.bincount(train_y, minlength=len(class_names))
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-4
    )
    best = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for features, masks, labels in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # Match broadcast reconstruction failures without adding visual input.
            if epoch > 3:
                player_nodes = (
                    (features[..., 4] + features[..., 5]) > 0
                )
                dropped = (
                    torch.rand(player_nodes.shape, device=device) < 0.08
                ) & player_nodes
                masks = masks & ~dropped
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, masks)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss) * len(labels)
        accuracy, confusion = evaluate(
            model, valid_loader, device, len(class_names)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_data),
                "valid_accuracy": accuracy,
            }
        )
        if accuracy > best:
            best = accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": class_names,
                    "feature_dim": train_x.shape[-1],
                    "width": 80,
                    "sequence_length": train_x.shape[1],
                },
                args.output,
            )
            args.output.with_suffix(".metrics.json").write_text(
                json.dumps(
                    {
                        "validation_fixture": args.valid_fixture,
                        "validation_accuracy": accuracy,
                        "confusion": confusion,
                        "labels": class_names,
                        "training_fixtures": [
                            path.stem for path in train_paths
                        ],
                        "train_sequences": len(train_data),
                        "valid_sequences": len(valid_data),
                        "human_seed_sequences": reviewed_sequences,
                        "human_seed_repeat": (
                            args.seed_repeat if reviewed_sequences else 0
                        ),
                        "class_counts_train": counts.tolist(),
                        "history": history,
                    },
                    indent=2,
                )
                + "\n"
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} loss={running / len(train_data):.4f} "
                f"valid_accuracy={accuracy:.4f}"
            )
    print(
        f"best_valid_accuracy={best:.4f} model={args.output} device={device}"
    )


if __name__ == "__main__":
    main()
