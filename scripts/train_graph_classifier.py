#!/usr/bin/env python3
"""Train a compact relation-aware graph classifier without PyG dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class GraphBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 2 + 3, width), nn.GELU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))

    def forward(
        self, h: torch.Tensor, xy: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        relative = xy[:, None, :, :] - xy[:, :, None, :]
        distance = relative.square().sum(-1, keepdim=True).sqrt()
        pair = torch.cat(
            [
                h[:, :, None, :].expand(-1, -1, h.size(1), -1),
                h[:, None, :, :].expand(-1, h.size(1), -1, -1),
                relative,
                distance,
            ],
            dim=-1,
        )
        weights = torch.exp(-distance * 8.0) * mask[:, None, :, None]
        message = (self.message(pair) * weights).sum(2) / weights.sum(2).clamp_min(1e-6)
        return (h + self.update(torch.cat([h, message], dim=-1))) * mask[..., None]


class TacticalGraphNet(nn.Module):
    def __init__(self, feature_dim: int, classes: int, width: int = 96) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([GraphBlock(width), GraphBlock(width), GraphBlock(width)])
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, classes))

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(features.shape[:2], device=features.device)
        h, xy = self.input(features) * mask[..., None], features[..., :2]
        for block in self.blocks:
            h = block(h, xy, mask)
        possession = features[..., 4]
        pressing = features[..., 5]
        pooled = torch.cat(
            [
                (h * possession[..., None]).sum(1)
                / possession.sum(1, keepdim=True).clamp_min(1),
                (h * pressing[..., None]).sum(1)
                / pressing.sum(1, keepdim=True).clamp_min(1),
            ],
            dim=-1,
        )
        return self.head(pooled)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = total = 0
    classes = model.head[-1].out_features
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
            for truth, guess in zip(y, pred):
                confusion[truth, guess] += 1
    return correct / total, confusion.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/graphs/synthetic.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/tactical_graph_net.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    payload = np.load(args.data)
    x = torch.from_numpy(payload["features"])
    y = torch.from_numpy(payload["labels"])
    names = payload["label_names"].tolist()
    dataset = TensorDataset(x, y)
    train_size = int(len(dataset) * 0.8)
    train, valid = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        train,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid,
        args.batch_size * 2,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticalGraphNet(x.shape[-1], len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(labels)
        score, confusion = accuracy(model, valid_loader, device)
        print(f"epoch={epoch:02d} loss={running/train_size:.4f} valid_accuracy={score:.4f}")
        if score > best:
            best = score
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": names, "feature_dim": x.shape[-1]}, args.output)
            args.output.with_suffix(".metrics.json").write_text(
                json.dumps({"validation_accuracy": score, "labels": names, "confusion": confusion}, indent=2) + "\n"
            )
    print(f"Best validation accuracy: {best:.4f}; model: {args.output}")


if __name__ == "__main__":
    main()
