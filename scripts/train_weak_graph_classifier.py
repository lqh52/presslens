#!/usr/bin/env python3
"""Train and evaluate the graph model on sequence-disjoint GSR weak labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_graph_classifier import TacticalGraphNet


def load(graph_path: Path, weak_path: Path, threshold: float) -> tuple[TensorDataset, list[str]]:
    graph = np.load(graph_path)
    weak = np.load(weak_path)
    keep = (weak["labels"] >= 0) & (weak["confidence"] >= threshold)
    dataset = TensorDataset(
        torch.from_numpy(graph["features"][keep]),
        torch.from_numpy(graph["masks"][keep]),
        torch.from_numpy(weak["labels"][keep]),
        torch.from_numpy(weak["confidence"][keep]),
    )
    return dataset, weak["label_names"].tolist()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> tuple[float, float, list[list[int]]]:
    model.eval()
    correct = total = 0
    confidence, correctness = [], []
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    with torch.inference_mode():
        for x, mask, y, _ in loader:
            probability = model(x.to(device), mask.to(device)).softmax(1).cpu()
            score, pred = probability.max(1)
            correct += int((pred == y).sum())
            total += len(y)
            confidence.extend(score.tolist())
            correctness.extend((pred == y).float().tolist())
            for truth, guess in zip(y, pred):
                confusion[truth, guess] += 1
    # Simple calibration gap over ten confidence bins.
    ece = 0.0
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness)
    for low in np.linspace(0, 0.9, 10):
        selected = (confidence >= low) & (confidence < low + 0.1)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correctness[selected].mean())
    return correct / max(total, 1), float(ece), confusion.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graphs", type=Path, default=Path("data/graphs/gsr_train.npz"))
    parser.add_argument("--train-labels", type=Path, default=Path("data/graphs/gsr_train_weak.npz"))
    parser.add_argument("--valid-graphs", type=Path, default=Path("data/graphs/gsr_valid.npz"))
    parser.add_argument("--valid-labels", type=Path, default=Path("data/graphs/gsr_valid_weak.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/tactical_graph_weak.pt"))
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    train, labels = load(args.train_graphs, args.train_labels, args.confidence)
    valid, valid_labels = load(args.valid_graphs, args.valid_labels, args.confidence)
    if labels != valid_labels:
        raise ValueError("Train/validation label vocabularies differ")
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid, args.batch_size * 2, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticalGraphNet(train.tensors[0].shape[-1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    best = 0.0
    print(f"High-confidence frames: train={len(train)}, valid={len(valid)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = weight_total = 0.0
        for x, mask, y, weight in train_loader:
            x, mask, y, weight = x.to(device), mask.to(device), y.to(device), weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(x, mask), y)
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()
            loss_total += float((losses * weight).sum())
            weight_total += float(weight.sum())
        accuracy, ece, confusion = evaluate(model, valid_loader, device, len(labels))
        print(f"epoch={epoch:02d} loss={loss_total/weight_total:.4f} valid_accuracy={accuracy:.4f} ece={ece:.4f}")
        if accuracy > best:
            best = accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": labels, "feature_dim": train.tensors[0].shape[-1]}, args.output)
            args.output.with_suffix(".metrics.json").write_text(json.dumps({"weak_validation_accuracy": accuracy, "ece": ece, "labels": labels, "confusion": confusion, "train_frames": len(train), "valid_frames": len(valid)}, indent=2) + "\n")
    print(f"Best weak-label validation accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
