#!/usr/bin/env python3
"""Generate controlled build-up/pressing graphs with hard negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


LABELS = ["unstructured", "central_screen", "trap_left", "trap_right", "high_press"]
# Match the observable GSR vocabulary: goalkeeper, player, referee, other, ball.
ROLES = np.array([0] + [1] * 10 + [0] + [1] * 10 + [4])

# A 4-3-3 in normalized coordinates, viewed in the possession direction.
POSSESSION_BASE = np.array(
    [
        [0.06, 0.50],
        [0.20, 0.12], [0.18, 0.38], [0.18, 0.62], [0.20, 0.88],
        [0.34, 0.28], [0.38, 0.50], [0.34, 0.72],
        [0.52, 0.16], [0.54, 0.50], [0.52, 0.84],
    ]
)


def make_graph(rng: np.random.Generator, label: int) -> tuple[np.ndarray, np.ndarray]:
    attack = POSSESSION_BASE.copy()
    attack[:, 0] += rng.uniform(-0.05, 0.08)
    attack += rng.normal(0, [0.018, 0.025], attack.shape)
    holder = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7]))
    ball = attack[holder] + rng.normal(0, 0.006, 2)

    # Start defenders in a plausible mid/high block, then impose relations.
    defend = POSSESSION_BASE.copy()
    defend[:, 0] = 0.80 - defend[:, 0] + rng.normal(0, 0.025, 11)
    defend[:, 1] += rng.normal(0, 0.035, 11)

    if label == 0:  # density-matched but not coordinated
        defend[:6, 0] = rng.uniform(0.30, 0.58, 6)
        defend[:6, 1] = rng.uniform(0.08, 0.92, 6)
    elif label == 1:  # cover shadow/screen around the central lane
        defend[:3] = np.array([[ball[0] + 0.10, 0.36], [ball[0] + 0.08, 0.50], [ball[0] + 0.10, 0.64]])
        defend[3:6] = np.array([[ball[0] + 0.20, 0.31], [ball[0] + 0.18, 0.50], [ball[0] + 0.20, 0.69]])
        defend[:6] += rng.normal(0, 0.018, (6, 2))
    elif label in (2, 3):  # curved press that closes inside and leaves touchline
        side = 0.12 if label == 2 else 0.88
        ball[:] = [rng.uniform(0.18, 0.40), side + rng.normal(0, 0.025)]
        attack[holder] = ball
        inward = 1 if label == 2 else -1
        defend[:4] = np.array(
            [
                [ball[0] + 0.04, ball[1] + inward * 0.07],
                [ball[0] + 0.11, ball[1] + inward * 0.16],
                [ball[0] - 0.02, ball[1] + inward * 0.18],
                [ball[0] + 0.19, 0.50],
            ]
        ) + rng.normal(0, 0.014, (4, 2))
    else:  # multiple close opponents with forward velocity toward the ball
        angles = np.linspace(-1.0, 1.0, 5)
        defend[:5] = ball + np.column_stack(
            [rng.uniform(0.05, 0.15, 5), angles * rng.uniform(0.08, 0.18)]
        )
        defend[:5] += rng.normal(0, 0.012, (5, 2))

    positions = np.clip(np.vstack([attack, defend, ball]), 0.01, 0.99)
    velocity = rng.normal(0, 0.015, (23, 2))
    if label == 4:
        delta = ball - defend[:5]
        velocity[11:16] = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6) * rng.uniform(0.03, 0.07)

    team = np.zeros((23, 3), dtype=np.float32)
    team[:11, 0], team[11:22, 1], team[22, 2] = 1, 1, 1
    role = np.eye(5, dtype=np.float32)[ROLES]
    control = np.zeros((23, 1), dtype=np.float32)
    control[holder] = 1
    features = np.concatenate([positions, velocity, team, role, control], axis=1)
    return features.astype(np.float32), np.int64(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/graphs/synthetic.npz"))
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    labels = np.arange(args.samples) % len(LABELS)
    rng.shuffle(labels)
    samples = [make_graph(rng, int(label))[0] for label in labels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(samples),
        labels=labels.astype(np.int64),
        label_names=np.array(LABELS),
    )
    print(f"Wrote {args.samples} graphs with shape {samples[0].shape} to {args.output}")


if __name__ == "__main__":
    main()
