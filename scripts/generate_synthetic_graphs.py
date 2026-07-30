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

DEFENDING_BASE = np.array(
    [
        [0.94, 0.50],
        [0.79, 0.14], [0.77, 0.38], [0.77, 0.62], [0.79, 0.86],
        [0.62, 0.15], [0.60, 0.38], [0.60, 0.62], [0.62, 0.85],
        [0.48, 0.38], [0.48, 0.62],
    ]
)


def make_graph(rng: np.random.Generator, label: int) -> tuple[np.ndarray, np.ndarray]:
    attack = POSSESSION_BASE.copy()
    attack[:, 0] += rng.uniform(-0.05, 0.08)
    attack += rng.normal(0, [0.018, 0.025], attack.shape)
    holder = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7]))
    ball = attack[holder] + rng.normal(0, 0.006, 2)

    # Start with a coherent 4-4-2. Tactical rules deform the unit while keeping
    # its covering lines, instead of scattering unrelated individual players.
    defend = DEFENDING_BASE.copy()
    block_shift = np.clip(ball[0] - 0.32, -0.08, 0.12)
    defend[1:, 0] += block_shift
    defend += rng.normal(0, [0.012, 0.018], defend.shape)

    if label == 0:  # realistic transition/broken shape, without random scatter
        defend[5:9, 0] += rng.normal(-0.05, 0.025, 4)
        defend[9:, 0] += rng.normal(-0.08, 0.02, 2)
        displaced = rng.choice(np.arange(1, 11), size=2, replace=False)
        defend[displaced] += rng.normal([0.0, 0.0], [0.055, 0.07], (2, 2))
    elif label == 1:  # cover shadow/screen around the central lane
        ball[1] = rng.uniform(0.38, 0.62)
        attack[holder] = ball
        # Two forwards screen the pivot lane; central midfield and back line
        # retain depth behind them.
        defend[9:11] = np.array(
            [[ball[0] + 0.09, ball[1] - 0.10], [ball[0] + 0.09, ball[1] + 0.10]]
        )
        defend[6:8] = np.array(
            [[ball[0] + 0.21, 0.40], [ball[0] + 0.21, 0.60]]
        )
        defend[5, 0] = defend[8, 0] = ball[0] + 0.24
        defend[1:5, 0] = ball[0] + 0.37
        defend[9:] += rng.normal(0, [0.012, 0.015], (2, 2))
    elif label in (2, 3):  # curved press that closes inside and leaves touchline
        side = 0.12 if label == 2 else 0.88
        ball[:] = [rng.uniform(0.18, 0.40), side + rng.normal(0, 0.025)]
        attack[holder] = ball
        sign = 1 if label == 2 else -1
        near_fullback = 1 if label == 2 else 4
        near_midfielder = 5 if label == 2 else 8
        near_forward = 9 if label == 2 else 10
        defend[near_forward] = ball + [0.035, sign * 0.065]
        defend[near_midfielder] = ball + [0.10, sign * 0.14]
        defend[near_fullback] = ball + [0.22, sign * 0.13]
        # Far-side players tuck in but preserve the block's three lines.
        defend[1:5, 1] = np.clip(defend[1:5, 1], 0.22, 0.78)
        defend[5:9, 1] = np.clip(defend[5:9, 1], 0.20, 0.80)
        defend[[near_forward, near_midfielder, near_fullback]] += rng.normal(
            0, [0.01, 0.012], (3, 2)
        )
    else:  # multiple close opponents with forward velocity toward the ball
        defend[0] = [0.94, 0.50]
        defend[1:5, 0] = np.array([0.68, 0.66, 0.66, 0.68])
        defend[5:9, 0] = np.array([0.51, 0.47, 0.47, 0.51])
        defend[9:11, 0] = 0.34
        closest = np.argsort(np.linalg.norm(defend[1:] - ball, axis=1))[:3] + 1
        angles = np.linspace(-0.8, 0.8, 3)
        defend[closest] = ball + np.column_stack(
            [rng.uniform(0.045, 0.11, 3), angles * rng.uniform(0.06, 0.13)]
        )
        defend[closest] += rng.normal(0, 0.01, (3, 2))

    positions = np.clip(np.vstack([attack, defend, ball]), 0.01, 0.99)
    velocity = rng.normal(0, 0.015, (23, 2))
    if label == 4:
        delta = ball - defend
        nearest = np.argsort(np.linalg.norm(delta, axis=1))[:4]
        velocity[11 + nearest] = (
            delta[nearest]
            / (np.linalg.norm(delta[nearest], axis=1, keepdims=True) + 1e-6)
            * rng.uniform(0.03, 0.06)
        )

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
