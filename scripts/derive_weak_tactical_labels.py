#!/usr/bin/env python3
"""Derive auditable weak tactical labels from canonical SoccerNet-GSR graphs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LABELS = ["unstructured", "central_screen", "trap_left", "trap_right", "high_press"]
DIRECTION_DEPENDENT_LABELS = frozenset(
    {"central_screen", "trap_left", "trap_right", "high_press"}
)
PITCH_SCALE = np.array([105.0, 68.0], dtype=np.float32)


def descriptors(features: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    xy = features[:, :2]
    velocity = features[:, 2:4] * PITCH_SCALE
    possession = mask & (features[:, 4] > 0.5)
    pressing = mask & (features[:, 5] > 0.5)
    ball_nodes = np.flatnonzero(mask & (features[:, 6] > 0.5))
    if not len(ball_nodes) or not possession.any() or not pressing.any():
        return {}
    ball = xy[ball_nodes[0]]
    press_xy = xy[pressing]
    press_v = velocity[pressing]
    delta_m = (ball - press_xy) * PITCH_SCALE
    distance = np.linalg.norm(delta_m, axis=1)
    direction = delta_m / np.maximum(distance[:, None], 1e-6)
    approach_speed = (press_v * direction).sum(1)
    ahead = press_xy[:, 0] > ball[0]
    central_corridor = ahead & (np.abs(press_xy[:, 1] - 0.5) < 0.14) & (press_xy[:, 0] < ball[0] + 0.32)
    return {
        "ball_x": float(ball[0]),
        "ball_y": float(ball[1]),
        "press_within_8m": int((distance < 8).sum()),
        "press_within_12m": int((distance < 12).sum()),
        "press_within_16m": int((distance < 16).sum()),
        "nearest_press_m": float(distance.min()),
        "mean_top3_press_m": float(np.sort(distance)[:3].mean()) if len(distance) >= 3 else float(distance.mean()),
        "max_approach_mps": float(approach_speed.max()),
        "approaching_pressers": int((approach_speed > 0.7).sum()),
        "central_screeners": int(central_corridor.sum()),
        "inside_left_trap": int(((distance < 12) & (press_xy[:, 1] > ball[1] + 0.025)).sum()),
        "inside_right_trap": int(((distance < 12) & (press_xy[:, 1] < ball[1] - 0.025)).sum()),
    }


def weak_label(d: dict[str, float], possession_confident: bool) -> tuple[int, float, str]:
    if not d or not possession_confident:
        return -1, 0.0, "abstain_possession"
    if d["ball_y"] < 0.22 and d["press_within_12m"] >= 2 and d["inside_left_trap"] >= 2:
        confidence = min(0.95, 0.62 + 0.08 * d["inside_left_trap"] + 0.06 * (0.22 - d["ball_y"]) / 0.22)
        return 2, confidence, "touchline_left+inside_pressure"
    if d["ball_y"] > 0.78 and d["press_within_12m"] >= 2 and d["inside_right_trap"] >= 2:
        confidence = min(0.95, 0.62 + 0.08 * d["inside_right_trap"] + 0.06 * (d["ball_y"] - 0.78) / 0.22)
        return 3, confidence, "touchline_right+inside_pressure"
    if (
        d["ball_x"] < 0.48
        and d["press_within_16m"] >= 3
        and (d["approaching_pressers"] >= 1 or d["press_within_8m"] >= 2)
    ):
        confidence = min(0.92, 0.58 + 0.06 * d["press_within_16m"] + 0.04 * d["approaching_pressers"])
        return 4, confidence, "defensive_third+dense_active_pressure"
    if d["ball_x"] < 0.60 and d["central_screeners"] >= 2 and d["press_within_16m"] >= 2:
        confidence = min(0.90, 0.58 + 0.07 * d["central_screeners"])
        return 1, confidence, "central_corridor_screened"
    if d["press_within_12m"] == 0 and d["nearest_press_m"] > 14:
        confidence = min(0.90, 0.62 + 0.02 * (d["nearest_press_m"] - 14))
        return 0, confidence, "no_local_pressure"
    return -1, 0.0, "abstain_ambiguous"


def direction_is_trusted(meta: dict) -> bool:
    """Require explicit, internally consistent canonical direction metadata."""

    if meta.get("direction_confident") is not True:
        return False
    for key in ("direction_status", "direction_source"):
        normalized = (
            str(meta.get(key, ""))
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if any(
            marker in normalized
            for marker in (
                "ambiguous",
                "abstain",
                "invalid",
                "unknown",
                "undetermined",
                "unavailable",
            )
        ):
            return False
    raw = meta.get("attacking_direction_raw")
    if isinstance(raw, bool) or raw not in (-1, 1):
        return False
    expected_label = "left_to_right" if int(raw) == 1 else "right_to_left"
    return (
        str(meta.get("attacking_direction_label", "")).strip().lower()
        == expected_label
    )


def gate_direction_dependent_label(
    label: int,
    score: float,
    rule: str,
    *,
    direction_trusted: bool,
) -> tuple[int, float, str]:
    """Abstain when a rule relies on untrusted attacker-relative coordinates."""

    if (
        label >= 0
        and LABELS[label] in DIRECTION_DEPENDENT_LABELS
        and not direction_trusted
    ):
        return -1, 0.0, "abstain_direction"
    return label, score, rule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph_data = np.load(args.graphs)
    metadata = [json.loads(line) for line in args.graphs.with_suffix(".jsonl").read_text().splitlines()]
    labels, confidence, rows = [], [], []
    for features, mask, meta in zip(graph_data["features"], graph_data["masks"], metadata):
        values = descriptors(features, mask)
        label, score, rule = weak_label(values, bool(meta["possession_confident"]))
        label, score, rule = gate_direction_dependent_label(
            label,
            score,
            rule,
            direction_trusted=direction_is_trusted(meta),
        )
        labels.append(label)
        confidence.append(score)
        rows.append({**meta, "weak_label": LABELS[label] if label >= 0 else "abstain", "weak_confidence": round(score, 4), "weak_rule": rule, "descriptors": {k: round(v, 4) for k, v in values.items()}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=np.asarray(labels, dtype=np.int64),
        confidence=np.asarray(confidence, dtype=np.float32),
        label_names=np.asarray(LABELS),
    )
    args.output.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts = Counter(row["weak_label"] for row in rows)
    print(f"Wrote {len(rows)} weak labels to {args.output}: {dict(counts)}")


if __name__ == "__main__":
    main()
