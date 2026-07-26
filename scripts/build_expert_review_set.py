#!/usr/bin/env python3
"""Create a class/confidence-balanced, sequence-diverse expert review set."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def confidence_band(value: float) -> str:
    if value < 0.60:
        return "low"
    if value < 0.85:
        return "medium"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, default=Path("data/graphs/gsr_valid.npz"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/graphs/gsr_valid_weak_model_predictions.jsonl"),
    )
    parser.add_argument(
        "--weak-labels",
        type=Path,
        default=Path("data/graphs/gsr_valid_weak.jsonl"),
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/gamestate-2025/valid.zip")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/review/gsr_expert")
    )
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    weak = [json.loads(line) for line in args.weak_labels.read_text().splitlines()]
    graph_data = np.load(args.graphs)
    if not (len(predictions) == len(weak) == len(graph_data["features"])):
        raise ValueError("Graph, prediction, and weak-label rows are not aligned")

    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, (prediction, weak_row) in enumerate(zip(predictions, weak)):
        label = prediction["predicted_situation"]
        band = confidence_band(float(prediction["confidence"]))
        candidates[(label, band)].append(
            {
                "graph_index": index,
                "sequence": prediction["sequence"],
                "frame": prediction["frame"],
                "time_seconds": prediction["time_seconds"],
                "model_prediction": label,
                "model_confidence": prediction["confidence"],
                "weak_label": weak_row["weak_label"],
                "weak_rule": weak_row["weak_rule"],
                "weak_confidence": weak_row["weak_confidence"],
                "possession_confident": prediction["possession_confident"],
                "visible_nodes": prediction["visible_nodes"],
            }
        )

    rng = random.Random(args.seed)
    labels = sorted({row["predicted_situation"] for row in predictions})
    selected: list[dict] = []
    per_band = args.per_class // 3
    for label in labels:
        chosen_ids: set[tuple[str, int]] = set()
        sequence_counts: Counter[str] = Counter()
        for band in ("low", "medium", "high"):
            pool = candidates[(label, band)]
            rng.shuffle(pool)
            chosen = []
            # Half agreement, half disagreement in each class/confidence cell.
            for agrees, target in (
                (False, per_band // 2),
                (True, per_band - per_band // 2),
            ):
                subset = [
                    row
                    for row in pool
                    if (row["model_prediction"] == row["weak_label"]) == agrees
                ]
                subset.sort(key=lambda row: sequence_counts[row["sequence"]])
                taken = 0
                for row in subset:
                    identity = (row["sequence"], row["frame"])
                    if identity in chosen_ids or sequence_counts[row["sequence"]] >= 2:
                        continue
                    chosen.append(row)
                    chosen_ids.add(identity)
                    sequence_counts[row["sequence"]] += 1
                    taken += 1
                    if taken == target:
                        break
            if len(chosen) < per_band:
                for row in sorted(pool, key=lambda item: sequence_counts[item["sequence"]]):
                    identity = (row["sequence"], row["frame"])
                    if identity in chosen_ids or sequence_counts[row["sequence"]] >= 2:
                        continue
                    chosen.append(row)
                    chosen_ids.add(identity)
                    sequence_counts[row["sequence"]] += 1
                    if len(chosen) == per_band:
                        break
            selected.extend(chosen)
        # Fill sparse confidence bands while retaining sequence diversity.
        missing = args.per_class - sum(x["model_prediction"] == label for x in selected)
        if missing:
            pool = [row for key, rows in candidates.items() if key[0] == label for row in rows]
            rng.shuffle(pool)
            pool.sort(key=lambda row: sequence_counts[row["sequence"]])
            for row in pool:
                identity = (row["sequence"], row["frame"])
                if identity in chosen_ids or sequence_counts[row["sequence"]] >= 3:
                    continue
                selected.append(row)
                chosen_ids.add(identity)
                sequence_counts[row["sequence"]] += 1
                missing -= 1
                if not missing:
                    break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, masks = graph_data["features"], graph_data["masks"]
    with zipfile.ZipFile(args.archive) as archive:
        for review_index, row in enumerate(selected, 1):
            filename = f"{row['frame']:06d}.jpg"
            member = f"{row['sequence']}/img1/{filename}"
            image_name = f"{review_index:03d}_{row['sequence']}_{filename}"
            (args.output_dir / image_name).write_bytes(archive.read(member))
            row["review_id"] = f"gsr-{review_index:03d}"
            row["image"] = image_name
            graph_index = row["graph_index"]
            row["nodes"] = [
                {
                    "x": round(float(node[0]), 5),
                    "y": round(float(node[1]), 5),
                    "team": "possession"
                    if node[4] > 0.5
                    else "pressing"
                    if node[5] > 0.5
                    else "ball",
                    "role": "ball"
                    if node[11] > 0.5
                    else "goalkeeper"
                    if node[7] > 0.5
                    else "player",
                    "controls_ball": bool(node[12] > 0.5),
                }
                for node, visible in zip(features[graph_index], masks[graph_index])
                if visible
            ]

    manifest = {
        "blind_fields": ["model_prediction", "model_confidence", "weak_label", "weak_rule"],
        "situations": [
            "unstructured",
            "central_screen",
            "trap_left",
            "trap_right",
            "high_press",
            "ambiguous",
            "not_applicable",
        ],
        "items": selected,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {len(selected)} review items to {args.output_dir}; "
        f"classes={dict(Counter(row['model_prediction'] for row in selected))}"
    )


if __name__ == "__main__":
    main()
