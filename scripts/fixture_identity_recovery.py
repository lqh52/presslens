#!/usr/bin/env python3
"""Recover unresolved fixture-local team identities with DINO, colour, and pitch gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from classify_track_identities import extract_track_features
except ImportError:
    from scripts.classify_track_identities import extract_track_features


SCHEMA_VERSION = 1
TEAM_LABELS = {"team_a", "team_b"}


def canonical_clip_id(value: str) -> str:
    return value.removesuffix("-published")


def fixture_id(clip_id: str) -> str:
    return "-".join(canonical_clip_id(clip_id).split("-")[:3])


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def stable_holdout(key: str, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return value / (2**64 - 1) < fraction


def load_evidence(path: Path, fixtures: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        row["key"]: row
        for row in payload["tracks"]
        if row["fixture_id"] in fixtures
    }


def load_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            output[row["key"]] = row
    return output


def load_track_features(
    evidence: dict[str, dict[str, Any]],
    dino_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load cached DINO and recompute colour directly from video tracks."""

    by_result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence.values():
        by_result[row["source_result"]].append(row)
    output = {}
    for source_result, rows in sorted(by_result.items()):
        result_path = Path(source_result)
        tracking = json.loads(result_path.read_text())
        clip_id = canonical_clip_id(str(tracking["clip_id"]))
        dino_path = dino_dir / result_path.name
        if not dino_path.exists() and result_path.name.endswith("-published.json"):
            dino_path = dino_dir / result_path.name.replace("-published.json", ".json")
        dino_payload = json.loads(dino_path.read_text())
        dino = {
            int(row["track_id"]): np.asarray(row["embedding"], dtype=np.float32)
            for row in dino_payload["tracks"]
        }
        colour_tracks = extract_track_features(
            Path(tracking["clip_path"]), tracking["frames"]
        )
        for row in rows:
            track_id = int(row["track_id"])
            if track_id not in dino or track_id not in colour_tracks:
                continue
            output[row["key"]] = {
                "key": row["key"],
                "clip_id": clip_id,
                "track_id": track_id,
                "dino": dino[track_id],
                "colour_raw": np.asarray(
                    colour_tracks[track_id]["features"][:-4], dtype=np.float32
                ),
                "detections": int(colour_tracks[track_id]["detections"]),
            }
    return output


def fit_colour_normalization(features: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([row["colour_raw"] for row in features.values()])
    centre = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - centre), axis=0) * 1.4826
    scale = np.where(scale > 1e-5, scale, 1.0)
    return centre, scale


def normalized_feature(
    row: dict[str, Any], colour_centre: np.ndarray, colour_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dino = row["dino"] / max(np.linalg.norm(row["dino"]), 1e-12)
    colour = (row["colour_raw"] - colour_centre) / colour_scale
    colour = colour / max(np.linalg.norm(colour), 1e-12)
    return dino, colour


def signal_distance(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    *,
    dino_weight: float,
) -> float:
    colour_weight = 1.0 - dino_weight
    return float(
        dino_weight * (1.0 - np.dot(left[0], right[0]))
        + colour_weight * (1.0 - np.dot(left[1], right[1]))
    )


def medoid(
    keys: list[str],
    vectors: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    dino_weight: float,
) -> str:
    return min(
        keys,
        key=lambda key: sum(
            signal_distance(
                vectors[key], vectors[other], dino_weight=dino_weight
            )
            for other in keys
        ),
    )


def projection_metrics(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    length = float(payload.get("pitch_length_m", 105.0))
    width = float(payload.get("pitch_width_m", 68.0))
    counts: dict[int, list[bool]] = defaultdict(list)
    for frame in payload.get("frames", []):
        for row in frame.get("objects", []):
            x, y = float(row["x"]), float(row["y"])
            inside = abs(x) <= length / 2 + 2.0 and abs(y) <= width / 2 + 2.0
            counts[int(row["track_id"])].append(inside)
    return {
        track_id: {
            "projection_frames": len(values),
            "on_pitch_ratio": round(sum(values) / len(values), 6),
        }
        for track_id, values in counts.items()
        if values
    }


def classify(
    vector: tuple[np.ndarray, np.ndarray],
    prototypes: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    dino_weight: float,
) -> dict[str, Any]:
    component_distances = {
        "dino": {
            label: float(1.0 - np.dot(vector[0], prototype[0]))
            for label, prototype in prototypes.items()
        },
        "colour": {
            label: float(1.0 - np.dot(vector[1], prototype[1]))
            for label, prototype in prototypes.items()
        },
    }
    component_labels = {
        signal: min(values, key=values.get)
        for signal, values in component_distances.items()
    }
    distances = {
        label: signal_distance(vector, prototype, dino_weight=dino_weight)
        for label, prototype in prototypes.items()
    }
    ordered = sorted(distances, key=distances.get)
    nearest, second = ordered
    margin = (distances[second] - distances[nearest]) / max(
        distances[second], 1e-8
    )
    return {
        "label": nearest,
        "distances": {key: round(value, 6) for key, value in distances.items()},
        "margin": round(float(margin), 6),
        "component_labels": component_labels,
        "component_distances": {
            signal: {key: round(value, 6) for key, value in values.items()}
            for signal, values in component_distances.items()
        },
        "signals_agree": len(set(component_labels.values())) == 1,
    }


def nearest_other(
    vector: tuple[np.ndarray, np.ndarray],
    other_vectors: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | None]:
    if not other_vectors:
        return {"dino": None, "colour": None}
    return {
        "dino": round(
            min(float(1.0 - np.dot(vector[0], other[0])) for other in other_vectors),
            6,
        ),
        "colour": round(
            min(float(1.0 - np.dot(vector[1], other[1])) for other in other_vectors),
            6,
        ),
    }


def other_veto(
    candidate: dict[str, Any],
    other_distances: dict[str, float | None],
) -> bool:
    label = candidate["label"]
    return any(
        other_distances[signal] is not None
        and float(other_distances[signal])
        <= float(candidate["component_distances"][signal][label])
        for signal in ("dino", "colour")
    )


def train_fixture_models(
    evidence_path: Path,
    prediction_paths: list[Path],
    dino_dir: Path,
    projection_dir: Path,
    output_dir: Path,
    fixtures: set[str],
    *,
    dino_weight: float,
    benchmark_holdout: float,
    minimum_anchors: int,
) -> dict[str, Any]:
    evidence = load_evidence(evidence_path, fixtures)
    predictions = load_predictions(prediction_paths)
    features = load_track_features(evidence, dino_dir)
    models = {}
    for fixture in sorted(fixtures):
        fixture_rows = {
            key: row for key, row in evidence.items() if row["fixture_id"] == fixture
        }
        fixture_features = {
            key: row for key, row in features.items() if key in fixture_rows
        }
        colour_centre, colour_scale = fit_colour_normalization(fixture_features)
        vectors = {
            key: normalized_feature(row, colour_centre, colour_scale)
            for key, row in fixture_features.items()
        }
        evaluation_keys = {
            key
            for key, row in fixture_rows.items()
            if row.get("split") == "evaluation" and row.get("reviewed_label")
        }
        holdout_keys = {
            key for key in evaluation_keys if stable_holdout(key, benchmark_holdout)
        }
        seeds = {
            key: row["reviewed_label"]
            for key, row in fixture_rows.items()
            if row.get("split") == "seed"
            and row.get("reviewed_label") in TEAM_LABELS
            and key in vectors
        }
        other_seeds = {
            key
            for key, row in fixture_rows.items()
            if row.get("split") == "seed"
            and row.get("reviewed_label") == "other"
            and key in vectors
        }
        anchors: dict[str, list[str]] = {"team_a": [], "team_b": []}
        other_keys = set(other_seeds)
        for key, prediction in predictions.items():
            if key not in fixture_rows or key not in vectors or key in holdout_keys:
                continue
            response = prediction.get("response", {})
            label = response.get("label")
            if (
                label in TEAM_LABELS
                and response.get("abstain") is not True
                and response.get("kit_visible", True) is True
                and features[key]["detections"] >= 3
            ):
                anchors[label].append(key)
            elif (
                label == "other"
                and response.get("abstain") is not True
                and response.get("identity_visible", True) is True
                and features[key]["detections"] >= 3
            ):
                other_keys.add(key)
        enabled = all(len(anchors[label]) >= minimum_anchors for label in TEAM_LABELS)
        prototype_keys = {}
        prototypes = {}
        if enabled:
            prototype_keys = {
                label: medoid(keys, vectors, dino_weight=dino_weight)
                for label, keys in anchors.items()
            }
            prototypes = {
                label: vectors[key] for label, key in prototype_keys.items()
            }
        seed_results = {}
        other_vectors = [vectors[key] for key in sorted(other_keys)]
        if enabled:
            for key, expected in seeds.items():
                candidate = classify(
                    vectors[key], prototypes, dino_weight=dino_weight
                )
                other_distances = nearest_other(vectors[key], other_vectors)
                seed_results[key] = {
                    "expected": expected,
                    **candidate,
                    "nearest_other": other_distances,
                    "other_veto": other_veto(candidate, other_distances),
                }
            enabled = bool(seed_results) and all(
                row["label"] == row["expected"]
                and row["signals_agree"]
                for row in seed_results.values()
            )
        other_seed_results = {}
        if prototypes:
            for key in sorted(other_seeds):
                candidate = classify(
                    vectors[key], prototypes, dino_weight=dino_weight
                )
                other_distances = nearest_other(vectors[key], other_vectors)
                rejected = other_veto(candidate, other_distances) or not candidate[
                    "signals_agree"
                ]
                other_seed_results[key] = {
                    **candidate,
                    "nearest_other": other_distances,
                    "rejected": rejected,
                }
            enabled = enabled and bool(other_seed_results) and all(
                row["rejected"] for row in other_seed_results.values()
            )
        radii = {}
        margin_threshold = 0.05
        if enabled:
            for label, keys in anchors.items():
                distances = [
                    signal_distance(
                        vectors[key], prototypes[label], dino_weight=dino_weight
                    )
                    for key in keys
                ]
                seed_distances = [
                    signal_distance(
                        vectors[key], prototypes[label], dino_weight=dino_weight
                    )
                    for key, expected in seeds.items()
                    if expected == label
                ]
                radii[label] = round(
                    max(
                        float(np.quantile(distances, 0.9)) * 1.15,
                        max(seed_distances, default=0.0) * 1.05,
                    ),
                    6,
                )
            seed_margins = [row["margin"] for row in seed_results.values()]
            margin_threshold = round(max(0.05, min(seed_margins) * 0.5), 6)
        model = {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture,
            "enabled": enabled,
            "method": "fixture_dino_colour_medoid_other_veto_v2",
            "weights": {"dino": dino_weight, "colour": 1.0 - dino_weight},
            "anchors": anchors,
            "prototype_keys": prototype_keys,
            "radii": radii,
            "margin_threshold": margin_threshold,
            "human_seed_validation": seed_results,
            "human_other_validation": other_seed_results,
            "other_keys": sorted(other_keys),
            "benchmark_holdout_keys": sorted(holdout_keys),
            "colour_normalization": {
                "centre": colour_centre.round(7).tolist(),
                "scale": colour_scale.round(7).tolist(),
            },
            "prototypes": {
                label: {
                    "dino": prototypes[label][0].round(7).tolist(),
                    "colour": prototypes[label][1].round(7).tolist(),
                }
                for label in prototypes
            },
            "other_vectors": [
                {
                    "dino": vector[0].round(7).tolist(),
                    "colour": vector[1].round(7).tolist(),
                }
                for vector in other_vectors
            ],
            "projection_directory": str(projection_dir.resolve()),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{fixture}.json").write_text(json.dumps(model) + "\n")
        models[fixture] = model
    summary = {
        "schema_version": SCHEMA_VERSION,
        "fixtures": {
            key: {
                "enabled": value["enabled"],
                "anchor_counts": {
                    label: len(keys) for label, keys in value["anchors"].items()
                },
                "seed_validation": value["human_seed_validation"],
                "other_validation": value["human_other_validation"],
                "holdout_tracks": len(value["benchmark_holdout_keys"]),
            }
            for key, value in models.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def recover_fixture(
    model_path: Path,
    evidence_path: Path,
    dino_dir: Path,
    projection_dir: Path,
    output_path: Path,
    *,
    benchmark: bool,
    prediction_paths: list[Path] | None,
    minimum_projection_frames: int,
    minimum_on_pitch_ratio: float,
) -> dict[str, Any]:
    model = json.loads(model_path.read_text())
    fixture = model["fixture_id"]
    evidence = load_evidence(evidence_path, {fixture})
    features = load_track_features(evidence, dino_dir)
    centre = np.asarray(model["colour_normalization"]["centre"], dtype=np.float32)
    scale = np.asarray(model["colour_normalization"]["scale"], dtype=np.float32)
    vectors = {
        key: normalized_feature(row, centre, scale)
        for key, row in features.items()
    }
    prototypes = {
        label: (
            np.asarray(row["dino"], dtype=np.float32),
            np.asarray(row["colour"], dtype=np.float32),
        )
        for label, row in model["prototypes"].items()
    }
    other_vectors = [
        (
            np.asarray(row["dino"], dtype=np.float32),
            np.asarray(row["colour"], dtype=np.float32),
        )
        for row in model.get("other_vectors", [])
    ]
    if benchmark:
        target_keys = set(model["benchmark_holdout_keys"])
    else:
        predictions = load_predictions(prediction_paths or [])
        target_keys = {
            key
            for key, row in evidence.items()
            if row.get("reviewed_label") is None and key not in predictions
        } | {
            key
            for key, row in predictions.items()
            if key in evidence
            and (
                row.get("response", {}).get("label") == "unknown"
                or row.get("response", {}).get("abstain") is True
            )
        }
    projection_by_clip = {}
    for row in evidence.values():
        clip_id = row["clip_id"]
        if clip_id not in projection_by_clip:
            path = projection_dir / f"{clip_id}.json"
            if not path.exists():
                path = projection_dir / f"{clip_id}-published.json"
            projection_by_clip[clip_id] = projection_metrics(path)
    decisions = {}
    for key in sorted(target_keys):
        if key not in vectors:
            decisions[key] = {"status": "needs_review", "reason": "missing_features"}
            continue
        candidate = classify(
            vectors[key], prototypes, dino_weight=float(model["weights"]["dino"])
        )
        other_distances = nearest_other(vectors[key], other_vectors)
        vetoed_as_other = other_veto(candidate, other_distances)
        row = evidence[key]
        projection = projection_by_clip.get(row["clip_id"], {}).get(
            int(row["track_id"]), {}
        )
        accepted = (
            model["enabled"]
            and candidate["distances"][candidate["label"]]
            <= float(model["radii"][candidate["label"]])
            and candidate["margin"] >= float(model["margin_threshold"])
            and candidate["signals_agree"]
            and not vetoed_as_other
            and int(projection.get("projection_frames", 0))
            >= minimum_projection_frames
            and float(projection.get("on_pitch_ratio", 0.0))
            >= minimum_on_pitch_ratio
        )
        decisions[key] = {
            "label": candidate["label"] if accepted else "unknown",
            "candidate_label": candidate["label"],
            "status": "recovered_proposal" if accepted else "needs_review",
            "appearance": candidate,
            "nearest_other": other_distances,
            "canonical": projection,
            "gates": {
                "fixture_validated": bool(model["enabled"]),
                "within_radius": candidate["distances"][candidate["label"]]
                <= float(model["radii"].get(candidate["label"], 0.0)),
                "margin": candidate["margin"] >= float(model["margin_threshold"]),
                "signals_agree": candidate["signals_agree"],
                "other_veto": not vetoed_as_other,
                "projection_frames": int(projection.get("projection_frames", 0))
                >= minimum_projection_frames,
                "on_pitch": float(projection.get("on_pitch_ratio", 0.0))
                >= minimum_on_pitch_ratio,
                "off_pitch_veto": float(projection.get("on_pitch_ratio", 0.0))
                >= minimum_on_pitch_ratio,
            },
        }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture,
        "model": str(model_path),
        "targets": len(target_keys),
        "recovered": sum(
            row.get("status") == "recovered_proposal" for row in decisions.values()
        ),
        "decisions": decisions,
    }
    if benchmark:
        scored = [
            (key, row)
            for key, row in decisions.items()
            if evidence[key].get("reviewed_label")
        ]
        recovered = [
            (key, row)
            for key, row in scored
            if row.get("status") == "recovered_proposal"
        ]
        correct = sum(
            row["label"] == evidence[key]["reviewed_label"]
            for key, row in recovered
        )
        recovered_team = [
            (key, row)
            for key, row in recovered
            if evidence[key]["reviewed_label"] in TEAM_LABELS
        ]
        correct_team = sum(
            row["label"] == evidence[key]["reviewed_label"]
            for key, row in recovered_team
        )
        report["benchmark"] = {
            "scored_tracks": len(scored),
            "recovered_tracks": len(recovered),
            "coverage": round(len(recovered) / len(scored), 6) if scored else 0.0,
            "correct": correct,
            "precision": round(correct / len(recovered), 6) if recovered else 0.0,
            "remaining_unknown": len(scored) - len(recovered),
            "team_recovered_tracks": len(recovered_team),
            "team_precision": round(correct_team / len(recovered_team), 6)
            if recovered_team
            else 0.0,
            "non_team_false_recoveries": sum(
                evidence[key]["reviewed_label"] not in TEAM_LABELS
                for key, _row in recovered
            ),
            "confusion": dict(
                Counter(
                    f"{evidence[key]['reviewed_label']}->{row['label']}"
                    for key, row in recovered
                )
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--evidence", type=Path, required=True)
    train.add_argument("--predictions", type=Path, action="append", required=True)
    train.add_argument("--dino-dir", type=Path, required=True)
    train.add_argument("--projection-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--fixture", action="append", required=True)
    train.add_argument("--dino-weight", type=float, default=0.8)
    train.add_argument("--benchmark-holdout", type=float, default=0.25)
    train.add_argument("--minimum-anchors", type=int, default=3)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--model", type=Path, required=True)
    recover.add_argument("--evidence", type=Path, required=True)
    recover.add_argument("--dino-dir", type=Path, required=True)
    recover.add_argument("--projection-dir", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--benchmark", action="store_true")
    recover.add_argument(
        "--predictions",
        type=Path,
        action="append",
        help="Current Gemini results; unknown/abstained tracks become recovery targets",
    )
    recover.add_argument("--minimum-projection-frames", type=int, default=3)
    recover.add_argument("--minimum-on-pitch-ratio", type=float, default=0.7)
    args = parser.parse_args()

    if args.command == "train":
        summary = train_fixture_models(
            args.evidence,
            args.predictions,
            args.dino_dir,
            args.projection_dir,
            args.output_dir,
            set(args.fixture),
            dino_weight=args.dino_weight,
            benchmark_holdout=args.benchmark_holdout,
            minimum_anchors=args.minimum_anchors,
        )
        print(json.dumps(summary, indent=2))
    else:
        report = recover_fixture(
            args.model,
            args.evidence,
            args.dino_dir,
            args.projection_dir,
            args.output,
            benchmark=args.benchmark,
            prediction_paths=args.predictions,
            minimum_projection_frames=args.minimum_projection_frames,
            minimum_on_pitch_ratio=args.minimum_on_pitch_ratio,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
