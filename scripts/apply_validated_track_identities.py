#!/usr/bin/env python3
"""Apply human, Gemini, and validated recovery labels to review identities."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("response"):
                rows[row["key"]] = row["response"]
    return rows


def normalized(label: str) -> tuple[str, bool]:
    if label == "team_a_goalkeeper":
        return "team_a", True
    if label == "team_b_goalkeeper":
        return "team_b", True
    return label, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reviewed = json.loads(args.reviewed_labels.read_text()).get("labels", {})
    predictions = read_jsonl(args.predictions)
    recovered = {}
    for path in args.recovery.glob("*.json"):
        report = json.loads(path.read_text())
        recovered.update(
            {
                key: row
                for key, row in report.get("decisions", {}).items()
                if row.get("status") == "recovered_proposal"
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for source in sorted(args.identities.glob("*.json")):
        if source.name == "match-anchors.json":
            continue
        payload = json.loads(source.read_text())
        clip_id = payload["clip_id"]
        counts = Counter()
        for track in payload["tracks"]:
            key = f"{clip_id}:{track['track_id']}"
            if key in reviewed:
                raw_label, source_name = reviewed[key]["label"], "human_review"
            elif (
                key in predictions
                and not predictions[key].get("abstain")
                and predictions[key]["label"] != "unknown"
            ):
                response = predictions[key]
                raw_label, source_name = response["label"], "gemini"
            elif key in recovered:
                raw_label, source_name = recovered[key]["label"], "fixture_recovery"
            else:
                raw_label, source_name = "unknown", "needs_review"
            label, goalkeeper = normalized(raw_label)
            track["label"] = label
            track["goalkeeper"] = goalkeeper
            track["identity_source"] = source_name
            counts[label] += 1
            totals[source_name] += 1
        payload["counts"] = dict(counts)
        payload["goalkeepers"] = sum(row["goalkeeper"] for row in payload["tracks"])
        payload["configuration"]["identity_precedence"] = (
            "human_review > gemini > validated_fixture_recovery > unknown"
        )
        (args.output / source.name).write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({"clips": len(list(args.output.glob("*.json"))), "sources": totals}, default=dict))


if __name__ == "__main__":
    main()
