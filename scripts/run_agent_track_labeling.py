#!/usr/bin/env python3
"""Prepare, label, reconcile, and evaluate track identities with Gemini."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent_track_labeling import (
    AGENT_LABEL_SCHEMA,
    build_evidence_manifest,
    evaluate_predictions,
    image_part,
    load_dotenv_key,
    prompt_for_track,
    reconcile_manifest,
    request_fingerprint,
    validate_agent_label,
)


def prepare(args: argparse.Namespace) -> None:
    result = build_evidence_manifest(
        args.results,
        args.labels,
        args.output,
        seed_per_label=args.seed_per_label,
        minimum_confidence=args.minimum_confidence,
        minimum_detections=args.minimum_detections,
    )
    print(f"Wrote {len(result['tracks'])} track evidence items to {args.output}")
    print(f"Splits: {result['counts']}")


def gemini_client(api_key: str):
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError(
            "Install google-genai from requirements-research.txt before labelling"
        ) from error
    return genai.Client(vertexai=True, api_key=api_key)


def label(args: argparse.Namespace) -> None:
    manifest_path = args.evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    api_key = load_dotenv_key(
        args.env,
        ("AGENT_PLATFORM_API", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    if not api_key:
        raise RuntimeError("No Agent Platform/Gemini API key was found")
    client = gemini_client(api_key)
    predictions_path = args.output
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = predictions_path.with_suffix(".errors.jsonl")
    completed = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["fingerprint"]] = row

    tracks = manifest["tracks"]
    seeds_by_fixture = {}
    for fixture in sorted({row["fixture_id"] for row in tracks}):
        seeds_by_fixture[fixture] = [
            row
            for row in tracks
            if row["fixture_id"] == fixture and row["split"] == "seed"
        ]
    targets = [
        row
        for row in tracks
        if row["split"] != "seed"
        and (args.include_unreviewed or row["split"] == "evaluation")
    ]
    if args.key:
        requested = set(args.key)
        known = {row["key"] for row in targets}
        missing = requested - known
        if missing:
            raise ValueError(f"Requested keys are not eligible targets: {sorted(missing)}")
        targets = [row for row in targets if row["key"] in requested]
    if args.limit is not None:
        targets = targets[: args.limit]
    for index, target in enumerate(targets, 1):
        seeds = seeds_by_fixture[target["fixture_id"]]
        fingerprint = request_fingerprint(args.model, target, seeds, root)
        if fingerprint in completed and not args.force:
            print(f"[{index}/{len(targets)}] {target['key']}: cached")
            continue
        contents = [image_part(root / row["evidence_image"]) for row in seeds]
        contents.append(image_part(root / target["evidence_image"]))
        contents.append({"text": prompt_for_track(target, seeds)})
        started = time.perf_counter()
        response = None
        parsed = None
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                candidate = client.models.generate_content(
                    model=args.model,
                    contents=contents,
                    config={
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        "response_json_schema": AGENT_LABEL_SCHEMA,
                    },
                )
                parsed = validate_agent_label(
                    json.loads(candidate.text),
                    {seed["reviewed_label"] for seed in seeds},
                )
                response = candidate
                break
            except Exception as error:
                last_error = error
                if attempt < args.retries:
                    delay = args.retry_delay * (2 ** (attempt - 1))
                    print(
                        f"[{index}/{len(targets)}] {target['key']}: "
                        f"{type(error).__name__}; retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
        if response is None:
            error_row = {
                "schema_version": 1,
                "key": target["key"],
                "model": args.model,
                "fingerprint": fingerprint,
                "error_type": type(last_error).__name__,
                "error": str(last_error),
                "attempts": args.retries,
            }
            with errors_path.open("a") as handle:
                handle.write(json.dumps(error_row) + "\n")
            print(
                f"[{index}/{len(targets)}] {target['key']}: failed after "
                f"{args.retries} attempts"
            )
            if args.fail_fast:
                raise RuntimeError(str(last_error)) from last_error
            continue
        row = {
            "schema_version": 1,
            "key": target["key"],
            "fixture_id": target["fixture_id"],
            "clip_id": target["clip_id"],
            "track_id": target["track_id"],
            "model": args.model,
            "fingerprint": fingerprint,
            "seed_keys": [seed["key"] for seed in seeds],
            "latency_seconds": round(time.perf_counter() - started, 4),
            "response": parsed,
            "usage": {
                name: getattr(response.usage_metadata, name, None)
                for name in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "total_token_count",
                )
            }
            if getattr(response, "usage_metadata", None)
            else None,
        }
        with predictions_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[{index}/{len(targets)}] {target['key']}: {parsed['label']}")
        if args.request_delay:
            time.sleep(args.request_delay)


def reconcile(args: argparse.Namespace) -> None:
    result = reconcile_manifest(
        args.evidence / "manifest.json", args.predictions, args.output
    )
    print(f"Wrote {len(result['clips'])} whole-video identity files to {args.output}")
    print(f"Decisions: {result['counts']}")


def evaluate(args: argparse.Namespace) -> None:
    report = evaluate_predictions(
        args.evidence / "manifest.json", args.predictions
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--results", type=Path, required=True)
    command.add_argument("--labels", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed-per-label", type=int, default=2)
    command.add_argument("--minimum-confidence", type=float, default=0.0)
    command.add_argument("--minimum-detections", type=int, default=1)
    command.set_defaults(func=prepare)

    command = subparsers.add_parser("label")
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--env", type=Path, default=Path(".env"))
    command.add_argument("--model", default="gemini-3.5-flash")
    command.add_argument("--limit", type=int)
    command.add_argument(
        "--key",
        action="append",
        help="Label only this clip:track key; may be supplied repeatedly",
    )
    command.add_argument("--include-unreviewed", action="store_true")
    command.add_argument("--force", action="store_true")
    command.add_argument("--retries", type=int, default=3)
    command.add_argument("--retry-delay", type=float, default=5.0)
    command.add_argument("--request-delay", type=float, default=1.0)
    command.add_argument("--fail-fast", action="store_true")
    command.set_defaults(func=label)

    command = subparsers.add_parser("reconcile")
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=reconcile)

    command = subparsers.add_parser("evaluate")
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=evaluate)

    args = parser.parse_args()
    if getattr(args, "seed_per_label", 1) < 1:
        parser.error("--seed-per-label must be positive")
    if not 0 <= getattr(args, "minimum_confidence", 0.0) <= 1:
        parser.error("--minimum-confidence must be between zero and one")
    if getattr(args, "minimum_detections", 1) < 1:
        parser.error("--minimum-detections must be positive")
    if getattr(args, "retries", 1) < 1:
        parser.error("--retries must be positive")
    if getattr(args, "retry_delay", 0) < 0 or getattr(args, "request_delay", 0) < 0:
        parser.error("retry and request delays cannot be negative")
    args.func(args)


if __name__ == "__main__":
    main()
