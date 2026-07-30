#!/usr/bin/env python3
"""Label track evidence sequentially with a local Hugging Face vision model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from .agent_track_labeling import (
        request_fingerprint,
        validate_agent_label,
    )
except ImportError:
    from agent_track_labeling import (
        request_fingerprint,
        validate_agent_label,
    )


def extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a JSON object")


def local_prompt(target: dict[str, Any], seeds: list[dict[str, Any]]) -> str:
    references = ", ".join(
        f"{index}={seed['reviewed_label']}"
        for index, seed in enumerate(seeds, 1)
    )
    return f"""Classify only the red-boxed person in the final TARGET image.

Verified references: {references}

Compare the target uniform directly with the verified reference images.
Choose team_a or team_b only when the target visibly matches a reference carrying
that label. Choose other only for visible referee/staff/non-player evidence.
Choose unknown when unclear. Do not identify clubs and do not default to team_b.

Return JSON only:
{{"label":"team_a|team_b|other|unknown","matched_reference":0,"official_evidence":false,"reason":"one short sentence"}}

matched_reference must be the 1-based number of one matching verified reference,
or 0 for unknown. official_evidence must be true when label is other."""


def compact_to_agent(
    payload: dict[str, Any], seeds: list[dict[str, Any]]
) -> dict[str, Any]:
    label = payload.get("label")
    if label not in {"team_a", "team_b", "other", "unknown"}:
        raise ValueError("compact label must be team_a, team_b, other, or unknown")
    reference = payload.get("matched_reference", 0)
    if not isinstance(reference, int) or not 0 <= reference <= len(seeds):
        raise ValueError("matched_reference is out of range")
    official = payload.get("official_evidence")
    if not isinstance(official, bool):
        raise ValueError("official_evidence must be boolean")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    expected_prefix = label if label in {"team_a", "team_b"} else label
    reference_label = seeds[reference - 1]["reviewed_label"] if reference else None
    reference_matches = (
        reference_label is not None
        and (
            reference_label == expected_prefix
            or reference_label.startswith(expected_prefix + "_")
        )
    )
    if label in {"team_a", "team_b"} and not reference_matches:
        raise ValueError("team label requires a same-team matched_reference")
    if label == "other" and (not official or reference_label != "other"):
        raise ValueError("other requires official evidence and an other reference")
    if label == "unknown":
        return validate_agent_label(safe_unknown(reason))

    participant_type = label
    role = "other" if label == "other" else "outfield"
    return validate_agent_label(
        {
            "participant_type": participant_type,
            "role": role,
            "label": label,
            "abstain": False,
            "kit_visible": label != "other",
            "identity_visible": True,
            "official_evidence_visible": official,
            "goalkeeper_seed_available": False,
            "goalkeeper_kit_match": False,
            "matched_seed_images": [reference],
            "consistent_crop_count": 1,
            "primary_visual_cues": [reason.strip()],
            "contradicting_evidence": [],
            "reason": reason.strip(),
        },
        {seed["reviewed_label"] for seed in seeds},
    )


def safe_unknown(reason: str) -> dict[str, Any]:
    return {
        "participant_type": "unknown",
        "role": "unknown",
        "label": "unknown",
        "abstain": True,
        "kit_visible": False,
        "identity_visible": False,
        "official_evidence_visible": False,
        "goalkeeper_seed_available": False,
        "goalkeeper_kit_match": False,
        "matched_seed_images": [],
        "consistent_crop_count": 0,
        "primary_visual_cues": [],
        "contradicting_evidence": [f"Persistent validation failure: {reason}"],
        "reason": "Abstained after repeated invalid model responses.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--key", action="append")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")

    manifest_path = args.evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    tracks = manifest["tracks"]
    seeds_by_fixture = {
        fixture: [
            row
            for row in tracks
            if row["fixture_id"] == fixture and row["split"] == "seed"
        ]
        for fixture in sorted({row["fixture_id"] for row in tracks})
    }
    targets = [
        row
        for row in tracks
        if row["split"] != "seed"
        and (args.include_unreviewed or row["split"] == "evaluation")
    ]
    if args.key:
        requested = set(args.key)
        targets = [row for row in targets if row["key"] in requested]
        missing = requested - {row["key"] for row in targets}
        if missing:
            raise ValueError(f"Requested keys are not eligible: {sorted(missing)}")
    if args.limit is not None:
        targets = targets[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = args.output.with_suffix(".errors.jsonl")
    completed = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["fingerprint"]] = row

    print(f"Loading {args.model} on CUDA in bfloat16")
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    print(f"Loaded {args.model}; selected {len(targets)} tracks")

    for index, target in enumerate(targets, 1):
        seeds = seeds_by_fixture[target["fixture_id"]]
        fingerprint = request_fingerprint(
            args.model + "|compact-interleaved-prompt-v3", target, seeds, root
        )
        if fingerprint in completed and not args.force:
            print(f"[{index}/{len(targets)}] {target['key']}: cached")
            continue

        images = [
            Image.open(root / row["evidence_image"]).convert("RGB")
            for row in [*seeds, target]
        ]
        content = []
        for seed_index, (seed, image) in enumerate(zip(seeds, images[:-1]), 1):
            content.extend(
                [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"REFERENCE IMAGE {seed_index} — HUMAN-VERIFIED LABEL: "
                            f"{seed['reviewed_label']}. Treat this pairing as ground "
                            "truth."
                        ),
                    },
                ]
            )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "The next image is the unlabeled TARGET. Classify only the "
                        "person marked by the red box."
                    ),
                },
                {"type": "image", "image": images[-1]},
            ]
        )
        content.append({"type": "text", "text": local_prompt(target, seeds)})
        messages = [{"role": "user", "content": content}]
        started = time.perf_counter()
        attempt = 0
        fallback_reason = None
        try:
            for attempt in range(1, args.retries + 1):
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                input_tokens = int(inputs["input_ids"].shape[-1])
                output_ids = generated[0, input_tokens:]
                output_text = processor.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                try:
                    parsed = compact_to_agent(extract_json(output_text), seeds)
                    break
                except (ValueError, json.JSONDecodeError) as error:
                    if attempt >= args.retries:
                        fallback_reason = str(error)
                        parsed = validate_agent_label(safe_unknown(fallback_reason))
                        print(
                            f"[{index}/{len(targets)}] {target['key']}: "
                            f"abstaining after {attempt} invalid responses"
                        )
                        break
                    print(
                        f"[{index}/{len(targets)}] {target['key']}: "
                        f"invalid response on attempt {attempt}; retrying"
                    )
                    messages.extend(
                        [
                            {"role": "assistant", "content": output_text},
                            {
                                "role": "user",
                                "content": (
                                    "Your response failed validation: "
                                    f"{error}. Regenerate the complete JSON object. "
                                    "Include every required field, obey all semantic "
                                    "constraints, and output JSON only."
                                ),
                            },
                        ]
                    )
        except Exception as error:
            error_row = {
                "schema_version": 1,
                "key": target["key"],
                "model": args.model,
                "fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
                "attempts": attempt,
            }
            with errors_path.open("a") as handle:
                handle.write(json.dumps(error_row) + "\n")
            print(
                f"[{index}/{len(targets)}] {target['key']}: "
                f"{type(error).__name__}: {error}"
            )
            if args.fail_fast:
                raise
            continue
        finally:
            for image in images:
                image.close()

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
            "attempts": attempt,
            "response": parsed,
            "fallback_reason": fallback_reason,
            "usage": {
                "prompt_token_count": input_tokens,
                "candidates_token_count": int(output_ids.numel()),
                "total_token_count": input_tokens + int(output_ids.numel()),
            },
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[{index}/{len(targets)}] {target['key']}: {parsed['label']}")


if __name__ == "__main__":
    main()
