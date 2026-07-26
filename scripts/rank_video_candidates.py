#!/usr/bin/env python3
"""Rank SoccerNet candidate windows with an X-CLIP video-text baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, XCLIPModel


@dataclass(frozen=True)
class QuerySpec:
    id: str
    text: str


DEFAULT_QUERY_SPECS = [
    QuerySpec(
        "high_press",
        "a football team building up in its defensive third while the opponents apply a coordinated high press",
    ),
    QuerySpec(
        "left_touchline_trap",
        "a coordinated football press trapping the team in possession against the left touchline",
    ),
    QuerySpec(
        "right_touchline_trap",
        "a coordinated football press trapping the team in possession against the right touchline",
    ),
]
DEFAULT_QUERIES = [query.text for query in DEFAULT_QUERY_SPECS]


def parse_query_specs(values: list[str] | None) -> list[QuerySpec]:
    """Parse optional ``ID=prompt`` queries while accepting legacy bare prompts."""
    if not values:
        return list(DEFAULT_QUERY_SPECS)
    specs = []
    for index, value in enumerate(values, start=1):
        identifier = f"query_{index}"
        text = value
        if "=" in value:
            possible_id, possible_text = value.split("=", 1)
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", possible_id):
                identifier = possible_id.lower().replace("-", "_")
                text = possible_text
        text = text.strip()
        if not text:
            raise ValueError(f"Query {identifier!r} has no prompt text")
        specs.append(QuerySpec(identifier, text))
    identifiers = [query.id for query in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Query IDs must be unique: {identifiers}")
    return specs


def make_ranked_item(
    item: dict,
    scores: list[float],
    query_specs: list[QuerySpec],
) -> dict:
    """Attach maximum and per-query similarity scores to one candidate."""
    if len(scores) != len(query_specs):
        raise ValueError("Score and query counts differ")
    query_index = int(np.asarray(scores).argmax())
    return {
        **item,
        "query_score": round(float(scores[query_index]), 6),
        "matched_query": query_specs[query_index].text,
        "matched_query_id": query_specs[query_index].id,
        "query_scores": {
            query.id: round(float(score), 6)
            for query, score in zip(query_specs, scores)
        },
    }


def select_balanced(
    ranked: list[dict],
    query_specs: list[QuerySpec],
    top_k: int,
) -> list[dict]:
    """Select a unique, near-equal quota of candidates for every query."""
    limit = min(max(top_k, 0), len(ranked))
    if not limit or not query_specs:
        return []
    base, remainder = divmod(limit, len(query_specs))
    targets = {
        query.id: base + (index < remainder)
        for index, query in enumerate(query_specs)
    }
    pools = {
        query.id: sorted(
            ranked,
            key=lambda item: (
                -float(item["query_scores"][query.id]),
                str(item["id"]),
            ),
        )
        for query in query_specs
    }
    positions = {query.id: 0 for query in query_specs}
    counts = {query.id: 0 for query in query_specs}
    selected_ids: set[str] = set()
    selected: list[dict] = []
    while len(selected) < limit:
        made_progress = False
        for query in query_specs:
            query_id = query.id
            if counts[query_id] >= targets[query_id]:
                continue
            pool = pools[query_id]
            while (
                positions[query_id] < len(pool)
                and pool[positions[query_id]]["id"] in selected_ids
            ):
                positions[query_id] += 1
            if positions[query_id] >= len(pool):
                continue
            item = pool[positions[query_id]]
            positions[query_id] += 1
            selected_ids.add(item["id"])
            selected.append(
                {
                    **item,
                    "selected_for_query_id": query_id,
                    "selected_for_query": query.text,
                    "selected_for_query_score": item["query_scores"][query_id],
                }
            )
            counts[query_id] += 1
            made_progress = True
        if not made_progress:
            break
    return selected


def pooled_features(output: torch.Tensor | object) -> torch.Tensor:
    """Handle both legacy tensor and current Transformers model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is None:
        raise TypeError(f"Model output has no pooled representation: {type(output)}")
    return pooler_output


def decode_frames(
    video: Path,
    start_seconds: float,
    duration: float,
    frames: int,
    size: int = 224,
) -> list[np.ndarray]:
    fps = frames / duration
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={fps:.8f},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    array = np.frombuffer(raw, dtype=np.uint8)
    expected = frames * size * size * 3
    if array.size != expected:
        raise RuntimeError(
            f"Decoded {array.size} values from {video}, expected {expected}"
        )
    return list(array.reshape(frames, size, size, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/candidates.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/ranked_candidates.json"),
    )
    parser.add_argument("--model", default="microsoft/xclip-base-patch32")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument(
        "--query",
        action="append",
        help=(
            "Optional prompt, either plain text or ID=prompt. Repeat for "
            "multiple tactical queries."
        ),
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Select a near-equal number of unique top candidates per query",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    candidates = payload["candidates"][: args.limit]
    query_specs = parse_query_specs(args.query)
    queries = [query.text for query in query_specs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(args.model)
    model = XCLIPModel.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    text_inputs = processor(text=queries, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = pooled_features(model.get_text_features(**text_inputs))
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    ranked: list[dict] = []
    duration = float(payload["window_seconds"])
    for offset in range(0, len(candidates), args.batch_size):
        batch = candidates[offset : offset + args.batch_size]
        videos = [
            decode_frames(
                Path(item["source_video"]),
                item["start_seconds"],
                duration,
                args.frames,
            )
            for item in batch
        ]
        inputs = processor(videos=videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            video_features = pooled_features(
                model.get_video_features(pixel_values=pixel_values)
            )
            video_features = torch.nn.functional.normalize(video_features, dim=-1)
            similarities = video_features @ text_features.T
        for item, query_scores in zip(batch, similarities.cpu().tolist()):
            ranked.append(make_ranked_item(item, query_scores, query_specs))
        print(f"Ranked {min(offset + len(batch), len(candidates))}/{len(candidates)}")

    ranked.sort(key=lambda item: item["query_score"], reverse=True)
    payload["model"] = args.model
    payload["queries"] = queries
    payload["query_specs"] = [
        {"id": query.id, "text": query.text} for query in query_specs
    ]
    payload["selection_mode"] = "balanced" if args.balanced else "global"
    payload["candidates"] = (
        select_balanced(ranked, query_specs, args.top_k)
        if args.balanced
        else ranked[: args.top_k]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote top {len(payload['candidates'])} candidates to {args.output}")


if __name__ == "__main__":
    main()
