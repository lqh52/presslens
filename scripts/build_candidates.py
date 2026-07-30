#!/usr/bin/env python3
"""Create uniformly sampled in-play candidate windows from a SoccerNet game.

SoccerNet-v2 labels do not identify build-ups directly. This script creates a
high-recall pool and excludes windows that overlap labelled stoppages. A
video-language model can then rank the pool before human tactical annotation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


STOPPAGE_LABELS = {
    "Ball out of play",
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Kick-off",
    "Offside",
    "Penalty",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Throw-in",
}

GAME_NAME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+-\s+\d{2}-\d{2}\s+(.+)\s+\d+\s+-\s+\d+\s+(.+)$"
)


@dataclass(frozen=True)
class Candidate:
    id: str
    game: str
    half: int
    start_seconds: float
    end_seconds: float
    source_video: str
    status: str = "unreviewed"
    query_score: float | None = None
    annotation: dict | None = None


def slugify(value: str) -> str:
    """Return a filesystem- and identifier-safe lower-case slug."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")


def resolve_game_slug(game_name: str, explicit_slug: str | None = None) -> str:
    """Resolve an explicit slug or infer one from SoccerNet's game name.

    The inferred slug uses the two club names and therefore preserves the
    historic ``burnley-arsenal`` prefix for the existing demo match. Use an
    explicit date-qualified slug when processing the same fixture more than
    once.
    """
    if explicit_slug is not None:
        game_slug = slugify(explicit_slug)
    else:
        match = GAME_NAME_PATTERN.match(game_name)
        source = "-".join(match.groups()) if match else game_name
        game_slug = slugify(source)
    if not game_slug:
        raise ValueError(f"Could not derive a game slug from {game_name!r}")
    return game_slug


def parse_position(annotation: dict) -> tuple[int, float]:
    half_text, _ = annotation["gameTime"].split(" - ", maxsplit=1)
    return int(half_text), int(annotation["position"]) / 1000.0


def overlaps_stoppage(
    start: float,
    end: float,
    events: list[tuple[float, str]],
    margin: float,
) -> bool:
    return any(
        label in STOPPAGE_LABELS and start - margin <= position <= end + margin
        for position, label in events
    )


def create_candidates(
    game_dir: Path,
    window_seconds: float,
    stride_seconds: float,
    margin_seconds: float,
    game_slug: str | None = None,
) -> list[Candidate]:
    labels_path = game_dir / "Labels-v2.json"
    labels = json.loads(labels_path.read_text())
    game_name = game_dir.name
    resolved_slug = resolve_game_slug(game_name, game_slug)
    events_by_half: dict[int, list[tuple[float, str]]] = {1: [], 2: []}
    for annotation in labels["annotations"]:
        half, position = parse_position(annotation)
        if half in events_by_half:
            events_by_half[half].append((position, annotation["label"]))

    candidates: list[Candidate] = []
    for half in (1, 2):
        video = game_dir / f"{half}_224p.mkv"
        if not video.exists():
            raise FileNotFoundError(f"Missing video: {video}")

        duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video),
                ],
                text=True,
            ).strip()
        )
        index = 0
        start = 8.0
        while start + window_seconds <= duration:
            end = start + window_seconds
            if not overlaps_stoppage(
                start, end, events_by_half[half], margin_seconds
            ):
                candidates.append(
                    Candidate(
                        id=f"{resolved_slug}-h{half}-{index:04d}",
                        game=game_name,
                        half=half,
                        start_seconds=start,
                        end_seconds=end,
                        source_video=str(video),
                    )
                )
                index += 1
            start += stride_seconds
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/candidates.json"),
    )
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--stoppage-margin-seconds", type=float, default=3.0)
    parser.add_argument(
        "--game-slug",
        help=(
            "Candidate ID prefix. Defaults to an inferred home-away slug; "
            "set explicitly to disambiguate repeated fixtures."
        ),
    )
    args = parser.parse_args()

    candidates = create_candidates(
        args.game_dir,
        args.window_seconds,
        args.stride_seconds,
        args.stoppage_margin_seconds,
        args.game_slug,
    )
    game_slug = resolve_game_slug(args.game_dir.name, args.game_slug)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": "SoccerNet",
        "licence_note": "Original videos are not redistributed.",
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "game_slug": game_slug,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
