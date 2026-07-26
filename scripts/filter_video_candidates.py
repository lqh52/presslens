#!/usr/bin/env python3
"""Reject close-ups/replays and create a balanced GSR candidate shortlist.

Run this script in the sn-gamestate environment, which already contains the
Ultralytics dependency and local YOLO weights used by the reconstruction
pipeline. Retrieval-side left/right remains only a diversity hint here; final
left/right labels must come from canonical pitch graphs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def read_frame(captures: dict[Path, cv2.VideoCapture], video: Path, second: float):
    capture = captures.get(video)
    if capture is None:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video}")
        captures[video] = capture
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode {video} at {second:.3f}s")
    return image


def frame_quality(image: np.ndarray, result) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grass = (
        (hsv[:, :, 0] >= 25)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 35)
    )
    green_fraction = float(grass.mean())
    height, width = image.shape[:2]
    person_heights = []
    for box in result.boxes.cpu().numpy():
        if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.18:
            continue
        left, top, right, bottom = map(float, box.xyxy[0])
        box_height = max(0.0, bottom - top) / height
        box_area = max(0.0, right - left) * max(0.0, bottom - top)
        if 0.025 <= box_height <= 0.48 and box_area / (width * height) <= 0.14:
            person_heights.append(box_height)
    person_count = len(person_heights)
    median_person_height = (
        float(np.median(person_heights)) if person_heights else 1.0
    )
    wide_score = (
        0.48 * min(green_fraction / 0.65, 1.0)
        + 0.42 * min(person_count / 12.0, 1.0)
        + 0.10 * float(median_person_height <= 0.22)
    )
    return {
        "green_fraction": round(green_fraction, 5),
        "person_count": person_count,
        "median_person_height": round(median_person_height, 5),
        "wide_score": round(wide_score, 5),
    }


def score_candidates(rows: list[dict], model: YOLO) -> list[dict]:
    captures: dict[Path, cv2.VideoCapture] = {}
    samples: list[tuple[int, np.ndarray]] = []
    try:
        for index, row in enumerate(rows):
            source = Path(row["source_video"])
            duration = float(row["end_seconds"] - row["start_seconds"])
            for fraction in (0.25, 0.50, 0.75):
                samples.append(
                    (
                        index,
                        read_frame(
                            captures,
                            source,
                            float(row["start_seconds"]) + duration * fraction,
                        ),
                    )
                )
    finally:
        for capture in captures.values():
            capture.release()

    metrics: dict[int, list[dict]] = defaultdict(list)
    batch_size = 48
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        results = model.predict(
            [image for _, image in batch],
            classes=[0],
            conf=0.10,
            imgsz=640,
            batch=min(batch_size, len(batch)),
            verbose=False,
        )
        for (index, image), result in zip(batch, results):
            metrics[index].append(frame_quality(image, result))
        print(f"Quality-scored {min(offset + len(batch), len(samples))}/{len(samples)} frames")

    output = []
    for index, row in enumerate(rows):
        values = metrics[index]
        green = float(np.median([value["green_fraction"] for value in values]))
        people = float(np.median([value["person_count"] for value in values]))
        person_height = float(
            np.median([value["median_person_height"] for value in values])
        )
        wide_score = float(np.median([value["wide_score"] for value in values]))
        eligible = green >= 0.28 and people >= 6 and person_height <= 0.30
        retrieval_score = float(row["selected_for_query_score"])
        output.append(
            {
                **row,
                "shot_quality": {
                    "eligible_wide_shot": eligible,
                    "median_green_fraction": round(green, 5),
                    "median_person_count": round(people, 2),
                    "median_person_height": round(person_height, 5),
                    "wide_score": round(wide_score, 5),
                    "samples": values,
                },
                "shortlist_score": round(
                    retrieval_score + 0.04 * wide_score, 6
                ),
            }
        )
    return output


def select_balanced(rows: list[dict], per_query: int) -> list[dict]:
    selected = []
    used: set[str] = set()
    query_ids = sorted({row["selected_for_query_id"] for row in rows})
    for query_id in query_ids:
        pool = [
            row
            for row in rows
            if row["selected_for_query_id"] == query_id
            and row["shot_quality"]["eligible_wide_shot"]
        ]
        pool.sort(
            key=lambda row: (-float(row["shortlist_score"]), str(row["id"]))
        )
        for row in pool:
            if row["id"] in used:
                continue
            selected.append(row)
            used.add(row["id"])
            if sum(
                item["selected_for_query_id"] == query_id for item in selected
            ) >= per_query:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional manifest containing quality metrics for all ranked inputs",
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=Path("third_party/sn-gamestate/pretrained_models/yolo/yolo11m.pt"),
    )
    parser.add_argument("--per-query", type=int, default=7)
    args = parser.parse_args()

    model = YOLO(str(args.yolo))
    all_scored = []
    clips = []
    for manifest in args.manifest:
        payload = json.loads(manifest.read_text())
        scored = score_candidates(payload["candidates"], model)
        selected = select_balanced(scored, args.per_query)
        all_scored.extend(scored)
        clips.extend(selected)
        counts = {
            query_id: sum(
                row["selected_for_query_id"] == query_id for row in selected
            )
            for query_id in sorted(
                {row["selected_for_query_id"] for row in scored}
            )
        }
        print(f"{payload.get('game_slug', manifest.stem)}: selected {len(selected)} {counts}")

    output_payload = {
        "schema_version": 1,
        "selection_note": (
            "X-CLIP target diversity plus three-frame wide-shot/person filter; "
            "left/right is not a final tactical label"
        ),
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2) + "\n")
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps({"schema_version": 1, "clips": all_scored}, indent=2)
            + "\n"
        )
    print(f"Wrote {len(clips)} shortlisted clips to {args.output}")


if __name__ == "__main__":
    main()
