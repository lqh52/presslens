#!/usr/bin/env python3
"""Render StatsBomb weak pressure maps in the repository canonical-pitch style."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, PITCH_HEIGHT, HEADER, PAD = 1050, 680, 80, 28
PRESSING = (104, 104, 255)
POSSESSION = (255, 167, 89)
ACCENT = (106, 240, 255)
LINE = (225, 238, 228)


def to_canvas(x: float, y: float) -> tuple[int, int]:
    return (
        int(PAD + x / 105 * (WIDTH - 2 * PAD)),
        int(HEADER + PAD + y / 68 * (PITCH_HEIGHT - 2 * PAD)),
    )


def background() -> np.ndarray:
    image = np.full((HEADER + PITCH_HEIGHT, WIDTH, 3), (13, 27, 20), np.uint8)
    image[HEADER:] = (42, 74, 42)
    top, bottom = HEADER + PAD, HEADER + PITCH_HEIGHT - PAD
    cv2.rectangle(image, (PAD, top), (WIDTH - PAD, bottom), LINE, 3)
    cv2.line(image, (WIDTH // 2, top), (WIDTH // 2, bottom), LINE, 3)
    cv2.circle(image, (WIDTH // 2, HEADER + PITCH_HEIGHT // 2), 92, LINE, 3)
    penalty_width = int(40.32 / 68 * (PITCH_HEIGHT - 2 * PAD))
    penalty_depth = int(16.5 / 105 * (WIDTH - 2 * PAD))
    goal_width = int(18.32 / 68 * (PITCH_HEIGHT - 2 * PAD))
    goal_depth = int(5.5 / 105 * (WIDTH - 2 * PAD))
    centre_y = HEADER + PITCH_HEIGHT // 2
    for edge, direction in ((PAD, 1), (WIDTH - PAD, -1)):
        cv2.rectangle(
            image,
            (
                min(edge, edge + direction * penalty_depth),
                centre_y - penalty_width // 2,
            ),
            (
                max(edge, edge + direction * penalty_depth),
                centre_y + penalty_width // 2,
            ),
            LINE,
            3,
        )
        cv2.rectangle(
            image,
            (min(edge, edge + direction * goal_depth), centre_y - goal_width // 2),
            (max(edge, edge + direction * goal_depth), centre_y + goal_width // 2),
            LINE,
            3,
        )
    return image


def render(row: dict) -> np.ndarray:
    image = background()
    label = row["weak_label"].replace("_", " ").upper()
    cv2.putText(
        image,
        label,
        (28, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        ACCENT,
        2,
        cv2.LINE_AA,
    )
    detail = (
        f"{row['pressing_team']} pressing | {row['timestamp']} | "
        f"{len(row['players'])} visible players"
    )
    cv2.putText(
        image,
        detail,
        (28, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 213, 202),
        1,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(image, (790, 28), (690, 28), PRESSING, 3, tipLength=0.16)
    cv2.putText(
        image,
        "Pressing attack",
        (795, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        PRESSING,
        1,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(image, (690, 58), (790, 58), POSSESSION, 3, tipLength=0.16)
    cv2.putText(
        image,
        "Possession attack",
        (795, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        POSSESSION,
        1,
        cv2.LINE_AA,
    )

    polygon = np.asarray(
        [to_canvas(*point) for point in row.get("visible_polygon", [])],
        dtype=np.int32,
    )
    if len(polygon) >= 3:
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], (72, 103, 73))
        image = cv2.addWeighted(overlay, 0.22, image, 0.78, 0)
        cv2.polylines(image, [polygon], True, (125, 164, 133), 2, cv2.LINE_AA)

    for side, colour in (
        ("possession_team", POSSESSION),
        ("pressing_team", PRESSING),
    ):
        points = [
            to_canvas(*player["xy"])
            for player in row["players"]
            if player["side"] == side
        ]
        for point in points:
            distances = sorted(
                (
                    (np.hypot(point[0] - other[0], point[1] - other[1]), other)
                    for other in points
                    if other != point
                ),
                key=lambda item: item[0],
            )
            for _, other in distances[:2]:
                cv2.line(image, point, other, tuple(int(c * 0.55) for c in colour), 1)

    for player in row["players"]:
        centre = to_canvas(*player["xy"])
        colour = PRESSING if player["side"] == "pressing_team" else POSSESSION
        radius = 11 if player["keeper"] else 8
        cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
        cv2.circle(image, centre, radius, LINE, 2, cv2.LINE_AA)
        if player["actor"]:
            cv2.circle(image, centre, radius + 7, LINE, 3, cv2.LINE_AA)

    if row.get("ball_xy"):
        possessor = to_canvas(*row["ball_xy"])
        cv2.circle(image, possessor, 15, ACCENT, 3, cv2.LINE_AA)
        actor_xy = next(
            (
                to_canvas(*player["xy"])
                for player in row["players"]
                if player["actor"]
            ),
            None,
        )
        if actor_xy:
            dx = possessor[0] - actor_xy[0]
            dy = possessor[1] - actor_xy[1]
            length = max(math.hypot(dx, dy), 1.0)
            ball = (
                int(possessor[0] + 15 * dx / length),
                int(possessor[1] + 15 * dy / length),
            )
        else:
            ball = (possessor[0] + 15, possessor[1])
        cv2.line(image, possessor, ball, ACCENT, 2, cv2.LINE_AA)
        cv2.circle(image, ball, 7, ACCENT, -1, cv2.LINE_AA)
        cv2.circle(image, ball, 7, (52, 47, 5), 2, cv2.LINE_AA)

    legend_y = HEADER + PITCH_HEIGHT - 7
    for x, colour, text in (
        (40, POSSESSION, "Possession"),
        (180, PRESSING, "Pressing"),
    ):
        cv2.circle(image, (x, legend_y - 5), 6, colour, -1)
        cv2.putText(
            image,
            text,
            (x + 11, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            LINE,
            1,
            cv2.LINE_AA,
        )
    cv2.circle(image, (310, legend_y - 5), 12, LINE, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "Pressing actor",
        (327, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        LINE,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(image, (440, legend_y - 5), 10, ACCENT, 2, cv2.LINE_AA)
    cv2.circle(image, (440, legend_y - 5), 4, POSSESSION, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "Ball proxy (possession)",
        (451, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        LINE,
        1,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    selected = []
    for label in ("counterpress", "high_press_candidate"):
        candidates = [row for row in payload["maps"] if row["weak_label"] == label]
        candidates.sort(key=lambda row: (-len(row["players"]), row["timestamp"]))
        selected.extend(candidates[: args.per_label])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index, row in enumerate(selected, 1):
        image = render(row)
        path = args.output_dir / f"{index:02d}-{row['weak_label']}.png"
        cv2.imwrite(str(path), image)
        images.append(image)
    cells = [cv2.resize(image, (630, 456)) for image in images]
    if len(cells) % 2:
        cells.append(np.full_like(cells[0], (13, 27, 20)))
    contact = np.vstack(
        [np.hstack(cells[index : index + 2]) for index in range(0, len(cells), 2)]
    )
    contact_path = args.output_dir / "contact-sheet.png"
    cv2.imwrite(str(contact_path), contact)
    print(f"Wrote {len(images)} maps and {contact_path}")


if __name__ == "__main__":
    main()
