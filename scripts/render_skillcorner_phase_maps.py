#!/usr/bin/env python3
"""Render canonical SkillCorner high-, medium-, and low-block examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH, PITCH_HEIGHT, HEADER, PAD = 1050, 680, 82, 28
OUT_OF_POSSESSION = (104, 104, 255)
POSSESSION = (255, 167, 89)
BALL = (106, 240, 255)
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
    label = row["out_of_possession_phase"].replace("_", " ").upper()
    cv2.putText(
        image,
        label,
        (28, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        BALL,
        2,
        cv2.LINE_AA,
    )
    detail = (
        f"{row['possession_team']} in possession | {row['timestamp']} | "
        f"{row['duration']:.1f}s phase"
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
    cv2.arrowedLine(image, (700, 28), (800, 28), POSSESSION, 3, tipLength=0.16)
    cv2.putText(
        image,
        "Possession attack",
        (805, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        POSSESSION,
        1,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        image, (800, 58), (700, 58), OUT_OF_POSSESSION, 3, tipLength=0.16
    )
    cv2.putText(
        image,
        "Defending attack",
        (805, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        OUT_OF_POSSESSION,
        1,
        cv2.LINE_AA,
    )

    for player in row["players"]:
        centre = to_canvas(*player["xy"])
        colour = (
            POSSESSION
            if player["side"] == "possession_team"
            else OUT_OF_POSSESSION
        )
        cv2.circle(image, centre, 8, colour, -1, cv2.LINE_AA)
        cv2.circle(image, centre, 8, LINE, 2, cv2.LINE_AA)
        if not player["detected"]:
            cv2.circle(image, centre, 12, (130, 145, 135), 1, cv2.LINE_AA)

    ball = to_canvas(*row["ball_xy"])
    cv2.circle(image, ball, 7, BALL, -1, cv2.LINE_AA)
    cv2.circle(image, ball, 7, (52, 47, 5), 2, cv2.LINE_AA)

    legend_y = HEADER + PITCH_HEIGHT - 7
    for x, colour, text in (
        (40, POSSESSION, "Possession"),
        (180, OUT_OF_POSSESSION, "Out of possession"),
        (355, BALL, "Tracked ball"),
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
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index, row in enumerate(payload["maps"], 1):
        image = render(row)
        path = args.output_dir / (
            f"{index:02d}-{row['out_of_possession_phase']}.png"
        )
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
