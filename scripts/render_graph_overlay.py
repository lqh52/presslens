"""Render thin tracking and graph overlays onto a short broadcast excerpt."""

from __future__ import annotations

import pickle
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

from convert_tracklab_state import detect_balls, recluster_player_teams, unproject


TEAM_COLOURS = {"left": (255, 190, 40), "right": (55, 90, 255)}


def render_overlay(
    state_path: Path,
    video_path: Path,
    output_path: Path,
    yolo_path: Path,
    centre_frame: int,
    label: str,
    canonical_output: Path | None = None,
    duration: float = 4.0,
    fps: int = 25,
    sequence_id: str | None = None,
    left_team_name: str = "Arsenal",
    right_team_name: str = "Burnley",
    team_labels_path: Path | None = Path("data/review/team_tracks/labels.json"),
    team_model_path: Path | None = Path(
        "models/team_identity_burnley_arsenal.npz"
    ),
    team_model_threshold: float | None = None,
) -> tuple[int, int]:
    with zipfile.ZipFile(state_path) as archive:
        detections = pickle.loads(archive.read("0.pkl"))
        images = pickle.loads(archive.read("0_image.pkl"))
    inferred, _ = recluster_player_teams(
        video_path,
        detections,
        sequence_id=sequence_id,
        labels_path=team_labels_path,
        model_path=team_model_path,
        model_threshold=team_model_threshold,
        club_to_internal={
            left_team_name.lower(): "left",
            right_team_name.lower(): "right",
        },
    )
    if inferred:
        player_rows = detections.role.isin(["player", "goalkeeper"])
        mapped = detections.loc[player_rows, "track_id"].map(
            lambda value: inferred.get(int(value), np.nan)
        )
        detections.loc[player_rows, "team"] = mapped.fillna(
            detections.loc[player_rows, "team"]
        )
    balls = detect_balls(video_path, yolo_path, len(images), 0.03)
    by_frame = {int(frame): rows for frame, rows in detections.groupby("image_id")}
    params_by_frame = {int(row.frame): row.parameters for _, row in images.iterrows()}
    span = int(duration * fps)
    start = max(0, min(centre_frame - span // 2, len(images) - span))
    end = min(len(images), start + span)

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    canonical_encoder = None
    if canonical_output is not None:
        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        canonical_encoder = subprocess.Popen(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-s", "630x408", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(canonical_output),
            ],
            stdin=subprocess.PIPE,
        )
    for frame in range(start, end):
        ok, image = capture.read()
        if not ok:
            break
        nodes = []
        rows = by_frame.get(frame)
        if rows is not None:
            for _, row in rows.iterrows():
                if row.team not in TEAM_COLOURS or row.role == "referee":
                    continue
                left, top, box_width, box_height = map(float, row.bbox_ltwh)
                centre = (int(left + box_width / 2), int(top + box_height / 2))
                pitch = row.bbox_pitch
                if not isinstance(pitch, dict):
                    continue
                point = np.array([pitch["x_bottom_middle"], pitch["y_bottom_middle"]], dtype=float)
                nodes.append((row.team, point, centre, (int(left), int(top), int(left + box_width), int(top + box_height))))

        # Within-team k-nearest edges describe the visible team structure.
        edges = set()
        for index, (team, point, centre, _) in enumerate(nodes):
            neighbours = sorted(
                (
                    (float(np.linalg.norm(point - other_point)), other_index)
                    for other_index, (other_team, other_point, _, _) in enumerate(nodes)
                    if other_index != index and other_team == team
                )
            )
            for distance, other_index in neighbours[:2]:
                if distance <= 22:
                    edges.add(tuple(sorted((index, other_index))))
        for left_index, right_index in edges:
            team = nodes[left_index][0]
            cv2.line(image, nodes[left_index][2], nodes[right_index][2], TEAM_COLOURS[team], 1, cv2.LINE_AA)

        # Cross-team links are drawn only for local pressure relations.
        for left_index, (team, point, centre, _) in enumerate(nodes):
            for right_index in range(left_index + 1, len(nodes)):
                other_team, other_point, other_centre, _ = nodes[right_index]
                if team != other_team and np.linalg.norm(point - other_point) <= 12:
                    cv2.line(image, centre, other_centre, (45, 215, 245), 1, cv2.LINE_AA)
        for team, _, _, box in nodes:
            cv2.rectangle(image, box[:2], box[2:], TEAM_COLOURS[team], 1, cv2.LINE_AA)
        if frame in balls:
            x, y = map(int, balls[frame]["image_xy"])
            cv2.circle(image, (x, y), 6, (70, 240, 245), 1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 2, (70, 240, 245), -1, cv2.LINE_AA)

        caption = label.replace("_", " ").upper()
        cv2.rectangle(image, (18, height - 49), (235, height - 18), (15, 25, 20), -1)
        cv2.putText(image, caption, (29, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 240), 1, cv2.LINE_AA)
        assert encoder.stdin is not None
        encoder.stdin.write(image.tobytes())
        if canonical_encoder is not None:
            canonical = np.full((408, 630, 3), (56, 75, 33), dtype=np.uint8)
            pitch_colour = (198, 214, 190)
            cv2.rectangle(canonical, (6, 6), (624, 402), pitch_colour, 1, cv2.LINE_AA)
            cv2.line(canonical, (315, 6), (315, 402), pitch_colour, 1, cv2.LINE_AA)
            cv2.circle(canonical, (315, 204), 55, pitch_colour, 1, cv2.LINE_AA)
            canonical_nodes = []
            for team, point, _, _ in nodes:
                px = int(np.clip((point[0] + 52.5) / 105, 0, 1) * 618 + 6)
                py = int(np.clip((point[1] + 34) / 68, 0, 1) * 396 + 6)
                canonical_nodes.append((team, point, (px, py)))
            canonical_edges = set()
            for index, (team, point, _) in enumerate(canonical_nodes):
                neighbours = sorted(
                    (
                        (float(np.linalg.norm(point - other_point)), other_index)
                        for other_index, (other_team, other_point, _) in enumerate(canonical_nodes)
                        if other_index != index and other_team == team
                    )
                )
                for distance, other_index in neighbours[:2]:
                    if distance <= 22:
                        canonical_edges.add(tuple(sorted((index, other_index))))
            for left_index, right_index in canonical_edges:
                team = canonical_nodes[left_index][0]
                cv2.line(canonical, canonical_nodes[left_index][2], canonical_nodes[right_index][2], TEAM_COLOURS[team], 1, cv2.LINE_AA)
            for team, _, point in canonical_nodes:
                cv2.circle(canonical, point, 5, TEAM_COLOURS[team], -1, cv2.LINE_AA)
            parameters = params_by_frame.get(frame)
            if frame in balls and parameters:
                ball_pitch = unproject(parameters, tuple(balls[frame]["image_xy"]))
                if ball_pitch is not None:
                    ball_point = (
                        int(np.clip((ball_pitch[0] + 52.5) / 105, 0, 1) * 618 + 6),
                        int(np.clip((ball_pitch[1] + 34) / 68, 0, 1) * 396 + 6),
                    )
                    cv2.circle(canonical, ball_point, 5, (70, 240, 245), -1, cv2.LINE_AA)
            cv2.rectangle(canonical, (14, 14), (190, 39), (11, 23, 17), -1)
            cv2.putText(canonical, label.replace("_", " ").upper(), (23, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (242, 246, 243), 1, cv2.LINE_AA)
            assert canonical_encoder.stdin is not None
            canonical_encoder.stdin.write(canonical.tobytes())
    capture.release()
    assert encoder.stdin is not None
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")
    if canonical_encoder is not None:
        assert canonical_encoder.stdin is not None
        canonical_encoder.stdin.close()
        if canonical_encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {canonical_output}")
    return start, end
