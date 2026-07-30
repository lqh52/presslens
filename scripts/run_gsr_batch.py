#!/usr/bin/env python3
"""Run SoccerNet Game State Reconstruction over extracted clips in parallel.

The input manifest is either a list of clip objects or an object containing a
``clips`` list. Each clip requires ``id`` and ``clip_path``. An optional
positive ``nframes`` value takes precedence over probing the clip with
``ffprobe``.

This runner intentionally owns only the expensive TrackLab step. It does not
delete or overwrite earlier Hydra run directories, and it skips a prior state
only after checking the state archive, TrackLab completion log, source clip,
and configured frame count.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_FRAME_COUNT = 300
DEFAULT_GPU_IDS = "1,2,3"
MANIFEST_LIST_KEYS = ("clips", "jobs", "videos", "candidates")
CLIP_PATH_KEYS = ("clip_path", "video_path", "video", "path", "output_path")
COMPLETION_MARKERS = (
    "Tracking ended, final TrackerState stats:",
    "Saved state at :",
)
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class GSRJob:
    """One manifest clip and its resolved TrackLab configuration."""

    id: str
    clip_path: Path
    nframes: int
    experiment_name: str
    state_filename: str


@dataclass(frozen=True)
class JobResult:
    """Terminal status for one job."""

    job: GSRJob
    status: str
    gpu_id: str | None = None
    state_path: Path | None = None
    log_path: Path | None = None
    message: str = ""


def safe_identifier(value: str) -> str:
    """Return a non-empty identifier safe for Hydra paths and state names."""

    cleaned = SAFE_ID_PATTERN.sub("-", value.strip()).strip("-.")
    if not cleaned:
        raise ValueError(f"Clip ID has no path-safe characters: {value!r}")
    return cleaned


def parse_gpu_ids(value: str) -> list[str]:
    """Parse a comma-separated CUDA device list without allowing duplicates."""

    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicates: {gpu_ids}")
    if any(not re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+)", item) for item in gpu_ids):
        raise ValueError(
            "GPU IDs must be numeric CUDA indices or full GPU UUIDs, "
            f"received: {gpu_ids}"
        )
    return gpu_ids


def manifest_records(payload: Any) -> list[dict[str, Any]]:
    """Extract clip records from supported manifest container shapes."""

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in MANIFEST_LIST_KEYS:
            if key in payload:
                records = payload[key]
                break
        if records is None:
            raise ValueError(
                "Manifest object must contain one of "
                + ", ".join(repr(key) for key in MANIFEST_LIST_KEYS)
            )
    else:
        raise ValueError("Manifest must be a JSON object or list")
    if not isinstance(records, list):
        raise ValueError("Manifest clip collection must be a list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every manifest clip must be a JSON object")
    return records


def parse_positive_frame_count(value: Any, *, clip_id: str) -> int | None:
    """Parse an optional positive integer frame count."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Clip {clip_id!r} has invalid nframes={value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Clip {clip_id!r} has invalid nframes={value!r}"
        ) from error
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ValueError(
            f"Clip {clip_id!r} nframes must be a positive integer, got {value!r}"
        )
    return parsed


def probe_nframes(video: Path, ffprobe: str = "ffprobe") -> int | None:
    """Return an exact or duration-derived video frame count when possible."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,duration,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ):
        return None
    streams = payload.get("streams", [])
    if not streams:
        return None
    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        raw = stream.get(key)
        if raw not in (None, "", "N/A"):
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    try:
        duration = float(stream["duration"])
    except (KeyError, TypeError, ValueError):
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw_rate = stream.get(key)
        if raw_rate in (None, "", "N/A", "0/0"):
            continue
        try:
            rate = float(Fraction(str(raw_rate)))
        except (ValueError, ZeroDivisionError):
            continue
        count = int(round(duration * rate))
        if count > 0:
            return count
    return None


def resolve_clip_path(record: dict[str, Any], path_root: Path) -> Path:
    """Resolve the extracted clip path without mistaking a full half for it."""

    raw_path = next(
        (
            record[key]
            for key in CLIP_PATH_KEYS
            if key in record and record[key] not in (None, "")
        ),
        None,
    )
    if raw_path is None:
        raise ValueError(
            f"Clip {record.get('id')!r} needs one of "
            + ", ".join(repr(key) for key in CLIP_PATH_KEYS)
            + "; source_video is deliberately not accepted because it often "
            "points to the complete match half"
        )
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = path_root / path
    return path.resolve()


def load_jobs(
    manifest: Path,
    path_root: Path,
    default_nframes: int = DEFAULT_FRAME_COUNT,
    ffprobe: str = "ffprobe",
    probe_frames: bool = True,
) -> list[GSRJob]:
    """Load, validate, and resolve jobs from a JSON manifest."""

    payload = json.loads(manifest.read_text())
    jobs: list[GSRJob] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, record in enumerate(manifest_records(payload), start=1):
        raw_id = record.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"Manifest clip {index} has no non-empty string id")
        clip_id = raw_id.strip()
        if clip_id in seen_ids:
            raise ValueError(f"Duplicate clip ID in manifest: {clip_id!r}")
        seen_ids.add(clip_id)
        resolved_id = safe_identifier(clip_id)
        if resolved_id in seen_names:
            raise ValueError(
                "Clip IDs collide after path sanitization: "
                f"{clip_id!r} -> {resolved_id!r}"
            )
        seen_names.add(resolved_id)

        clip_path = resolve_clip_path(record, path_root)
        if not clip_path.is_file():
            raise FileNotFoundError(f"Clip {clip_id!r} is missing: {clip_path}")

        nframes = parse_positive_frame_count(
            record.get("nframes"), clip_id=clip_id
        )
        if nframes is None and probe_frames:
            nframes = probe_nframes(clip_path, ffprobe)
        if nframes is None:
            nframes = default_nframes
            print(
                f"Warning: using default nframes={nframes} for {clip_id}; "
                "ffprobe did not return a frame count",
                file=sys.stderr,
            )
        jobs.append(
            GSRJob(
                id=clip_id,
                clip_path=clip_path,
                nframes=nframes,
                experiment_name=resolved_id,
                state_filename=f"{resolved_id}.pklz",
            )
        )
    if not jobs:
        raise ValueError(f"Manifest contains no clips: {manifest}")
    return jobs


def configured_value(overrides_path: Path, key: str) -> str | None:
    """Read one scalar Hydra override without requiring a YAML dependency."""

    prefix = f"- {key}="
    try:
        lines = overrides_path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip("'\"")
    return None


def valid_completed_state(state_path: Path, job: GSRJob) -> tuple[bool, str]:
    """Validate that a state belongs to this job and TrackLab completed it."""

    run_directory = state_path.parent.parent
    log_path = run_directory / "main.log"
    overrides_path = run_directory / "configs" / "overrides.yaml"
    if not state_path.is_file():
        return False, "state does not exist"
    try:
        with zipfile.ZipFile(state_path) as archive:
            if archive.testzip() is not None:
                return False, "state archive failed its CRC check"
            names = archive.namelist()
            detection_members = [
                name
                for name in names
                if name.endswith(".pkl") and not name.endswith("_image.pkl")
            ]
            image_members = [
                name for name in names if name.endswith("_image.pkl")
            ]
            if "summary.json" not in names:
                return False, "state archive has no summary.json"
            summary = json.loads(archive.read("summary.json"))
            if not isinstance(summary.get("columns"), (dict, list)):
                return False, "state summary has no columns"
            if not detection_members or not image_members:
                return False, "state archive has no detection/image tables"
            for member in detection_members + image_members:
                if archive.getinfo(member).file_size <= 0:
                    return False, f"state member is empty: {member}"
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError) as error:
        return False, f"invalid state archive: {error}"

    try:
        log_text = log_path.read_text(errors="replace")
    except OSError as error:
        return False, f"cannot read completion log: {error}"
    missing_markers = [
        marker for marker in COMPLETION_MARKERS if marker not in log_text
    ]
    if missing_markers:
        return False, "TrackLab completion markers are missing"

    configured_video = configured_value(
        overrides_path, "dataset.video_path"
    )
    if configured_video is None:
        return False, "configured source clip is missing"
    if Path(configured_video).resolve() != job.clip_path:
        return False, "configured source clip does not match"

    configured_nframes = configured_value(overrides_path, "dataset.nframes")
    if configured_nframes is None:
        return False, "configured nframes is missing"
    try:
        previous_nframes = int(configured_nframes)
    except ValueError:
        return False, "configured nframes is invalid"
    if previous_nframes != job.nframes:
        return False, "configured nframes does not match"
    return True, "valid completed state"


def find_completed_state(output_root: Path, job: GSRJob) -> Path | None:
    """Return the newest valid completed state for a job, if one exists."""

    experiment_root = output_root / job.experiment_name
    candidates = sorted(
        experiment_root.glob(f"**/states/{job.state_filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for state_path in candidates:
        valid, _ = valid_completed_state(state_path, job)
        if valid:
            return state_path
    return None


def build_tracklab_command(
    tracklab: Path,
    job: GSRJob,
    run_directory: Path | None = None,
) -> list[str]:
    """Build the established external-video TrackLab command."""

    command = [
        str(tracklab),
        "-cn",
        "soccernet",
        "dataset=video",
        "dataset.eval_set=val",
        f"dataset.video_path={job.clip_path}",
        f"dataset.nframes={job.nframes}",
        "test_tracking=true",
        "eval_tracking=false",
        f"experiment_name={job.experiment_name}",
        f"state.save_file=states/{job.state_filename}",
        "visualization=none",
    ]
    if run_directory is not None:
        command.append(f"hydra.run.dir={run_directory}")
    return command


def new_run_directory(output_root: Path, job: GSRJob) -> Path:
    """Choose a unique, readable Hydra directory without touching old runs."""

    now = datetime.now()
    leaf = now.strftime("%H-%M-%S-%f")
    run_directory = (
        output_root
        / job.experiment_name
        / now.strftime("%Y-%m-%d")
        / leaf
    )
    suffix = 1
    while run_directory.exists():
        run_directory = run_directory.with_name(f"{leaf}-{suffix}")
        suffix += 1
    return run_directory


def printable_command(command: list[str], gpu_id: str) -> str:
    """Return a shell-readable command for logs and dry runs."""

    return f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} {shlex.join(command)}"


def run_one_job(
    job: GSRJob,
    gpu_id: str,
    *,
    tracklab: Path,
    tracklab_root: Path,
    output_root: Path,
    log_directory: Path,
) -> JobResult:
    """Run one TrackLab subprocess on exactly one visible GPU."""

    run_directory = new_run_directory(output_root, job)
    command = build_tracklab_command(tracklab, job, run_directory)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_directory / f"{timestamp}-{job.experiment_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment.setdefault("HYDRA_FULL_ERROR", "1")
    started = datetime.now().isoformat(timespec="seconds")
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"started={started}\n")
        log.write(f"clip_id={job.id}\n")
        log.write(f"gpu_id={gpu_id}\n")
        log.write(f"clip_path={job.clip_path}\n")
        log.write(f"nframes={job.nframes}\n")
        log.write(f"command={printable_command(command, gpu_id)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=tracklab_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"\nreturncode={completed.returncode}\n")
    if completed.returncode:
        return JobResult(
            job=job,
            status="failed",
            gpu_id=gpu_id,
            log_path=log_path,
            message=f"TrackLab exited with code {completed.returncode}",
        )
    state_path = run_directory / "states" / job.state_filename
    valid, validation_message = valid_completed_state(state_path, job)
    if not valid:
        return JobResult(
            job=job,
            status="failed",
            gpu_id=gpu_id,
            log_path=log_path,
            message=(
                "TrackLab exited successfully but its state is invalid: "
                f"{validation_message}"
            ),
        )
    return JobResult(
        job=job,
        status="completed",
        gpu_id=gpu_id,
        state_path=state_path,
        log_path=log_path,
    )


def run_parallel(
    jobs: list[GSRJob],
    gpu_ids: list[str],
    *,
    tracklab: Path,
    tracklab_root: Path,
    output_root: Path,
    log_directory: Path,
) -> list[JobResult]:
    """Run a shared job queue with one long-lived worker per GPU."""

    pending: queue.Queue[GSRJob] = queue.Queue()
    for job in jobs:
        pending.put(job)
    results: list[JobResult] = []
    result_lock = threading.Lock()
    print_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            with print_lock:
                print(
                    f"[GPU {gpu_id}] starting {job.id} "
                    f"({job.nframes} frames)",
                    flush=True,
                )
            try:
                result = run_one_job(
                    job,
                    gpu_id,
                    tracklab=tracklab,
                    tracklab_root=tracklab_root,
                    output_root=output_root,
                    log_directory=log_directory,
                )
            except Exception as error:  # keep other GPUs/jobs progressing
                result = JobResult(
                    job=job,
                    status="failed",
                    gpu_id=gpu_id,
                    message=f"{type(error).__name__}: {error}",
                )
            with result_lock:
                results.append(result)
            with print_lock:
                detail = f": {result.message}" if result.message else ""
                print(
                    f"[GPU {gpu_id}] {result.status} {job.id}{detail}",
                    flush=True,
                )
            pending.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id,),
            name=f"gsr-gpu-{gpu_id}",
        )
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def validate_runtime(tracklab: Path, tracklab_root: Path) -> None:
    """Fail before scheduling if the TrackLab runtime is not usable."""

    if not tracklab_root.is_dir():
        raise FileNotFoundError(f"TrackLab root does not exist: {tracklab_root}")
    if not tracklab.is_file():
        raise FileNotFoundError(f"TrackLab executable does not exist: {tracklab}")
    if not os.access(tracklab, os.X_OK):
        raise PermissionError(f"TrackLab executable is not executable: {tracklab}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SoccerNet GSR on extracted clips, with one concurrent job "
            "per configured GPU."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path.cwd(),
        help="Base for relative clip paths (default: current directory)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=DEFAULT_GPU_IDS,
        help="Comma-separated CUDA indices/UUIDs (default: 1,2,3)",
    )
    parser.add_argument(
        "--tracklab-root",
        type=Path,
        default=Path("third_party/sn-gamestate"),
    )
    parser.add_argument(
        "--tracklab",
        type=Path,
        default=Path("third_party/sn-gamestate/.venv/bin/tracklab"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("third_party/sn-gamestate/outputs"),
    )
    parser.add_argument(
        "--log-directory",
        type=Path,
        default=Path("data/logs/gsr-batch"),
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable used when nframes is absent",
    )
    parser.add_argument(
        "--default-nframes",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help="Fallback when the manifest and ffprobe provide no frame count",
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Use --default-nframes whenever a clip omits nframes",
    )
    parser.add_argument(
        "--id",
        action="append",
        default=[],
        help="Run only this clip ID; repeat to select multiple clips",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print scheduling decisions without launching TrackLab",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.default_nframes <= 0:
        raise ValueError("--default-nframes must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    path_root = args.path_root.resolve()
    manifest = args.manifest.resolve()
    tracklab_root = args.tracklab_root.resolve()
    tracklab = args.tracklab.resolve()
    output_root = args.output_root.resolve()
    log_directory = args.log_directory.resolve()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    validate_runtime(tracklab, tracklab_root)
    jobs = load_jobs(
        manifest,
        path_root,
        default_nframes=args.default_nframes,
        ffprobe=args.ffprobe,
        probe_frames=not args.no_ffprobe,
    )

    if args.id:
        requested = set(args.id)
        known = {job.id for job in jobs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Requested clip IDs are not in manifest: {sorted(unknown)}")
        jobs = [job for job in jobs if job.id in requested]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    skipped: list[JobResult] = []
    pending: list[GSRJob] = []
    for job in jobs:
        state_path = find_completed_state(output_root, job)
        if state_path is not None:
            skipped.append(
                JobResult(
                    job=job,
                    status="skipped",
                    state_path=state_path,
                    message="valid completed state already exists",
                )
            )
        else:
            pending.append(job)

    print(
        f"Manifest: {len(jobs)} selected; {len(skipped)} complete; "
        f"{len(pending)} pending; GPUs={','.join(gpu_ids)}"
    )
    for result in skipped:
        print(f"[skip] {result.job.id}: {result.state_path}")

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            command = build_tracklab_command(
                tracklab,
                job,
                new_run_directory(output_root, job),
            )
            print(f"[dry-run GPU {gpu_id}] {printable_command(command, gpu_id)}")
        return 0
    if not pending:
        print("Nothing to run.")
        return 0

    results = run_parallel(
        pending,
        gpu_ids,
        tracklab=tracklab,
        tracklab_root=tracklab_root,
        output_root=output_root,
        log_directory=log_directory,
    )
    failures = [result for result in results if result.status == "failed"]
    completed = [result for result in results if result.status == "completed"]
    print(
        f"Finished: {len(completed)} completed, {len(skipped)} skipped, "
        f"{len(failures)} failed."
    )
    for result in sorted(failures, key=lambda item: item.job.id):
        log_detail = f"; log={result.log_path}" if result.log_path else ""
        print(
            f"[failed] {result.job.id}: {result.message}{log_detail}",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
