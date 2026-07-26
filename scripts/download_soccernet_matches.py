#!/usr/bin/env python3
"""Download selected SoccerNet matches without persisting the NDA password."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader


def parse_match(value: str) -> tuple[str, str]:
    try:
        split, game = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "matches must use SPLIT=championship/season/game"
        ) from error
    if split not in {"train", "valid", "test", "challenge"} or not game:
        raise argparse.ArgumentTypeError(f"invalid SoccerNet match: {value}")
    return split, game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/soccernet"))
    parser.add_argument(
        "--match",
        action="append",
        type=parse_match,
        required=True,
        help="SPLIT=championship/season/game (repeat for each match)",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--resolution",
        choices=("224p", "720p"),
        default="224p",
        help="Original SoccerNet broadcast-video resolution to download",
    )
    args = parser.parse_args()

    password = getpass("SoccerNet video password: ")
    downloader = SoccerNetDownloader(str(args.root))
    downloader.password = password

    # Labels use SoccerNet's public label credentials. Fetch them before the
    # concurrent video jobs because the upstream downloader installs a global
    # urllib authentication opener.
    for split, game in args.match:
        downloader.downloadGame(
            game=game,
            files=["Labels-v2.json"],
            spl=split,
            verbose=True,
        )

    def download_videos(item: tuple[str, str]) -> str:
        split, game = item
        worker = SoccerNetDownloader(str(args.root))
        worker.password = password
        worker.downloadGame(
            game=game,
            files=[
                f"1_{args.resolution}.mkv",
                f"2_{args.resolution}.mkv",
            ],
            spl=split,
            verbose=True,
        )
        return game

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_videos, item): item for item in args.match
        }
        for future in as_completed(futures):
            print(f"Finished {future.result()}")


if __name__ == "__main__":
    main()
