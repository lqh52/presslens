#!/usr/bin/env python3
"""Build semantic-vector and static-browser retrieval indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from text_embeddings import DEFAULT_MODEL, TextEmbedder


def retrieval_document(clip: dict) -> str:
    return ". ".join(
        [
            f"Tactical situation: {clip['title']}",
            clip["description"],
            f"Tactical class: {clip['situation'].replace('_', ' ')}",
        ]
    )


def browser_payload(
    ids: list[str],
    situations: list[str],
    documents: list[str],
    vectors: np.ndarray,
) -> dict:
    return {
        "model": "Xenova/all-MiniLM-L6-v2",
        "dimensions": int(vectors.shape[1]),
        "weights": {"cosine": 0.65, "bm25": 0.35},
        "items": [
            {
                "id": identifier,
                "situation": situation,
                "document": document,
                "vector": [
                    round(float(value), 7) for value in vector
                ],
            }
            for identifier, situation, document, vector in zip(
                ids,
                situations,
                documents,
                vectors,
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("public/demo/manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--vector-output",
        type=Path,
        default=Path("data/embeddings/fullmatch_text.npz"),
    )
    parser.add_argument(
        "--browser-output",
        type=Path,
        default=Path("public/demo/search-index.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    ids = [str(clip["id"]) for clip in manifest["clips"]]
    situations = [
        str(clip["situation"]) for clip in manifest["clips"]
    ]
    documents = [
        retrieval_document(clip) for clip in manifest["clips"]
    ]
    vectors = TextEmbedder(args.model, args.device).encode(documents)

    args.vector_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.vector_output,
        ids=np.asarray(ids),
        situations=np.asarray(situations),
        documents=np.asarray(documents),
        vectors=vectors,
    )
    args.browser_output.parent.mkdir(parents=True, exist_ok=True)
    args.browser_output.write_text(
        json.dumps(
            browser_payload(
                ids,
                situations,
                documents,
                vectors,
            ),
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"Wrote {len(ids)} normalized {vectors.shape[1]}D vectors "
        f"to {args.vector_output} and {args.browser_output}"
    )


if __name__ == "__main__":
    main()
