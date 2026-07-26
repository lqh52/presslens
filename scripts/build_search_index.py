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
    *,
    browser_model: str,
    pooling: str,
    query_prefix: str,
    cosine_weight: float,
) -> dict:
    return {
        "model": browser_model,
        "dimensions": int(vectors.shape[1]),
        "pooling": pooling,
        "queryPrefix": query_prefix,
        "weights": {"cosine": cosine_weight, "bm25": 1.0 - cosine_weight},
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
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument(
        "--browser-model",
        default="Xenova/all-MiniLM-L6-v2",
    )
    parser.add_argument("--cosine-weight", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.cosine_weight <= 1:
        parser.error("--cosine-weight must be between 0 and 1")

    manifest = json.loads(args.manifest.read_text())
    ids = [str(clip["id"]) for clip in manifest["clips"]]
    situations = [
        str(clip["situation"]) for clip in manifest["clips"]
    ]
    documents = [
        retrieval_document(clip) for clip in manifest["clips"]
    ]
    vectors = TextEmbedder(
        args.model,
        args.device,
        pooling=args.pooling,
    ).encode(documents)

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
                browser_model=args.browser_model,
                pooling=args.pooling,
                query_prefix=args.query_prefix,
                cosine_weight=args.cosine_weight,
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
