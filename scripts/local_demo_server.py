#!/usr/bin/env python3
"""Serve the production app and local sentence-vector retrieval API."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from text_embeddings import (
    DEFAULT_MODEL,
    TextEmbedder,
    bm25_scores,
    directional_tactical_class,
    expand_tactical_query,
    hybrid_scores,
)


class DemoServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        directory: Path,
        model: Path,
        index: Path,
    ) -> None:
        payload = np.load(index)
        self.ids = payload["ids"].tolist()
        self.situations = payload["situations"].tolist()
        self.documents = payload["documents"].tolist()
        self.vectors = payload["vectors"]
        self.embedder = TextEmbedder(model)
        self.directory = str(directory.resolve())
        super().__init__(address, DemoHandler)


class DemoHandler(SimpleHTTPRequestHandler):
    server: DemoServer

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=args[2].directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/search":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            query = str(body.get("query", "")).strip()
            if not query or len(query) > 500:
                raise ValueError("Query must contain 1–500 characters")
            query_vector = self.server.embedder.encode([expand_tactical_query(query)])[0]
            similarities = self.server.vectors @ query_vector
            lexical = bm25_scores(query, self.server.documents)
            combined = hybrid_scores(similarities, lexical)
            required_class = directional_tactical_class(query)
            eligible = np.asarray(
                [
                    required_class is None or situation == required_class
                    for situation in self.server.situations
                ]
            )
            order = np.flatnonzero(eligible)[np.argsort(-combined[eligible])]
            response = {
                "model": self.server.embedder.model.config._name_or_path,
                "metric": "hybrid_cosine_bm25",
                "weights": {"cosine": 0.65, "bm25": 0.35},
                "results": [
                    {
                        "id": self.server.ids[index],
                        "score": round(float(combined[index]), 6),
                        "cosine": round(float(similarities[index]), 6),
                        "bm25": round(float(lexical[index]), 6),
                    }
                    for index in order
                ],
            }
            encoded = json.dumps(response).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (ValueError, json.JSONDecodeError) as error:
            encoded = str(error).encode()
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("out"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/fullmatch_text.npz"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    server = DemoServer((args.host, args.port), args.directory, args.model, args.index)
    print(f"PressLens vector demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
