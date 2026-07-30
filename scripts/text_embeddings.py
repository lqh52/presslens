"""Local sentence embeddings used by the PressLens vector index and API."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = Path("models/text-embedding/all-MiniLM-L6-v2")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def expand_tactical_query(query: str) -> str:
    """Map explicit flank language to the catalogue's canonical description.

    Retrieval remains one text embedding followed by cosine similarity; this
    only makes left/right intent explicit for a model that otherwise embeds
    antonymic tactical descriptions too closely.
    """
    words = set(query.lower().replace("-", " ").split())
    if "left" in words and "right" not in words:
        return (
            f"{query}. Left touchline trap. Pressure is concentrated around "
            "the left touchline with defenders positioned inside the ball. "
            "Tactical class: trap left."
        )
    if "right" in words and "left" not in words:
        return (
            f"{query}. Right touchline trap. Pressure is concentrated around "
            "the right touchline with defenders positioned inside the ball. "
            "Tactical class: trap right."
        )
    return query


def directional_tactical_class(query: str) -> str | None:
    words = set(query.lower().replace("-", " ").split())
    tactical = {"press", "pressing", "pressure", "trap", "touchline", "flank"}
    if not words.intersection(tactical):
        return None
    if "left" in words and "right" not in words:
        return "trap_left"
    if "right" in words and "left" not in words:
        return "trap_right"
    return None


def lexical_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().replace("-", " "))


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Return BM25 scores without requiring an external search service."""
    tokenized = [lexical_tokens(document) for document in documents]
    query_terms = set(lexical_tokens(query))
    scores = np.zeros(len(documents), dtype=np.float32)
    if not query_terms or not documents:
        return scores
    average_length = sum(map(len, tokenized)) / len(tokenized)
    for term in query_terms:
        frequency_by_document = np.asarray(
            [tokens.count(term) for tokens in tokenized],
            dtype=np.float32,
        )
        document_frequency = int(np.count_nonzero(frequency_by_document))
        if document_frequency == 0:
            continue
        inverse_document_frequency = np.log(
            1.0
            + (
                len(documents) - document_frequency + 0.5
            ) / (document_frequency + 0.5)
        )
        lengths = np.asarray([len(tokens) for tokens in tokenized])
        denominator = frequency_by_document + k1 * (
            1.0 - b + b * lengths / max(average_length, 1.0)
        )
        scores += inverse_document_frequency * (
            frequency_by_document * (k1 + 1.0)
        ) / np.maximum(denominator, 1e-8)
    return scores


def hybrid_scores(
    cosine_scores: np.ndarray,
    lexical_scores: np.ndarray,
    *,
    cosine_weight: float = 0.65,
) -> np.ndarray:
    """Blend normalized semantic and lexical relevance."""
    semantic = np.clip((cosine_scores + 1.0) / 2.0, 0.0, 1.0)
    lexical_max = float(lexical_scores.max(initial=0.0))
    lexical = (
        lexical_scores / lexical_max
        if lexical_max > 0
        else np.zeros_like(lexical_scores)
    )
    return cosine_weight * semantic + (1.0 - cosine_weight) * lexical


class TextEmbedder:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        device: str = "cpu",
        *,
        pooling: str = "mean",
    ) -> None:
        self.device = torch.device(device)
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        prefix: str = "",
    ) -> np.ndarray:
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    [prefix + text for text in texts[start : start + batch_size]],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self.model(**tokens).last_hidden_state
                if self.pooling == "cls":
                    pooled = output[:, 0]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
                rows.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)
