from __future__ import annotations

import os
import math
import hashlib
from typing import Iterable, List, Optional
import importlib

def _np():  # lazy numpy import for environments without deps until runtime
    try:
        return importlib.import_module("numpy")
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("NumPy is required. Install 'numpy' or switch embedder to 'hash'.") from e


class BaseEmbedder:
    def encode(self, texts: List[str]) -> np.ndarray:  # (n, d)
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """
    Fully offline, dependency-light embedder.
    - Tokenizes by words and char trigrams
    - Hashes into a fixed-size vector with signed counts
    - L2 normalizes for cosine similarity usage
    """

    def __init__(self, dim: int = 768, seed: int = 13) -> None:
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def _hash(self, s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
        return int.from_bytes(h.digest(), "big")

    def _features(self, text: str) -> Iterable[tuple[int, float]]:
        t = text.lower()
        # word tokens
        word = []
        for ch in t:
            if ch.isalnum() or ch == "_":
                word.append(ch)
            else:
                if word:
                    w = "".join(word)
                    idx = self._hash("w:" + w) % self._dim
                    sign = 1.0 if (self._hash("s:" + w) & 1) else -1.0
                    yield idx, sign
                    word = []
        if word:
            w = "".join(word)
            idx = self._hash("w:" + w) % self._dim
            sign = 1.0 if (self._hash("s:" + w) & 1) else -1.0
            yield idx, sign

        # char trigrams
        trig = ["<"] + list(t) + [">"]
        for i in range(len(trig) - 2):
            g = "".join(trig[i : i + 3])
            idx = self._hash("g:" + g) % self._dim
            sign = 1.0 if (self._hash("gs:" + g) & 1) else -1.0
            yield idx, sign * 0.5

    def encode(self, texts: List[str]):
        np = _np()
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = np.zeros(self._dim, dtype=np.float32)
            for idx, val in self._features(text):
                vec[idx] += val
            # L2 normalize
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
            out[i] = vec
        return out


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name_or_path: Optional[str] = None) -> None:
        model_name = model_name_or_path or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not available. Install it or use HashingEmbedder."
            ) from e

        self._model = SentenceTransformer(model_name)
        # warm query for dim
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]):
        np = _np()
        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32, copy=False)


def build_embedder(name: Optional[str]) -> BaseEmbedder:
    if name is None or name.lower() == "hash":
        return HashingEmbedder()
    if name.lower() in {"st", "sentence-transformers", "sentence"}:
        return SentenceTransformerEmbedder()
    # treat as path/name for ST
    return SentenceTransformerEmbedder(name)
