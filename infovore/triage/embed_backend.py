from collections.abc import Sequence
from importlib import import_module
from typing import Any

from infovore.config import ConfigError


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        try:
            module = import_module("sentence_transformers")
        except ImportError as error:
            raise ConfigError("install the embed extra: uv sync --extra embed") from error
        self.model_id = model_id
        self.revision = revision
        self._model: Any = module.SentenceTransformer(model_id, revision=revision, device=device)
        self._tokenizer: Any = self._model.tokenizer
        self.max_tokens = int(self._model.max_seq_length) - 2

    def token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def split(self, text: str, limit: int) -> list[str]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [self._tokenizer.decode(ids[i : i + limit]) for i in range(0, len(ids), limit)]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(v) for v in vector] for vector in vectors]


def load_embedder(model_id: str, revision: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_id, revision, "mps")
