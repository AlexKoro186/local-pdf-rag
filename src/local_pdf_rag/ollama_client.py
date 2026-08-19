"""Small adapter around Ollama's Python client."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


class OllamaGateway:
    """Provides batched embeddings and streaming chat generation."""

    def __init__(self, host: str, embedding_model: str, chat_model: str) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - exercised in installed app
            raise RuntimeError("Missing dependency 'ollama'. Run: pip install -e .") from exc

        # Ollama is a local/private endpoint; ambient system proxies should not intercept it.
        self._client = ollama.Client(host=host, trust_env=False)
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def ping(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(model=self.embedding_model, input=list(texts))
        embeddings = _field(response, "embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected number of embeddings.")
        return [list(vector) for vector in embeddings]

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        stream = self._client.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            options={"temperature": 0.1},
        )
        for event in stream:
            message = _field(event, "message", {})
            content = _field(message, "content", "")
            if content:
                yield str(content)
