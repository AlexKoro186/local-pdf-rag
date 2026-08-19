import sys
from types import SimpleNamespace

import pytest

from local_pdf_rag.ollama_client import OllamaGateway, _field


class FakeClient:
    def __init__(self, host: str, trust_env: bool) -> None:
        self.host = host
        assert trust_env is False
        self.available = True

    def list(self):
        if not self.available:
            raise ConnectionError("offline")
        return {"models": []}

    def embed(self, model, input):
        return {"embeddings": [[float(index), 1.0] for index, _ in enumerate(input)]}

    def chat(self, **kwargs):
        assert kwargs["stream"] is True
        return [
            {"message": {"content": "first "}},
            SimpleNamespace(message=SimpleNamespace(content="second")),
        ]


def gateway(monkeypatch) -> OllamaGateway:
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    return OllamaGateway("http://ollama", "embed-model", "chat-model")


def test_gateway_embeds_pings_and_streams(monkeypatch) -> None:
    client = gateway(monkeypatch)

    assert client.ping()
    assert client.embed([]) == []
    assert client.embed(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert "".join(client.stream_chat("system", "user")) == "first second"

    client._client.available = False
    assert not client.ping()


def test_gateway_rejects_malformed_embedding_response(monkeypatch) -> None:
    client = gateway(monkeypatch)
    client._client.embed = lambda **_: {"embeddings": []}
    with pytest.raises(RuntimeError, match="unexpected number"):
        client.embed(["text"])


def test_field_supports_mappings_objects_and_defaults() -> None:
    assert _field({"value": 1}, "value") == 1
    assert _field(SimpleNamespace(value=2), "value") == 2
    assert _field({}, "missing", 3) == 3
