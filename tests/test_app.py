from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_ollama(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_CHROMA_PATH", str(tmp_path / "chroma"))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert len(app.chat_input) == 1
    assert len(app.slider) == 2
    assert any(item.value == "Knowledge base" for item in app.subheader)
    assert any(item.label == "Index documents" for item in app.button)
    assert any("Ollama unavailable" in item.value for item in app.caption)
