"""Streamlit interface for the local PDF RAG assistant."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from local_pdf_rag.factory import create_components  # noqa: E402

st.set_page_config(
    page_title="Local PDF RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --accent: #6C63FF; }
      .block-container { max-width: 1120px; padding-top: 2.2rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
      .hero { padding: .25rem 0 1.25rem; }
      .eyebrow { color: #6C63FF; font-weight: 700; letter-spacing: .12em;
                 text-transform: uppercase; font-size: .78rem; }
      .hero h1 { font-size: clamp(2.1rem, 5vw, 3.7rem); line-height: 1.02;
                 margin: .45rem 0 .65rem; letter-spacing: -.045em; }
      .hero p { color: rgba(128,128,128,.95); font-size: 1.08rem; max-width: 720px; }
      div[data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.14);
                                         border-radius: 16px; padding: .35rem .75rem; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def components():
    return create_components()


def safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(". ")
    stem = Path(cleaned).stem[:120].strip(". ")
    return f"{stem or 'document'}.pdf"


def source_dicts(chunks) -> list[dict]:
    return [asdict(chunk) for chunk in chunks]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} retrieved excerpts"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{index}] {source['source']} · page {source['page']}**  "
                f"`{source['relevance']:.0%} relevance`"
            )
            excerpt = source["text"].strip()
            st.caption(excerpt[:650] + ("…" if len(excerpt) > 650 else ""))
            if index < len(sources):
                st.divider()


try:
    app = components()
except Exception as exc:
    st.error("The application could not start.")
    st.code(str(exc))
    st.info("Install the project dependencies and ensure the configuration in `.env` is valid.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Knowledge base")
    count_slot = st.empty()
    count_slot.metric("Indexed chunks", app.repository.count())
    status_text = "Ollama connected" if app.gateway.ping() else "Ollama unavailable"
    st.caption(f"● {status_text}")

    uploaded = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files stay on this machine and are excluded from Git.",
    )
    if st.button("Index documents", type="primary", use_container_width=True):
        app.settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        for item in uploaded:
            destination = app.settings.pdf_dir / safe_filename(item.name)
            destination.write_bytes(item.getvalue())

        paths = app.ingestor.discover()
        if not paths:
            st.warning("Add at least one PDF first.")
        elif not app.gateway.ping():
            st.error("Ollama is not reachable. Start it before indexing.")
        else:
            with st.spinner("Extracting, embedding, and indexing…"):
                report = app.ingestor.ingest(paths)
            count_slot.metric("Indexed chunks", app.repository.count())
            if report.files_indexed or report.files_skipped:
                st.success(report.summary())
            for error in report.errors:
                st.error(error)

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Source excerpts", 1, 10, app.settings.top_k)
    min_relevance = st.slider(
        "Minimum relevance",
        0.0,
        1.0,
        app.settings.min_relevance,
        0.05,
        help="Higher values reduce weak context but may omit useful passages.",
    )

    st.divider()
    st.caption(f"Chat · `{app.settings.chat_model}`")
    st.caption(f"Embeddings · `{app.settings.embedding_model}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Private · grounded · local-first</div>
      <h1>Ask your documents.<br/>Verify every answer.</h1>
      <p>A retrieval-augmented assistant that turns local PDFs into cited answers—
         with Ollama for inference and Chroma for semantic search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info("Index PDFs from the sidebar, then ask a question about their contents.")
    st.caption("Try one of these after indexing:")
    examples = [
        "Summarize the main argument in five points.",
        "Which definitions are essential to understand this document?",
        "What evidence supports the conclusion?",
    ]
    columns = st.columns(3)
    for column, example in zip(columns, examples, strict=True):
        if column.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input("Ask a question about your indexed PDFs")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant passages…"):
                chunks = app.rag.retrieve(prompt, top_k=top_k, min_relevance=min_relevance)
            answer = st.write_stream(app.rag.stream_answer(prompt, chunks))
            sources = source_dicts(chunks)
            render_sources(sources)
        except Exception as exc:
            answer = "I could not complete this request. Check the local runtime and try again."
            sources = []
            st.error(answer)
            st.code(str(exc))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
