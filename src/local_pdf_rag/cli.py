"""Command-line interface for indexing, querying, and retrieval evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .evaluation import evaluate, load_cases
from .factory import create_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-pdf-rag",
        description="Private, local retrieval-augmented generation for PDF documents.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Index PDFs from data/pdfs or explicit paths.")
    ingest.add_argument("paths", nargs="*", type=Path)

    ask = commands.add_parser("ask", help="Ask a grounded question.")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--min-relevance", type=float)

    commands.add_parser("status", help="Show local runtime and index status.")

    evaluation = commands.add_parser("evaluate", help="Evaluate retrieval with a JSONL dataset.")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    components = create_components()

    if args.command == "ingest":
        paths = args.paths or None
        report = components.ingestor.ingest(paths)
        print(f"Indexing complete: {report.summary()}.")
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1 if report.errors else 0

    if args.command == "ask":
        question = " ".join(args.question).strip()
        chunks = components.rag.retrieve(
            question,
            top_k=args.top_k or components.settings.top_k,
            min_relevance=(
                args.min_relevance
                if args.min_relevance is not None
                else components.settings.min_relevance
            ),
        )
        print(components.rag.answer(question, chunks))
        if chunks:
            print("\nSources")
            for index, chunk in enumerate(chunks, start=1):
                print(f"[{index}] {chunk.citation} · relevance {chunk.relevance:.0%}")
        return 0

    if args.command == "status":
        print(f"Ollama: {'reachable' if components.gateway.ping() else 'not reachable'}")
        print(f"Indexed chunks: {components.repository.count()}")
        print(f"Chat model: {components.settings.chat_model}")
        print(f"Embedding model: {components.settings.embedding_model}")
        return 0

    cases = load_cases(args.dataset)
    result = evaluate(components.rag, cases, args.top_k)
    print(f"Cases: {result.cases}")
    print(f"Hit@{args.top_k}: {result.hit_rate:.1%}")
    print(f"MRR@{args.top_k}: {result.mean_reciprocal_rank:.3f}")
    return 0
