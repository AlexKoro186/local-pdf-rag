"""Backward-compatible entry point. Prefer `local-pdf-rag ingest`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from local_pdf_rag.cli import main  # noqa: E402

raise SystemExit(main(["ingest", *sys.argv[1:]]))
