"""Backward-compatible entry point. Prefer `local-pdf-rag ask`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from local_pdf_rag.cli import main  # noqa: E402

question = sys.argv[1:] or [input("Question: ")]
raise SystemExit(main(["ask", *question]))
