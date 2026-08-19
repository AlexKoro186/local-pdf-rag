.PHONY: install run ingest test lint format check

install:
	python -m pip install -e ".[dev]"

run:
	streamlit run app.py

ingest:
	local-pdf-rag ingest

test:
	pytest --cov=local_pdf_rag --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

check: lint test
