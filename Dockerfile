FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY app.py ./
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/pdfs /app/chroma_db \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
