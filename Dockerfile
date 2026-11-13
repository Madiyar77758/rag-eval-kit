FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY rag_eval ./rag_eval
RUN pip install .

# The corpus is not stored in git: download the pinned Django docs (sha256-verified) at build time.
COPY data/questions.jsonl ./data/questions.jsonl
RUN python -m rag_eval fetch

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["python", "-m", "rag_eval", "serve", "--host", "0.0.0.0", "--port", "8000"]
