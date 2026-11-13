# rag-eval-kit

[![CI](https://github.com/Madiyar77758/rag-eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Madiyar77758/rag-eval-kit/actions/workflows/ci.yml)

A small, dependency-light retrieval pipeline for documentation search **plus the harness that measures it**.
The point is not the retriever, it is the discipline around it: a pinned corpus, a labelled question set,
a dev/test split, one reproducible command, and numbers you can regenerate.

```
corpus (pinned Django 5.2.8 docs, 640 files)
   └─ chunking: fixed windows | RST-section aware
        ├─ BM25                (own implementation)
        ├─ hashed n-gram vectors (numpy, cosine)
        └─ hybrid: reciprocal rank fusion of both
             └─ doc-level ranking → Hit@k, MRR@10, nDCG@10, latency p50/p95
                  └─ FastAPI: /healthz  /search  /answer (LLM optional, extractive fallback)
```

## Results

`python -m rag_eval eval` builds 12 indexes (2 chunkers × 3 sizes × with/without heading prefix) and scores
3 retrievers on each, 36 configurations in total. The configuration is **selected on the 17 dev questions only**.
The table below is the **32 held-out test questions** (`results/latest.md` has all 36 rows).

| config | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | latency p50 / p95 (ms) |
|---|---|---|---|---|---|
| baseline: fixed/300 · BM25 | 0.469 | 0.812 | 0.615 | 0.701 | 1.6 / 3.8 |
| **selected on dev: fixed/600 · hybrid** | 0.531 | 0.844 | 0.647 | 0.712 | 2.3 / 3.2 |

How to read this, honestly:

- **The gain over plain BM25 is small and not significant.** On 32 questions one question is 3.1 points of Hit@1;
  the difference above is two questions. Treat the ranking of the top configurations as a tie.
  What the harness *does* show clearly is the spread: the worst configurations (`section/300 · vectors`, MRR@10 0.420)
  are far behind the best (0.657).
- **RST-section chunking did not help here.** It was the more "principled" idea and it lost to plain fixed windows
  on this corpus. It is kept in the grid because the comparison is the result.
- Latency is measured in-process on a CI-class CPU, one query at a time, after the index is built.
  Index build time per configuration is in `results/latest.json`.

### Limits (read before quoting any number)

- **49 questions, 32 in the test split.** Small by design so the whole thing runs in minutes; confidence intervals are wide.
- Questions were written for this repo as paraphrases rather than copies of documentation wording. Gold labels are
  **document level** (any gold file counts), and a script (`tests/test_questions.py`) checks that every gold file exists
  and contains its evidence phrase. They were not labelled by independent annotators.
- The "vectors" retriever is a **hashed word + bigram + character n-gram TF-IDF**, i.e. lexical. It is *not* a neural embedding model.
  `rag_eval/embedders.py` has an optional `sentence-transformers` adapter, but it is untested here and is not part of the reported numbers.
- `/answer` with an LLM works against any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_MODEL`, optional `LLM_API_KEY`),
  covered by tests with a mocked transport. It has **not** been run against a live model in this repo, and answer quality is not evaluated;
  only retrieval is.
- The Docker image is built and smoke-tested (`/healthz`, `/search`) in GitHub Actions, not on my machine.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m rag_eval fetch      # downloads Django 5.2.8 from PyPI, sha256-checked, extracts docs/
.venv/bin/python -m rag_eval eval       # writes results/latest.json and results/latest.md
.venv/bin/python -m rag_eval ask "How do I make several DB operations all succeed or roll back?"
.venv/bin/python -m rag_eval serve      # http://127.0.0.1:8000/docs
.venv/bin/python -m pytest -q           # 98 tests
```

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "csrf token in forms", "k": 3, "retriever": "hybrid"}'
```

Docker: `docker build -t rag-eval-kit . && docker run -p 8000:8000 rag-eval-kit`

## Design notes

- **No framework.** BM25, tokenizer with light stemming, chunkers, fusion and metrics are ~700 lines of plain Python/numpy,
  each unit-tested, so every number in the table is explainable.
- **Reciprocal rank fusion** (`k=60`) instead of score mixing: BM25 scores and cosine similarities are not on the same scale.
- **Document-level aggregation.** Chunks are retrieved, then collapsed to documents by best chunk, because the gold labels are documents.
- **Selection discipline.** `evaluate.run` picks the configuration by dev MRR@10 (ties: nDCG@10, then smaller chunks) and only then reports test.
  The test split is never used to choose anything.
- **Failure isolation.** A failing LLM endpoint falls back to the extractive answer instead of breaking `/answer`.

## Layout

```
rag_eval/     text · chunking · bm25 · vectors · embedders · fusion · index · metrics · corpus · evaluate · answer · api
data/         questions.jsonl (49 questions: id, question, gold docs, evidence phrase, split)
tests/        98 tests (unit + API + dataset consistency)
results/      latest.json, latest.md (regenerate with `eval`)
```

## License

MIT
