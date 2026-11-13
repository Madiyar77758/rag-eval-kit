"""Evaluation harness: tune on the dev split, report on the held-out test split."""

from __future__ import annotations

import itertools
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .index import RETRIEVERS, Index
from .metrics import hit_at_k, ndcg_at_k, percentile, reciprocal_rank


@dataclass(frozen=True)
class Config:
    chunking: str
    size: int
    title: bool
    retriever: str

    @property
    def label(self) -> str:
        return f"{self.chunking}/{self.size}{'+title' if self.title else ''} · {self.retriever}"


DEFAULT_GRID = {
    "chunking": ("fixed", "section"),
    "size": (150, 300, 600),
    "title": (False, True),
    "retriever": RETRIEVERS,
}
BASELINE = Config("fixed", 300, False, "bm25")


def score(index: Index, questions: list[dict], retriever: str, k: int = 10) -> dict:
    hits1, hits5, rr, ndcg, lat = [], [], [], [], []
    for q in questions:
        gold = set(q["gold"])
        started = time.perf_counter()
        ranked = [h.doc_id for h in index.search_docs(q["question"], retriever=retriever, k=k)]
        lat.append((time.perf_counter() - started) * 1000)
        hits1.append(hit_at_k(ranked, gold, 1))
        hits5.append(hit_at_k(ranked, gold, 5))
        rr.append(reciprocal_rank(ranked, gold))
        ndcg.append(ndcg_at_k(ranked, gold, k))
    return {
        "n": len(questions),
        "hit@1": statistics.fmean(hits1),
        "hit@5": statistics.fmean(hits5),
        "mrr@10": statistics.fmean(rr),
        "ndcg@10": statistics.fmean(ndcg),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }


def run(docs: dict[str, str], questions: list[dict], grid: dict | None = None, log=lambda m: None) -> dict:
    grid = grid or DEFAULT_GRID
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    rows = []
    index_cfgs = list(itertools.product(grid["chunking"], grid["size"], grid["title"]))
    for n, (chunking, size, title) in enumerate(index_cfgs, start=1):
        log(f"[{n}/{len(index_cfgs)}] building index {chunking}/{size} title={title}")
        index = Index(docs, chunking=chunking, size=size, overlap=size // 6, title=title)
        for retriever in grid["retriever"]:
            cfg = Config(chunking, size, title, retriever)
            rows.append(
                {
                    "config": asdict(cfg),
                    "label": cfg.label,
                    "chunks": len(index.chunks),
                    "build_seconds": round(index.build_seconds, 1),
                    "dev": score(index, dev, retriever),
                    "test": score(index, test, retriever),
                }
            )
    best = max(rows, key=lambda r: (r["dev"]["mrr@10"], r["dev"]["ndcg@10"], -r["config"]["size"]))
    baseline = next((r for r in rows if r["config"] == asdict(BASELINE)), None)
    return {
        "corpus": {"documents": len(docs), "questions": len(questions), "dev": len(dev), "test": len(test)},
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "selected_on_dev": best["label"],
        "baseline": baseline["label"] if baseline else None,
        "rows": rows,
    }


def _fmt(m: dict) -> str:
    quality = f"{m['hit@1']:.3f} | {m['hit@5']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f}"
    return f"{quality} | {m['latency_p50_ms']:.1f} / {m['latency_p95_ms']:.1f}"


def to_markdown(result: dict) -> str:
    c = result["corpus"]
    rows = {r["label"]: r for r in result["rows"]}
    best = rows[result["selected_on_dev"]]
    base = rows.get(result["baseline"]) if result["baseline"] else None
    head = "| config | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | latency p50 / p95 (ms) |\n|---|---|---|---|---|---|\n"
    out = [
        f"Corpus: {c['documents']} documents. Questions: {c['questions']} "
        f"({c['dev']} dev for tuning, {c['test']} test held out).",
        "",
        "### Baseline vs configuration selected on dev (test split)",
        "",
        head
        + (f"| baseline: {base['label']} | {_fmt(base['test'])} |\n" if base else "")
        + f"| **selected: {best['label']}** | {_fmt(best['test'])} |",
        "",
        "### All configurations (test split, sorted by MRR@10)",
        "",
        head
        + "\n".join(f"| {r['label']} | {_fmt(r['test'])} |" for r in sorted(result["rows"], key=lambda r: -r["test"]["mrr@10"])),
        "",
        "Selection used the dev split only; the test split is reported, never tuned on.",
    ]
    return "\n".join(out)


def save(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (directory / "latest.md").write_text(to_markdown(result) + "\n", encoding="utf-8")


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
