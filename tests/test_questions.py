"""The question set is data, so it gets tests too: every gold document must really contain its evidence."""

from pathlib import Path

import pytest

from rag_eval import corpus

QUESTIONS = corpus.load_questions(Path(__file__).parent.parent / "data" / "questions.jsonl")
CORPUS = Path(__file__).parent.parent / "data" / "corpus"


def test_ids_unique_and_splits_balanced():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    dev = sum(q["split"] == "dev" for q in QUESTIONS)
    assert 0.25 <= dev / len(QUESTIONS) <= 0.45


def test_no_gold_document_listed_twice():
    for q in QUESTIONS:
        assert len(q["gold"]) == len(set(q["gold"])), q["id"]


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not fetched: run `python -m rag_eval fetch`")
@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_gold_documents_exist_and_contain_the_evidence(q):
    for doc in q["gold"]:
        path = CORPUS / doc
        assert path.exists(), f"{q['id']}: {doc} missing"
        assert q["evidence"].lower() in path.read_text(encoding="utf-8", errors="replace").lower(), f"{q['id']}: {doc}"
