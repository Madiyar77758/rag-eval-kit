import pytest

QUESTIONS = {
    "How do I make several database operations all succeed or all roll back?": "db/transactions.txt",
    "Where is per visitor data stored and how do I change the backend?": "http/sessions.txt",
    "How can I avoid one query per row when reading related objects?": "db/queries.txt",
    "How do I gather stylesheets for deployment?": "howto/static.txt",
}


@pytest.mark.parametrize("retriever", ["bm25", "vectors", "hybrid"])
def test_every_retriever_finds_the_right_document_first(index, retriever):
    for question, gold in QUESTIONS.items():
        assert index.search_docs(question, retriever=retriever, k=3)[0].doc_id == gold, (retriever, question)


def test_results_are_one_hit_per_document(index):
    hits = index.search_docs("atomic savepoints transaction", k=5)
    assert len({h.doc_id for h in hits}) == len(hits)


def test_unknown_retriever_is_rejected(index):
    with pytest.raises(ValueError, match="unknown retriever"):
        index.search_docs("x", retriever="nope")


def test_index_reports_size(index, docs):
    assert index.n_docs == len(docs) and len(index.chunks) >= len(docs)
