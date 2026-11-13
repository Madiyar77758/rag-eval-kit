from rag_eval import evaluate

QUESTIONS = [
    {
        "id": "q1",
        "question": "how do I make operations all succeed or roll back",
        "gold": ["db/transactions.txt"],
        "split": "dev",
    },
    {
        "id": "q2",
        "question": "which setting switches the session storage backend",
        "gold": ["http/sessions.txt"],
        "split": "test",
    },
    {"id": "q3", "question": "avoid one query per row for related objects", "gold": ["db/queries.txt"], "split": "test"},
]
GRID = {"chunking": ("fixed", "section"), "size": (40,), "title": (False, True), "retriever": ("bm25", "hybrid")}


def test_run_scores_every_configuration_on_dev_and_test(docs):
    result = evaluate.run(docs, QUESTIONS, GRID)
    assert len(result["rows"]) == 2 * 1 * 2 * 2
    assert result["corpus"] == {"documents": 4, "questions": 3, "dev": 1, "test": 2}
    row = result["rows"][0]
    assert row["dev"]["n"] == 1 and row["test"]["n"] == 2
    assert 0 <= row["test"]["mrr@10"] <= 1


def test_selection_uses_dev_only(docs):
    result = evaluate.run(docs, QUESTIONS, GRID)
    best = max(result["rows"], key=lambda r: (r["dev"]["mrr@10"], r["dev"]["ndcg@10"], -r["config"]["size"]))
    assert result["selected_on_dev"] == best["label"]


def test_markdown_and_save_roundtrip(docs, tmp_path):
    grid = {**GRID, "chunking": ("fixed",), "title": (False,), "retriever": ("bm25",)}
    grid["size"] = (300,)  # includes the declared baseline configuration
    result = evaluate.run(docs, QUESTIONS, grid)
    md = evaluate.to_markdown(result)
    assert "Baseline vs configuration selected on dev" in md and "never tuned on" in md
    evaluate.save(result, tmp_path)
    assert (tmp_path / "latest.json").exists() and (tmp_path / "latest.md").read_text().startswith("Corpus:")
