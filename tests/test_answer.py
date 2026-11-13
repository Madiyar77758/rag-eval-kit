import httpx
import pytest

from rag_eval.answer import answer


@pytest.fixture
def hits(index):
    return index.search_docs("how do I roll back database operations", k=2)


def test_extractive_fallback_cites_top_passage(hits, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    out = answer("how do I roll back database operations", hits)
    assert out["mode"] == "extractive" and out["answer"].endswith("[1]")
    assert out["citations"][0]["doc_id"] == hits[0].doc_id


def test_llm_mode_sends_passages_and_returns_model_text(hits, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "demo")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Use atomic [1]."}}]})

    out = answer("q", hits, transport=httpx.MockTransport(handler))
    assert out["mode"] == "llm" and out["answer"] == "Use atomic [1]."
    assert seen["url"] == "http://llm.test/v1/chat/completions" and "[1]" in seen["body"]


def test_llm_failure_falls_back_to_extractive(hits, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "demo")
    out = answer("q", hits, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert out["mode"] == "extractive"


def test_no_hits_is_reported():
    assert answer("q", [])["mode"] == "none"
