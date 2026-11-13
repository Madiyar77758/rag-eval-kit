import pytest
from fastapi.testclient import TestClient

from rag_eval.api import create_app


@pytest.fixture
def client(index, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with TestClient(create_app(index)) as c:
        yield c


def test_healthz(client, docs):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["documents"] == len(docs)


def test_search_returns_ranked_documents(client):
    r = client.post("/search", json={"query": "roll back database operations", "k": 2})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["doc_id"] == "db/transactions.txt" and results[0]["rank"] == 1


def test_answer_has_citations(client):
    body = client.post("/answer", json={"query": "which setting changes the session backend"}).json()
    assert body["citations"][0]["doc_id"] == "http/sessions.txt" and body["mode"] == "extractive"


@pytest.mark.parametrize(
    "payload", [{"query": "x"}, {"query": "valid query", "k": 0}, {"query": "valid query", "retriever": "nope"}]
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/search", json=payload).status_code == 422
