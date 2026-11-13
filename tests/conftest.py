import pytest

from rag_eval.index import Index

DOCS = {
    "db/transactions.txt": "Database transactions\n=====================\n\nWrap operations in atomic so that they all succeed or all roll back. "
    "Use the atomic decorator or context manager to start a transaction and commit at the end.\n\nSavepoints\n----------\n\n"
    "Nested atomic blocks create savepoints that allow partial rollback inside a transaction.",
    "http/sessions.txt": "Sessions\n========\n\nSessions store per visitor data between requests. Change the SESSION_ENGINE setting to switch "
    "the storage backend to cache, files or cookies.",
    "db/queries.txt": "Making queries\n==============\n\nUse select_related to follow foreign keys in a single query and avoid one query per row "
    "when reading related objects in a loop. Use prefetch_related for many to many relations.",
    "howto/static.txt": "Static files\n============\n\nRun collectstatic to gather stylesheets and scripts into STATIC_ROOT for deployment.",
}


@pytest.fixture(scope="session")
def docs():
    return DOCS


@pytest.fixture(scope="session")
def index(docs):
    return Index(docs, chunking="section", size=60, overlap=10, title=True)
