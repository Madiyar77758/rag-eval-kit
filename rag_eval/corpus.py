"""Corpus handling: download the pinned Django documentation and load documents from disk."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

DJANGO_VERSION = "5.2.8"
PYPI_JSON = f"https://pypi.org/pypi/Django/{DJANGO_VERSION}/json"
DEFAULT_DIR = Path("data/corpus")


def fetch_django_docs(dest: Path = DEFAULT_DIR) -> int:
    """Download the Django sdist from PyPI, verify its sha256 and extract docs/**/*.txt into ``dest``."""
    with urllib.request.urlopen(PYPI_JSON, timeout=60) as resp:
        meta = json.load(resp)
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    with urllib.request.urlopen(sdist["url"], timeout=300) as resp:
        blob = resp.read()
    if hashlib.sha256(blob).hexdigest() != sdist["digests"]["sha256"]:
        raise RuntimeError("sha256 mismatch for the downloaded Django sdist")
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] != "docs" or not member.isfile() or not member.name.endswith(".txt"):
                continue
            rel = Path(*parts[2:])
            if ".." in rel.parts:  # defensive: never write outside dest
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
            count += 1
    return count


def load_docs(root: Path = DEFAULT_DIR) -> dict[str, str]:
    """Map relative posix path -> text for every .txt file under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `python -m rag_eval fetch` first.")
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace") for p in sorted(root.rglob("*.txt"))}


def load_questions(path: Path = Path("data/questions.jsonl")) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r["split"] not in {"dev", "test"} or not r["gold"]:
            raise ValueError(f"bad question row: {r['id']}")
    return rows
