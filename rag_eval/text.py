"""Tokenisation shared by every retriever."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """a an and are as at be but by do does for from how i if in into is it its my of on or so that the
    their then there these this to use used using via was what when where which who why will with you your
    can should would could not no than too very""".split()
)


def stem(token: str) -> str:
    """A deliberately small suffix stripper: enough to match plurals and -ing/-ed forms."""
    if len(token) <= 3 or "_" in token:  # identifiers such as select_related are kept whole
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if suffix == "s" and token.endswith("ss"):
            continue
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, stop: bool = True, do_stem: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if stop:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_stem:
        tokens = [stem(t) for t in tokens]
    return tokens
