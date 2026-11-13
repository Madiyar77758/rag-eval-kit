"""Command line: fetch the corpus, run the evaluation, ask a question, start the API."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import corpus
from .evaluate import progress, run, save, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the pinned Django docs from PyPI")
    f.add_argument("--dest", type=Path, default=corpus.DEFAULT_DIR)

    e = sub.add_parser("eval", help="run the evaluation grid and write results/")
    e.add_argument("--corpus", type=Path, default=corpus.DEFAULT_DIR)
    e.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    e.add_argument("--out", type=Path, default=Path("results"))

    a = sub.add_parser("ask", help="retrieve (and answer) one question from the command line")
    a.add_argument("question")
    a.add_argument("--corpus", type=Path, default=corpus.DEFAULT_DIR)
    a.add_argument("-k", type=int, default=5)

    s = sub.add_parser("serve", help="start the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        print(f"extracted {corpus.fetch_django_docs(args.dest)} documents into {args.dest}")
    elif args.cmd == "eval":
        result = run(corpus.load_docs(args.corpus), corpus.load_questions(args.questions), log=progress)
        save(result, args.out)
        print(to_markdown(result))
    elif args.cmd == "ask":
        from .answer import answer
        from .index import Index

        index = Index(corpus.load_docs(args.corpus))
        result = answer(args.question, index.search_docs(args.question, k=args.k))
        print(result["answer"])
        for c in result["citations"]:
            print(f"  [{c['n']}] {c['doc_id']}  ({c['heading']})")
    else:
        import uvicorn

        uvicorn.run("rag_eval.api:app", host=args.host, port=args.port, factory=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
