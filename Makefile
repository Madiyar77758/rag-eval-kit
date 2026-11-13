.PHONY: setup fetch test eval serve

setup:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

fetch:
	.venv/bin/python -m rag_eval fetch

test:
	.venv/bin/python -m pytest -q

eval:
	.venv/bin/python -m rag_eval eval

serve:
	.venv/bin/python -m rag_eval serve
