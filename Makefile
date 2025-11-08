
.PHONY: sync test doc bmark bmark_save bmark_cmp

sync:
	uv sync --all-groups

test:
	uv run pytest tests

doc:
	uv run sphinx-build -b html docs docs/_build

bmark:
	uv run pytest benchmarks --benchmark-sort=fullname

bmark_save:
	uv run pytest benchmarks --benchmark-save=baseline --benchmark-sort=fullname

bmark_cmp:
	uv run pytest benchmarks --benchmark-compare --benchmark-sort=fullname

lint:
	uv run black --check .
	uv run ruff check .
	uv run mypy src

pre-commit:
	uv run pre-commit install
