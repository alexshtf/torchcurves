
.PHONY: sync test doc bmark bmark_save bmark_cmp

sync:
	uv sync --all-groups

test:
	uv run pytest

doc:
	uv run sphinx-build -b html docs docs/_build

bmark:
	uv run pytest benchmarks

bmark_save:
	uv run pytest benchmarks --benchmark-save=baseline

bmark_cmp:
	uv run pytest benchmarks --benchmark-compare --benchmark-sort=fullname