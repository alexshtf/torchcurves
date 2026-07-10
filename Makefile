
.PHONY: sync test doc bmark bmark_save bmark_cmp

sync:
	uv sync --all-groups

test:
	uv run pytest tests

doc:
	rm -rf doc/source/generated
	uv run sphinx-build -W -b html doc/source doc/build/html

bmark:
	uv run pytest benchmarks --benchmark-sort=fullname

bmark_save:
	uv run pytest benchmarks --benchmark-save=baseline --benchmark-sort=fullname

bmark_cmp:
	uv run pytest benchmarks --benchmark-compare --benchmark-sort=fullname

lint:
	uv run ruff check .
	uv run pyright src

pre-commit:
	uv run pre-commit install
