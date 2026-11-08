.PHONY: run render

run:
	uv sync --locked
	uv run --locked pytest
	uv run --locked python run.py full

render:
	uv run --locked python run.py render
