AFFINITY ?= 1

export OMP_NUM_THREADS := $(AFFINITY)
export MKL_NUM_THREADS := $(AFFINITY)
export OPENBLAS_NUM_THREADS := $(AFFINITY)
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

PIN = taskset --cpu-list 0-$$(( $(AFFINITY) - 1 ))

.PHONY: run test render

run: test
	$(PIN) uv run --locked python run.py full

test:
	uv sync --locked
	$(PIN) uv run --locked pytest

render:
	$(PIN) uv run --locked python run.py render
