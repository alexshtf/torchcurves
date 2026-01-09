## AGENTS.md

This file documents local working agreements for this workspace.

### Non-negotiables
- We use `uv` for Python environments, dependency management, and running tools.
- Tests and lint must be run prior to any commits.
- Always run tests and lint after implementing changes.
- Even if this file is **not** under source control, it should NEVER be deleted and will always remain in this workspace.

### Repository overview
- Python project with sources under `src/` and tests under `tests/`.
- Project metadata lives in `pyproject.toml` and the lockfile is `uv.lock`.

### Common commands (use `uv`)
- Install/sync: `uv sync --all-groups` (or `make sync`)
- Run tests: `uv run pytest tests` (or `make test`)
- Run lint: `uv run ruff check .`, `uv run mypy src` (or `make lint`)
- Build docs: `make -C doc html`

### Coding style (observed)
- Python 3.9+ with type hints; annotate functions, class attributes, and tensor shapes in comments when helpful.
- Docstrings follow Google style (Args/Returns/Raises/Note) and often use raw strings for math markup.
- Prefer vectorized Torch ops; loops are used only for algorithmic recurrences (e.g., Clenshaw, Cox-de Boor).
- Validate inputs early; raise `ValueError`/`TypeError` with clear messages.
- Formatting follows ruff's 120-character line length; imports grouped stdlib, third-party, local.

### Documentation style (observed)
- Sphinx + reStructuredText in `doc/source` with `.. toctree::`, `.. automodule::`, and `.. autosummary::`.
- API docs come from docstrings via napoleon (Google style) and `sphinx_autodoc_typehints`.
- Math uses `:math:` and `.. math::` with MathJax; example notebooks live in `doc/source/examples`.

### Testing style (observed)
- Pytest with unittest-style `Test*` classes under `tests/`.
- Numeric checks use `torch.testing.assert_close` and `torch.autograd.gradcheck`.
- Tests handle device/dtype explicitly (CPU/GPU selection, float64 for gradcheck).

### Workflow notes
- Keep edits minimal and focused.
- Prefer updating tests alongside behavior changes.
