import json
import re
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "doc" / "source" / "examples"

LOCAL_PATH_PATTERNS = (
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.IGNORECASE),
    re.compile(r"(?:^|[\s\"'(])~[/\\]"),
    re.compile(r"(?:^|[/\\])\.venv[/\\]"),
)

UNWANTED_OUTPUT_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "CUDA environment warning": re.compile(r"CUDA initialization:"),
    "stale dependency warning": re.compile(r"Looks like you're using an outdated"),
}


def _join_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(item for item in value if isinstance(item, str))
    return ""


def _output_text(output: dict) -> str:
    chunks = [_join_text(output.get("text")), _join_text(output.get("traceback"))]

    data = output.get("data")
    if isinstance(data, dict):
        chunks.extend(
            _join_text(value)
            for mime_type, value in data.items()
            if mime_type.startswith("text/") or mime_type == "application/json"
        )

    return "".join(chunks)


def test_documentation_notebooks_are_sanitized() -> None:
    violations = []
    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebook_paths, f"No documentation notebooks found in {NOTEBOOK_DIR}"

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))

        for cell_number, cell in enumerate(notebook["cells"], start=1):
            source = _join_text(cell.get("source"))
            if "torch.torch." in source:
                violations.append(f"{path.name}: cell {cell_number}: torch.torch typo")

            for output_number, output in enumerate(cell.get("outputs", []), start=1):
                location = f"{path.name}: cell {cell_number}, output {output_number}"
                if output.get("output_type") == "error":
                    violations.append(f"{location}: error output")
                if "jetTransient" in output:
                    violations.append(f"{location}: transient Jupyter metadata")

                text = _output_text(output)
                if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
                    violations.append(f"{location}: machine-local path")

                for description, pattern in UNWANTED_OUTPUT_PATTERNS.items():
                    if pattern.search(text):
                        violations.append(f"{location}: {description}")

    assert not violations, "Notebook hygiene failures:\n" + "\n".join(violations)
