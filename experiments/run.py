"""Run the independent paper studies, or render saved measurements."""

import argparse
import json
from pathlib import Path

from common import metadata, prepare, save

# isort: split
from curves import run as run_curves
from kan import run as run_kan
from render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "render"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    directory = args.output / "full"

    if args.mode == "full":
        prepare()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps({**metadata(), "mode": args.mode}, indent=2) + "\n")
        for name, runner in (("curves", run_curves), ("kan", run_kan)):
            save(directory, name, runner())

    render(directory)
    print(f"Wrote {args.mode} results to {directory}")


if __name__ == "__main__":
    main()
