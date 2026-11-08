"""Run the independent paper studies, or render saved measurements."""

import argparse
import json
from pathlib import Path

from common import metadata, prepare, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "render"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    directory = args.output / "full"

    if args.mode != "render":
        prepare()
        from curves import run as run_curves
        from kan import run as run_kan

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps({**metadata(), "mode": args.mode}, indent=2) + "\n")
        for name, runner in (("curves", run_curves), ("kan", run_kan)):
            raw, aggregates = runner()
            save(directory, f"{name}_raw", raw)
            save(directory, f"{name}_summary", aggregates)

    from render import render

    render(directory)
    print(f"Wrote {args.mode} results to {directory}")


if __name__ == "__main__":
    main()
