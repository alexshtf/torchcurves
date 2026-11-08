"""Create the paper figure and tables from saved measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"TorchCurves": "#2764a5", "NumPy": "#cf6c38", "SciPy": "#cf6c38"}
HEADER = ("Coefficients", "Batch", "CPU TorchCurves", "CPU efficient-kan", "GPU TorchCurves", "GPU efficient-kan")


def plot_curves(results: list[dict], directory: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for row, family in enumerate(("Legendre", "B-spline")):
        reference = "NumPy" if family == "Legendre" else "SciPy"
        for column, count in enumerate((1, 32)):
            axis = axes[row, column]
            for implementation in ("TorchCurves", reference):
                points = sorted(
                    (
                        item
                        for item in results
                        if item["family"] == family
                        and item["curves"] == count
                        and item["implementation"] == implementation
                    ),
                    key=lambda item: item["batch"],
                )
                if not points:
                    continue
                batches = [item["batch"] for item in points]
                axis.plot(
                    batches,
                    [item["median_ms"] for item in points],
                    marker="o",
                    color=COLORS[implementation],
                    label=implementation,
                )
                axis.fill_between(
                    batches,
                    [item["q1_ms"] for item in points],
                    [item["q3_ms"] for item in points],
                    color=COLORS[implementation],
                    alpha=0.16,
                )
            axis.set(xscale="log", yscale="log", title=f"{family} · {count} curve{'s' if count > 1 else ''}")
            axis.grid(alpha=0.2)
            if row == 1:
                axis.set_xlabel("Batch size")
            if column == 0:
                axis.set_ylabel("Milliseconds per call")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"curves.{extension}", dpi=200)
    plt.close(figure)


def formatted_pair(results: dict, coefficients: int, batch: int, device: str) -> tuple[str, str]:
    torchcurves = results.get((coefficients, batch, device, "TorchCurves"))
    baseline = results.get((coefficients, batch, device, "efficient-kan"))
    if torchcurves is None or baseline is None:
        return "N/A", "N/A"

    def format_time(item: dict, faster: bool) -> str:
        value = f"{item['median_ms']:.3f}"
        return f"**{value}**" if faster else value

    return (
        format_time(torchcurves, torchcurves["q3_ms"] < baseline["q1_ms"]),
        format_time(baseline, baseline["q3_ms"] < torchcurves["q1_ms"]),
    )


def table_rows(results: list[dict]) -> list[tuple[str, ...]]:
    lookup = {(item["coefficients"], item["batch"], item["device"], item["implementation"]): item for item in results}
    rows = []
    for coefficients in (8, 32):
        for batch in (64, 256, 1024):
            rows.append(
                (
                    str(coefficients),
                    str(batch),
                    *formatted_pair(lookup, coefficients, batch, "cpu"),
                    *formatted_pair(lookup, coefficients, batch, "cuda"),
                )
            )
    return rows


def table_caption(metadata: dict, example: dict) -> str:
    return (
        f"Median milliseconds per training step for a [32, 64, 64, 1] cubic KAN; "
        f"5 blocks of {example['steps_per_block']} steps after {example['warmup_steps']} warm-up steps. "
        "Both models use matched parameters and knots, input clamping, SiLU branches, trainable spline scaling, "
        "MSE, and Adam. Float32 eager execution; one CPU thread. "
        f"CPU: {metadata['cpu']}; GPU: {metadata['gpu'] or 'N/A (CUDA unavailable)'}. "
        "Bold indicates non-overlapping interquartile intervals. Lower is better."
    )


def write_table(results: list[dict], metadata: dict, directory: Path) -> None:
    rows = table_rows(results)
    caption = table_caption(metadata, results[0])
    markdown = ["| " + " | ".join(HEADER) + " |", "|" + " --- |" * len(HEADER)]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    (directory / "kan_table.md").write_text("\n".join(markdown) + "\n\n" + caption + "\n")

    def latex_cell(value: str) -> str:
        return r"\textbf{" + value.strip("*") + "}" if value.startswith("**") else value

    line_break = " " + "\\" * 2
    latex = [r"\begin{tabular}{rrrrrr}", r"\hline", " & ".join(HEADER) + line_break, r"\hline"]
    latex.extend(" & ".join(latex_cell(value) for value in row) + line_break for row in rows)
    latex.extend((r"\hline", r"\end{tabular}", "% " + caption))
    (directory / "kan_table.tex").write_text("\n".join(latex) + "\n")


def write_curve_caption(directory: Path) -> None:
    caption = (
        "CPU evaluation of independent vector-valued Legendre and cubic B-spline curves. Outputs have dimension 16; "
        "each scalar output uses 16 coefficients. Float64, one CPU thread. Bands show interquartile ranges over "
        "5 measurements. The SciPy comparison evaluates one vector-valued spline object per curve."
    )
    (directory / "curves_caption.md").write_text(caption + "\n")


def render(directory: Path) -> None:
    curves = json.loads((directory / "curves_summary.json").read_text())
    kan = json.loads((directory / "kan_summary.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    plot_curves(curves, directory)
    write_table(kan, metadata, directory)
    write_curve_caption(directory)
