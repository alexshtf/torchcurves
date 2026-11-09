"""Create the paper figure and tables from saved measurements."""

import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")

HEADER = ("Coefficients", "Batch", "CPU TorchCurves", "CPU efficient-kan", "GPU TorchCurves", "GPU efficient-kan")


def plot_curves(results: list[dict], directory: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
    panels = product(("Legendre", "B-spline"), (1, 32))

    for axis, (family, count) in zip(axes.flat, panels, strict=True):
        reference = "NumPy" if family == "Legendre" else "SciPy"
        for implementation in ("TorchCurves", reference):
            points = sorted(
                (
                    item
                    for item in results
                    if (item["family"], item["curves"], item["implementation"]) == (family, count, implementation)
                ),
                key=lambda item: item["batch"],
            )
            batches = [item["batch"] for item in points]
            (line,) = axis.plot(batches, [item["median_ms"] for item in points], marker="o", label=implementation)
            axis.fill_between(
                batches,
                [item["q1_ms"] for item in points],
                [item["q3_ms"] for item in points],
                color=line.get_color(),
                alpha=0.16,
            )
        axis.set(xscale="log", yscale="log", title=f"{family} · {count} curve{'s' if count > 1 else ''}")
        axis.grid(alpha=0.2)

    figure.supxlabel("Batch size")
    figure.supylabel("Milliseconds per call")
    axes[0, 0].legend(frameon=False)
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"curves.{extension}", dpi=200)
    plt.close(figure)


def format_time(item: dict, other: dict) -> str:
    value = f"{item['median_ms']:.3f}"
    return f"**{value}**" if item["q3_ms"] < other["q1_ms"] else value


def formatted_pair(results: dict, coefficients: int, batch: int, device: str) -> tuple[str, str]:
    torchcurves = results.get((coefficients, batch, device, "TorchCurves"))
    baseline = results.get((coefficients, batch, device, "efficient-kan"))
    if torchcurves is None or baseline is None:
        return "N/A", "N/A"

    return format_time(torchcurves, baseline), format_time(baseline, torchcurves)


def table_rows(results: list[dict]) -> list[tuple[str, ...]]:
    lookup = {(item["coefficients"], item["batch"], item["device"], item["implementation"]): item for item in results}
    rows = []
    for coefficients, batch in product((8, 32), (64, 256, 1024)):
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
        "Median milliseconds per training step for idiomatic [32, 64, 64, 1] cubic KANs; "
        f"5 blocks of {example['steps_per_block']} steps after {example['warmup_steps']} warm-up steps. "
        "TorchCurves uses BSplineCurve/Sum layers with clamping and repeated boundary knots. "
        "EfficientKAN uses its KAN model with SiLU branches, trainable spline scaling, "
        "extended knots, and no added clamp. "
        "Each library initializes its own parameters. Architectures and parameter counts differ. "
        "Both use the same generated data, MSE, and Adam with fixed grids. Float32 eager execution; one CPU thread. "
        f"CPU: {metadata['cpu']}; GPU: {metadata['gpu'] or 'N/A (CUDA unavailable)'}. "
        "Bold indicates non-overlapping interquartile intervals. Lower is better. "
        "These timings describe step cost, not predictive quality or convergence."
    )


def latex_cell(value: str) -> str:
    return r"\textbf{" + value.strip("*") + "}" if value.startswith("**") else value


def write_table(results: list[dict], metadata: dict, directory: Path) -> None:
    rows = table_rows(results)
    caption = table_caption(metadata, results[0])
    markdown = ["| " + " | ".join(HEADER) + " |", "|" + " --- |" * len(HEADER)]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    (directory / "kan_table.md").write_text("\n".join(markdown) + "\n\n" + caption + "\n")

    line_break = r" \\"
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
