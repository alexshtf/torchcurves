"""Create the paper figure and tables from saved measurements."""

import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.style.use("tableau-colorblind10")

HEADER = ("Coefficients", "Batch", "CPU TorchCurves", "CPU efficient-kan", "GPU TorchCurves", "GPU efficient-kan")


def draw_heatmap(axis, ratios: np.ndarray, batches: list[int], coefficients: list[int]):
    image = axis.imshow(ratios, cmap="RdBu_r", norm=LogNorm(0.01, 100))
    for (row, column), value in np.ndenumerate(ratios):
        axis.text(
            column,
            row,
            f"{value:.2g}",
            ha="center",
            va="center",
            color="white" if value < 0.1 or value > 10 else "black",
        )
    axis.set_xticks(range(len(coefficients)), coefficients)
    axis.set_yticks(range(len(batches)), batches)
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def curve_figure(results: list[dict], family: str):
    """Place every configuration in a curve-count × output-dimension atlas."""
    reference = "NumPy" if family == "Legendre" else "SciPy"
    results = [item for item in results if item["family"] == family]
    counts, dimensions, batches, coefficients = (
        sorted({item[key] for item in results}) for key in ("curves", "outputs", "batch", "coefficients")
    )
    times = {
        (item["curves"], item["outputs"], item["batch"], item["coefficients"], item["implementation"]): item[
            "median_ms"
        ]
        for item in results
    }
    figure, axes = plt.subplots(len(counts), len(dimensions), figsize=(9, 8), squeeze=False, layout="constrained")
    for row, count in enumerate(counts):
        for column, dimension in enumerate(dimensions):
            ratios = np.array(
                [
                    [
                        times[count, dimension, batch, size, "TorchCurves"]
                        / times[count, dimension, batch, size, reference]
                        for size in coefficients
                    ]
                    for batch in batches
                ]
            )
            axis = axes[row, column]
            image = draw_heatmap(axis, ratios, batches, coefficients)
            if row == 0:
                axis.set_title(f"Dimension D = {dimension}")
            if row == len(counts) - 1:
                axis.set_xlabel("Coefficients K")
            if column == 0:
                axis.set_ylabel(f"Curves C = {count}\nBatch size B")
            else:
                axis.tick_params(labelleft=False)

    figure.suptitle(f"{family} curves · TorchCurves vs {reference}")
    colorbar = figure.colorbar(
        image, ax=axes, location="bottom", shrink=0.8, aspect=40, extend="both", ticks=[0.01, 0.1, 1, 10, 100]
    )
    colorbar.ax.set_xticklabels(["0.01×", "0.1×", "1×", "10×", "100×"])
    colorbar.minorticks_off()
    colorbar.set_label(
        f"Median TorchCurves time / median {reference} time\nBelow 1: faster · 1: equal · above 1: slower"
    )
    return figure


def plot_curves(results: list[dict], directory: Path) -> None:
    for family, stem in (("Legendre", "legendre"), ("B-spline", "bspline")):
        figure = curve_figure(results, family)
        for extension in ("pdf", "png"):
            figure.savefig(directory / f"{stem}.{extension}", dpi=200)
        plt.close(figure)
        (directory / f"{stem}_caption.md").write_text(curve_caption(family) + "\n")


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


def curve_caption(family: str) -> str:
    representation = (
        "Legendre expansions have degree K - 1 and are compared with NumPy."
        if family == "Legendre"
        else "Cubic B-splines have K + 4 knots, counting repeated boundary knots, and are compared with SciPy. "
        "SciPy evaluates one vector-valued spline object per curve."
    )
    return (
        "CPU evaluation of independent vector-valued curves. Panel rows show curve count C; panel columns show "
        "output dimension D. Within each panel, rows show batch size B and columns show coefficients per scalar "
        "output K. Each cell is median TorchCurves time divided by median reference time, rounded to two significant "
        "digits; values below 1 mean TorchCurves is faster. Medians use five timing measurements. "
        "The logarithmic color scale is shared across both figures and centered at 1; ratios outside [0.01, 100] "
        "use the endpoint colors. Float64, one CPU thread; input and evaluator construction are excluded from timing. "
        + representation
    )


def render(directory: Path) -> None:
    curves = json.loads((directory / "curves_summary.json").read_text())
    kan = json.loads((directory / "kan_summary.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    plot_curves(curves, directory)
    write_table(kan, metadata, directory)
