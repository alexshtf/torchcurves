"""Create annotated paper heatmaps from saved measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.style.use("tableau-colorblind10")


def draw_heatmap(axis, ratios: np.ndarray, row_values: list[int], column_values: list[int]):
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
    axis.set_xticks(range(len(column_values)), column_values)
    axis.set_yticks(range(len(row_values)), row_values)
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def add_ratio_colorbar(figure, axes, image, reference: str) -> None:
    colorbar = figure.colorbar(
        image, ax=axes, location="bottom", shrink=0.8, aspect=40, extend="both", ticks=[0.01, 0.1, 1, 10, 100]
    )
    colorbar.ax.set_xticklabels(["0.01×", "0.1×", "1×", "10×", "100×"])
    colorbar.minorticks_off()
    colorbar.set_label(
        f"Median TorchCurves time / median {reference} time\nBelow 1: faster · 1: equal · above 1: slower"
    )


def save_figure(figure, directory: Path, stem: str, caption: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"{stem}.{extension}", dpi=200)
    plt.close(figure)
    (directory / f"{stem}_caption.md").write_text(caption + "\n")


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
    add_ratio_colorbar(figure, axes, image, reference)
    return figure


def plot_curves(results: list[dict], directory: Path) -> None:
    for family, stem in (("Legendre", "legendre"), ("B-spline", "bspline")):
        figure = curve_figure(results, family)
        save_figure(figure, directory, stem, curve_caption(family))


def kan_figure(results: list[dict]):
    """Show devices × coefficient counts, with width × batch size in each panel."""
    widths, batches, coefficients = (
        sorted({item[key] for item in results}) for key in ("hidden_width", "batch", "coefficients")
    )
    devices = {item["device"] for item in results}
    times = {
        (item["device"], item["coefficients"], item["hidden_width"], item["batch"], item["implementation"]): item[
            "median_ms"
        ]
        for item in results
    }
    figure, axes = plt.subplots(2, len(coefficients), figsize=(9, 7), squeeze=False, layout="constrained")
    for row, (device, label) in enumerate((("cpu", "CPU"), ("cuda", "GPU"))):
        for column, size in enumerate(coefficients):
            axis = axes[row, column]
            axis.set_box_aspect(len(widths) / len(batches))
            if device in devices:
                ratios = np.array(
                    [
                        [
                            times[device, size, width, batch, "TorchCurves"]
                            / times[device, size, width, batch, "efficient-kan"]
                            for batch in batches
                        ]
                        for width in widths
                    ]
                )
                image = draw_heatmap(axis, ratios, widths, batches)
            else:
                axis.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=axis.transAxes)
                axis.set(xticks=[], yticks=[], frame_on=False)
            if row == 0:
                axis.set_title(f"Coefficients K = {size}")
            if row == 1:
                axis.set_xlabel("Batch size B")
            if column == 0:
                axis.set_ylabel(f"{label}\nHidden width W")
            else:
                axis.tick_params(labelleft=False)

    figure.suptitle("KAN training steps · TorchCurves vs EfficientKAN")
    add_ratio_colorbar(figure, axes, image, "EfficientKAN")
    return figure


def kan_caption(metadata: dict, example: dict) -> str:
    return (
        "Training-step time ratios for idiomatic [32, W, W, 1] cubic KANs. Panel rows show CPU and GPU; "
        "panel columns show coefficient count K. Within each panel, rows show hidden width W and columns show "
        "batch size B. Each cell is median TorchCurves time divided by median EfficientKAN time, rounded to "
        "two significant digits; below 1 means TorchCurves is faster. The shared logarithmic color scale spans "
        "0.01–100, centered at 1; ratios beyond the limits use endpoint colors. "
        f"5 blocks of {example['steps_per_block']} steps after {example['warmup_steps']} warm-up steps. "
        "TorchCurves uses BSplineCurve/Sum layers with the default rational input map and repeated boundary knots. "
        "EfficientKAN uses its KAN model with SiLU branches, trainable spline scaling, "
        "extended knots, and no added clamp. "
        "Each library initializes its own parameters. Architectures and parameter counts differ. "
        "Both use the same generated data, MSE, and Adam with fixed grids. Float32 eager execution; one CPU thread. "
        f"CPU: {metadata['cpu']}; GPU: {metadata['gpu'] or 'N/A (CUDA unavailable)'}. "
        "These timings describe step cost, not predictive quality or convergence."
    )


def plot_kan(results: list[dict], metadata: dict, directory: Path) -> None:
    save_figure(kan_figure(results), directory, "kan", kan_caption(metadata, results[0]))


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
    plot_kan(kan, metadata, directory)
