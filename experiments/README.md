# Paper timing experiments

This is an independent uv project. Its dependencies and lockfile do not change
the library's package metadata. The `efficient-kan` source is pinned to commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`.

From this directory, run the tests and full study with:

```sh
make run
```

`make render` regenerates the figures and tables from saved measurements.

The tests cover evaluator values, KAN training and Adam updates,
restoration of training blocks, heatmap ratios and layout, and result exports.
`full` runs every specified configuration and writes raw
measurements, summaries, metadata, the curve figures, and the KAN table to
`results/full/`. `render` rebuilds the figures and tables from saved summaries.
The full run can take a long time, especially the larger KAN configurations.
CUDA results are marked unavailable when no CUDA device is accessible.

The curve study measures the Cartesian product of four dimensions for each family:

| Dimension | Values |
| --- | --- |
| Batch size B | 1, 256, 2048 |
| Curve count C | 1, 16, 64 |
| Output dimension D | 1, 16, 64 |
| Coefficients per scalar output K | 4, 16, 64 |

That gives 81 configurations per family, each measured with TorchCurves and
NumPy (Legendre) or SciPy (B-splines). Legendre degree is K - 1. B-splines remain
cubic, with K + 4 entries in the knot vector including repeated boundary knots.
Timing covers complete evaluator calls, including SciPy's loop over one
vector-valued spline per curve. Inputs and evaluator objects are constructed
outside timing. All curve measurements use float64 and one CPU thread.

`legendre.pdf` and `bspline.pdf` (also PNG) contain separate annotated heatmaps.
Panel rows show curve count and panel columns show output dimension; within each
panel, rows show batch size and columns show coefficient count. Cell values are
median TorchCurves time divided by median reference time, rounded to two
significant digits. Below 1 means TorchCurves is faster. Both figures use the same
logarithmic color scale, centered at 1 and spanning 0.01–100; numbers retain ratios
outside that range. Each figure has a matching `_caption.md` file. Raw timings and
summaries retain absolute milliseconds and all four dimensions.

Rerun `make run` to collect the expanded grid; earlier measurements have only one
output dimension and coefficient count.

The KAN study compares each library's idiomatic construction:

- TorchCurves: the README's `BSplineCurve`/`Sum` network with `real.clamp`.
- EfficientKAN: upstream `KAN`, using its default branches, scaling,
  initialization, and extended knots, with no additional input clamp.

Both use widths `[32, 64, 64, 1]`, cubic splines, and 8 or 32 coefficients per
spline (`grid_size=coefficients-3` for EfficientKAN). The architectures,
initial parameters, knot vectors, and parameter counts intentionally differ.
Parameter counts are recorded alongside the timings.

Each step includes forward evaluation, MSE, backward propagation, and Adam.
Both networks reuse the same generated batch, fixed grids, and Adam options;
grid adaptation and regularization are disabled. Timings describe step cost,
not training quality or convergence. Each run records the local commit and
dirty state, hardware, versions, and raw repetitions. Run on a quiet machine
for publication-quality timings.
