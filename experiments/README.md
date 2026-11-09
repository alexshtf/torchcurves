# Paper timing experiments

This is an independent uv project. Its dependencies and lockfile do not change
the library's package metadata. The `efficient-kan` source is pinned to commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`.

From this directory, run the tests and full study with:

```sh
make run AFFINITY=4
```

`AFFINITY=k` pins the test and experiment processes to logical CPUs `0` through
`k-1` using Linux `taskset`, and limits the numerical thread pools to `k` threads.
The default is 1. Make exports `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` before Python starts; Python does not override them.
PyTorch's benchmark timer inherits the effective thread count instead of its
usual one-thread default. Metadata records the actual CPU affinity and thread
settings. Choose CPUs available to the process; this requires Linux and `taskset`.

`make test AFFINITY=4` runs only the tests. `make render` regenerates the figures
from saved measurements, using the recorded thread count in their captions.

Each study displays one terminal progress bar with completed configurations,
elapsed time, the current configuration, and its implementation and timing phase.
Progress updates happen outside timed measurements. No ETA is shown; bars are
disabled when output is not a terminal.

The tests cover evaluator values, KAN training and Adam updates,
restoration of training blocks, heatmap ratios and layout, and result exports.
`full` runs every specified configuration and writes raw
measurements, summaries, metadata, and the three heatmap figures to
`results/full/`. `render` rebuilds the figures from saved summaries.
The full run can take a long time, especially the larger KAN configurations.
GPU panels are marked "Not measured" when no CUDA device is accessible.

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
outside timing. All curve measurements use float64 and the configured CPU thread limit.

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

- TorchCurves: a `BSplineCurve`/`Sum` network with the default `real.rational` input map.
- EfficientKAN: upstream `KAN`, using its default branches, scaling,
  initialization, and extended knots, with no additional input clamp.

Both use widths `[32, W, W, 1]` and cubic splines. The grid covers all combinations
of batch size B = 64, 256, 1024; hidden width W = 32, 64, 128; and coefficients
per spline K = 8, 32, 64 (`grid_size=K-3` for EfficientKAN). This gives 27
configurations per device, or 54 with both CPU and CUDA. Input/output dimensions
and the three-layer depth stay fixed. The architectures,
initial parameters, knot vectors, and parameter counts intentionally differ.
Hidden widths and parameter counts are recorded alongside the timings.

`kan.pdf` and `kan.png` show CPU/GPU as panel rows and coefficient counts as panel
columns. Within each panel, rows show hidden width and columns show batch size.
Each number is median TorchCurves step time divided by median EfficientKAN step
time. The heatmap uses the same color scale as the curve figures, with a caption
in `kan_caption.md`. It replaces the KAN table; absolute timings remain in CSV/JSON.
Earlier KAN measurements lack the hidden-width field and need to be rerun.

Each step includes forward evaluation, MSE, backward propagation, and Adam.
Both networks reuse the same generated batch, fixed grids, and Adam options;
grid adaptation and regularization are disabled. Timings describe step cost,
not training quality or convergence. Each run records the local commit and
dirty state, hardware, versions, and raw repetitions. Run on a quiet machine
for publication-quality timings.
