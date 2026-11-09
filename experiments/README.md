# Paper timing experiments

This is an independent uv project. Its dependencies and lockfile do not change
the library's package metadata. The `efficient-kan` source is pinned to commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`.

From this directory, run the tests and full study with:

```sh
make run
```

`make render` regenerates the figure and tables from saved measurements.

The tests cover evaluator values, KAN training and Adam updates,
restoration of training blocks, and result exports.
`full` runs every specified configuration and writes raw
measurements, summaries, metadata, the curve figure, and the KAN table to
`results/full/`. `render` rebuilds the figure and tables from saved summaries.
The full run can take a long time, especially the larger KAN configurations.
CUDA results are marked unavailable when no CUDA device is accessible.

The curve figure compares complete evaluator calls, including SciPy's loop over
one vector-valued spline per curve.

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
