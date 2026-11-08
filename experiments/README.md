# Paper timing experiments

This is an independent uv project. Its dependencies and lockfile do not change
the library's package metadata. The `efficient-kan` source is pinned to commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`.

From this directory, run the tests and full study with:

```sh
make run
```

`make render` regenerates the figure and tables from saved measurements.

The tests check the evaluator values and matched KAN gradients without timing.
`full` runs every specified configuration and writes raw
measurements, summaries, metadata, the curve figure, and the KAN table to
`results/full/`. `render` rebuilds the figure and tables from saved summaries.
The full run can take a long time, especially the larger KAN configurations.
CUDA results are marked unavailable when no CUDA device is accessible.

The curve figure compares complete evaluator calls, including SciPy's loop over
one vector-valued spline per curve. The KAN timings include input clamping,
forward evaluation, MSE, backward propagation, and Adam updates. They measure
step cost, not training quality or convergence. Each run records the local
commit and dirty state, hardware, versions, and raw repetitions. Run on a
quiet machine for publication-quality timings.
