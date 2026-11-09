# Timing study run

The November 9, 2026 run covers the full CPU curve grid and all six KAN
configurations on both CPU and CUDA. Hardware: Intel Core Ultra 7 356H and
NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. Package versions, thread
settings, CUDA build, and source checkout details are in
[metadata.json](metadata.json). The run used the working-tree changes to the
recorded commit.

The KAN study uses each library's idiomatic network: upstream EfficientKAN
`KAN`, and TorchCurves `BSplineCurve`/`Sum` layers with `real.clamp`. Both have
widths `[32, 64, 64, 1]` and cubic splines. Their branches, knots, input
handling, and initialization intentionally differ. Trainable parameter counts
are:

| Coefficients per spline | TorchCurves | EfficientKAN |
| --- | ---: | ---: |
| 8 | 49,664 | 62,080 |
| 32 | 198,656 | 211,072 |

Both models train on the same generated batch using MSE and Adam. Each timing
is the median of five blocks of 100 steps after 50 warm-up steps. Every block
restores independent copies of the warmed model and optimizer state; execution
order alternates, and CUDA is synchronized around timing. Execution uses
float32, fixed grids, no regularization, and one CPU thread.

EfficientKAN has lower median step time in five of six configurations on each
device. TorchCurves is faster at 32 coefficients and batch 256 on CPU
(19.885 vs 21.091 ms), and at 32 coefficients and batch 1024 on CUDA
(7.230 vs 10.600 ms). Full timings and the interquartile-interval comparison
are in [kan_table.md](kan_table.md). These measurements describe training-step
cost, not predictive quality or convergence.

All 15 experiment tests passed, including CPU and CUDA training, parameter
updates, and restoration of repeated training blocks. Curve outputs were also
checked against NumPy or SciPy at `rtol=1e-10, atol=1e-12` before timing. The
saved data contain 160 curve measurements and 120 KAN measurements, with
32 and 24 summaries respectively.
