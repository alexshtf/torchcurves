# Timing study run

The full CPU curve grid and all six CPU KAN configurations completed. Before
timing, the curve outputs matched NumPy or SciPy at `rtol=1e-10, atol=1e-12`.
The single-layer and full-network KAN checks passed for outputs, MSE, input
gradients, parameter gradients, and one matched Adam update in CPU float64 and
float32. No non-finite training parameters or gradients were detected.

CUDA was unavailable, so GPU timings are N/A. The CPU was an Intel Core Ultra 7
356H, with one numerical-library and PyTorch CPU thread. Package versions,
the pinned efficient-kan commit, the local TorchCurves commit and dirty state,
and the CUDA build version are in [metadata.json](metadata.json).

The CPU results are mixed. At batch 2048 with one Legendre curve, TorchCurves
took 0.248 ms per call versus NumPy's 0.838 ms; with 32 curves it took 51.3 ms
versus NumPy's 37.1 ms. SciPy was faster than TorchCurves for most cubic
B-spline configurations, including 5.47 ms versus 20.6 ms at batch 2048 and
32 curves. In the matched KAN study, efficient-kan was faster in five of six
CPU rows by median; at 32 coefficients and batch 256, the medians were nearly
equal and their interquartile intervals overlapped.

These measurements describe evaluator-call and training-step cost for the
specified generated inputs. They do not measure prediction quality or
convergence. The TorchCurves checkout had uncommitted changes during the run;
rerun from a clean commit before treating these timings as final paper data.
