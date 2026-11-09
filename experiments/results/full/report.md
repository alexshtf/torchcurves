# Timing study run

These saved measurements include the full CPU curve grid and all six KAN
configurations on CPU and CUDA. They were collected on an Intel Core Ultra 7
356H and an NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. The source
commit, package versions, thread settings, and CUDA build are recorded in
[metadata.json](metadata.json).

The KAN timings predate the fix to Adam state restoration between blocks.
Although model weights were restored, Adam could mutate the saved optimizer
state, so later blocks did not repeat the same training trajectory. These
measurements are retained for reference; rerun `make run` before using the
KAN table in the paper.

The curve results are unaffected by that state-restoration issue. Their
outputs were checked against NumPy or SciPy at `rtol=1e-10, atol=1e-12` before
timing. The study measures evaluator-call and training-step cost; it does not
measure prediction quality or convergence.
