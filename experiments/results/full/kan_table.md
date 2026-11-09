| Coefficients | Batch | CPU TorchCurves | CPU efficient-kan | GPU TorchCurves | GPU efficient-kan |
| --- | --- | --- | --- | --- | --- |
| 8 | 64 | 4.283 | **1.950** | 3.443 | **2.905** |
| 8 | 256 | 24.560 | **5.718** | 3.458 | **2.912** |
| 8 | 1024 | 134.216 | **26.328** | 7.028 | **2.922** |
| 32 | 64 | 4.969 | **4.466** | 3.457 | **2.899** |
| 32 | 256 | **17.934** | 19.225 | 3.464 | **2.900** |
| 32 | 1024 | 139.166 | **95.806** | **7.225** | 10.455 |

Median milliseconds per training step for idiomatic [32, 64, 64, 1] cubic KANs; 5 blocks of 100 steps after 50 warm-up steps. TorchCurves uses BSplineCurve/Sum layers with clamping and repeated boundary knots. EfficientKAN uses its KAN model with SiLU branches, trainable spline scaling, extended knots, and no added clamp. Each library initializes its own parameters. Architectures and parameter counts differ. Both use the same generated data, MSE, and Adam with fixed grids. Float32 eager execution; one CPU thread. CPU: Intel(R) Core(TM) Ultra 7 356H; GPU: NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. Bold indicates non-overlapping interquartile intervals. Lower is better. These timings describe step cost, not predictive quality or convergence.
