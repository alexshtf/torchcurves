| Coefficients | Batch | CPU TorchCurves | CPU efficient-kan | GPU TorchCurves | GPU efficient-kan |
| --- | --- | --- | --- | --- | --- |
| 8 | 64 | 4.982 | **2.184** | N/A | N/A |
| 8 | 256 | 29.225 | **6.514** | N/A | N/A |
| 8 | 1024 | 150.554 | **38.664** | N/A | N/A |
| 32 | 64 | 6.004 | **5.058** | N/A | N/A |
| 32 | 256 | 30.741 | 30.812 | N/A | N/A |
| 32 | 1024 | 157.496 | **139.563** | N/A | N/A |

Median milliseconds per training step for a [32, 64, 64, 1] cubic KAN; 5 blocks of 100 steps after 50 warm-up steps. Both models use matched parameters and knots, input clamping, SiLU branches, trainable spline scaling, MSE, and Adam. Float32 eager execution; one CPU thread. CPU: Intel(R) Core(TM) Ultra 7 356H; GPU: N/A (CUDA unavailable). Bold indicates non-overlapping interquartile intervals. Lower is better.
