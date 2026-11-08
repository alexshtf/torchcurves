| Coefficients | Batch | CPU TorchCurves | CPU efficient-kan | GPU TorchCurves | GPU efficient-kan |
| --- | --- | --- | --- | --- | --- |
| 8 | 64 | 4.954 | **2.181** | 5.845 | 6.418 |
| 8 | 256 | 28.285 | **6.367** | 5.808 | 5.684 |
| 8 | 1024 | 149.437 | **38.735** | 7.139 | **4.795** |
| 32 | 64 | 5.677 | **4.986** | 5.896 | 5.702 |
| 32 | 256 | **19.311** | 20.793 | 5.005 | **3.985** |
| 32 | 1024 | 135.513 | **126.345** | **7.349** | 10.553 |

Median milliseconds per training step for a [32, 64, 64, 1] cubic KAN; 5 blocks of 100 steps after 50 warm-up steps. Both models use matched parameters and knots, input clamping, SiLU branches, trainable spline scaling, MSE, and Adam. Float32 eager execution; one CPU thread. CPU: Intel(R) Core(TM) Ultra 7 356H; GPU: NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. Bold indicates non-overlapping interquartile intervals. Lower is better.
