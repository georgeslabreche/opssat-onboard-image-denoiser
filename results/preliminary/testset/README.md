## Preliminary Results

Preliminary results on the test set reveal that Autoencoders perform marginally better in denoising FPN when trained with full images while WGANs perform significantly better when trained with patched images.

| Model               | PSNR      | SSIM    | MSE        |
|---------------------|-----------|---------|------------|
| Autoencoder Full    | 32.011851 | 0.956779| 56.626925  |
| Autoencoder Patch   | 33.374308 | 0.952792| 53.447579  |
| WGANs Full          | 25.499213 | 0.493948| 238.529979 |
| WGANs Patch         | 29.222343 | 0.871755| 410.838258 |

The averages were calculated using the [`calculate_metrics_averages.py`](./calculate_metrics_averages.py) script.