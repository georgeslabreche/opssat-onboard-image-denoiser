## FlatSat Results
Experiment results on the engineering model of the spacecraft, i.e. the flatsat.

- **[ae_and_wgan_fnp50_cfnp50_p6_01](./ae_and_wgan_fnp50_cfnp50_p6_01):** Autoencoder and WGAN runs \#1 for fixed-noise pattern (FNP) and column fixed-noise pattern (CFNP) noising-denoising at noise factor 50. The WGAN runs used a denoising patched margin of 6 pixels.
- **[wgan_fnp50_p0-8_01](./wgan_fnp50_p0-8_01):** WGAN runs \#1 for fixed-noise pattern (FNP) noising-denoising at noise factor 50 for denoising patched margin increments of 2 pixels from 0 to 8 pixels.
- **[wgan_fnp50_p6_01](./wgan_fnp50_p6_01):** WGAN runs \#1 for fixed-noise pattern (FNP) noising-denoising at noise factor 50 with denoising patched margin of 6 pixels.
- **[wgan_fnp50_p6_02](./wgan_fnp50_p6_02):** WGAN runs \#2 for fixed-noise pattern (FNP) noising-denoising at noise factor 50 with denoising patched margin of 6 pixels.

Execution times in the `metrics.csv` files were measured using the `time` command to run the experiment executables. To run the python scripts that calculate the image similarity metrics and plots the image histograms:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python calculate_metrics.py
python plot_histograms.py
```

The `opencv-python` package is used to plot the grayscale histograms. The libGL.so library is required by OpenCV. To install it in Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```