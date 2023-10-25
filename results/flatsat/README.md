## FlatSat Results
Experiment results on the engineering model of the spacecraft, i.e. the flatsat.

- **[ae_and_wgan_fpn50_cfpn50_p6_01](./ae_and_wgan_fpn50_cfpn50_p6_01):** Autoencoder and WGAN runs \#1 for fixed-pattern noise (FPN) and column fixed-pattern noise (cFPN) noising-denoising at noise factor 50. The WGAN runs used a denoising patched margin of 6 pixels.
- **[wgan_fpn50_p0-8_01](./wgan_fpn50_p0-8_01):** WGAN runs \#1 for FPN noising-denoising at noise factor 50 for denoising patched margin increments of 2 pixels from 0 to 8 pixels.
- **[wgan_fpn50_p6_01](./wgan_fpn50_p6_01):** WGAN runs \#1 for FPN noising-denoising at noise factor 50 with denoising patched margin of 6 pixels.
- **[wgan_fpn50_p6_02](./wgan_fpn50_p6_02):** WGAN runs \#2 for FPN noising-denoising at noise factor 50 with denoising patched margin of 6 pixels.

Execution times in the `metrics.csv` files were measured using the `time` command to run the experiment executables. To run the python scripts that calculate the image similarity metrics and plots the image histograms:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 calculate_metrics.py --reference_folder_path wgan_fpn50_p0-8_01 --noise fpn50
python3 plot_histograms.py --reference_folder_path wgan_fpn50_p0-8_01 --noise_image fpn50.p6 --output_three_channels histogram_rgb_denoised_wgan_p6.svg --output_final histogram_grayscale_original_and_denoised_wgan_p0-8.svg
```

Where:
- --noise_image is related to the files in the `images` folder 
- --output_three_channels and --output_final will be stored in the folder `figures`

The `opencv-python` package is used to plot the grayscale histograms. The libGL.so library is required by OpenCV. To install it in Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```