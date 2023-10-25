# Results

- Results from running the experiment on both the flatsat and the on-board spacecraft.
- The "extra" directory just contains scripts to generate figures for the paper. 

## Plot the results

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 plot_histograms.py --reference_folder_path flatsat/wgan_fpn50_p0-8_01 --noise_image fpn50.p6 --output_three_channels histogram_rgb_denoised_wgan_p6.svg --output_final histogram_grayscale_original_and_denoised_wgan_p0-8.svg --caption_histogram WGAN

python3 plot_histograms.py --reference_folder_path spacecraft/images/AE/FPN-50/Earth --noise_image fpn50 --output_three_channels histogram_rgb_denoised_AE.svg --output_final histogram_grayscale_original_and_denoised_AE.svg --caption_histogram Autoencoder
```

Where:
- --noise_image is related to the files in the `images` folder 
- --output_three_channels and --output_final will be stored in the folder `figures`

The `opencv-python` package is used to plot the grayscale histograms. The libGL.so library is required by OpenCV. To install it in Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```