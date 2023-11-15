# Results

- Results from running the experiment on both the flatsat and the on-board spacecraft.
- The "extra" directory just contains scripts to generate figures for the paper. 

## Plot the results

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 plot_histograms.py --image_folder_path ./flatsat/wgan_fpn50_p0-8_01 --reference_folder_path flatsat/wgan_fpn50_p0-8_01 --noise_image fpn50.p6 --output_three_channels histogram_rgb_denoised_wgan_p6.svg --output_final histogram_grayscale_original_and_denoised_wgan_p0-8.svg --caption_histogram "WGAN Denoised (margin is 6 pixels)"

python3 plot_histograms.py --image_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1695963889066 --reference_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1695963889066 --noise_image fpn50 --output_three_channels histogram_rgb_denoised_wgan_1695963889066.svg --output_final histogram_grayscale_original_and_denoised_wgan_1695963889066.svg --caption_histogram "WGAN Denoised"

python3 plot_histograms.py --image_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1695964476824 --reference_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1695964476824 --noise_image fpn50 --output_three_channels histogram_rgb_denoised_wgan_1695964476824.svg --output_final histogram_grayscale_original_and_denoised_wgan_1695964476824.svg --caption_histogram "WGAN Denoised"

python3 plot_histograms.py --image_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1697455926224 --reference_folder_path ./spacecraft/images/WGAN_FPN50/Earth/1697455926224 --noise_image fpn50 --output_three_channels histogram_rgb_denoised_wgan_1697455926224.svg --output_final histogram_grayscale_original_and_denoised_wgan_1697455926224.svg --caption_histogram "WGAN Denoised"
```

Where:
- --noise_image is related to the files in the `images` folder 
- --output_three_channels and --output_final will be stored in the folder `figures`

The `opencv-python` package is used to plot the grayscale histograms. The libGL.so library is required by OpenCV. To install it in Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx
sudo apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
```