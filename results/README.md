# Results

- Results from running the experiment on both the flatsat and the on-board spacecraft.
- The "preliminary" folder contains FlatSat results for denoising tests on a sample image.
- The "extra" directory just contains scripts to generate figures for the paper.
- The "blue_dot" folder contains denoising results of "a pale blue dot" executed onboard both the FlatSat and the Spacecraft.

## Image Hisograms
How to produce the grayscale and RGB histograms.

### Install Dependencies
The `opencv-python` package is used to plot the grayscale histograms. The libGL.so library is required by OpenCV. To install it in Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx
sudo apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
```

Set up the Python virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### Generate Histograms
Generate grayscale and RGB histograms for the Autoencoder results from the FlatSat:

#### FlatSat
##### WGANs
```bash
cp flatsat/wgan_fpn50_p0-8_01/images/sample.fpn50.p6.denoised.jpeg flatsat/wgan_fpn50_p0-8_01/images/sample.denoised.jpeg
python plot_histograms.py -i flatsat/wgan_fpn50_p0-8_01/images/sample.jpeg -d flatsat/histograms
rm -f flatsat/wgan_fpn50_p0-8_01/images/sample.denoised.jpeg
```

Crop the x-axis for the figure that will be included in the publication:
```bash
cp flatsat/wgan_fpn50_p0-8_01/images/sample.fpn50.p6.denoised.jpeg flatsat/wgan_fpn50_p0-8_01/images/sample.denoised.jpeg
python plot_histograms.py -i flatsat/wgan_fpn50_p0-8_01/images/sample.jpeg -d flatsat/histograms/publication/ -x0 20 -x1 150
rm -f flatsat/wgan_fpn50_p0-8_01/images/sample.denoised.jpeg
```

#### Spacecraft
##### Autoencoder
```bash
python plot_histograms.py -i spacecraft/images/AE/FPN-50/Earth/ -d spacecraft/histograms/AE/FPN-50/Earth/
python plot_histograms.py -i spacecraft/images/AE/FPN-50/Bad/ -d spacecraft/histograms/AE/FPN-50/Bad/
```


##### WGANs
```bash
python plot_histograms.py -i spacecraft/images/WGAN/FPN-50/Earth/ -d spacecraft/histograms/WGAN/FPN-50/Earth/
```

Crop the x-axis for the figure that will be included in the publication:
```bash
python plot_histograms.py -i spacecraft/images/WGAN/FPN-50/Earth/1697455926224.jpeg -d spacecraft/histograms/publication/ -x0 20 -x1 150
```