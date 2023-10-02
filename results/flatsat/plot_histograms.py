# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from palettable.colorbrewer.diverging import RdYlBu_5

def plot_histogram(image, color, label, linestyle='-'):
  """
  Plot histogram for a given image.
  :param image: Image read by OpenCV
  :param color: Color of histogram curve
  :param label: Label for the curve
  :param linestyle: Type of line
  """
  hist = cv2.calcHist([image], [0], None, [256], [0, 256])
  plt.plot(hist, color=color, label=label, linestyle=linestyle)
  plt.xlim([30, 110])

# Load and resize the reference image
reference_image_path = "wgan_fnp50_p0-8_01/images/sample.jpeg"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.resize(reference_image, (224, 224))

# Plot histogram for the reference image
plt.figure(figsize=(10, 9))
plot_histogram(reference_image, color='black', label='Original Image')

# Read the CSV file
csv_path = "wgan_fnp50_p0-8_01/metrics.csv"
df = pd.read_csv(csv_path)

# Define colors using the RdYlBu palette from palettable
colors = RdYlBu_5.mpl_colors
if len(df['patch_margin_pixels']) > len(colors):
  print("Warning: There are more images than available colors. Colors will repeat.")

# Plot histograms for each pX image
for index, patch_margin_pixels in enumerate(df['patch_margin_pixels']):
  # Derive the denoised image name based on the patch_margin_pixels column
  denoised_image_name = f"sample.fnp50.p{int(patch_margin_pixels)}.denoised.jpeg"
  denoised_image_path = os.path.join("wgan_fnp50_p0-8_01/images", denoised_image_name)

  denoised_image = cv2.imread(denoised_image_path, cv2.IMREAD_GRAYSCALE)

  plot_histogram(denoised_image, color=colors[index % len(colors)], label=f'{int(patch_margin_pixels)} Pixels', linestyle='--')

#plt.title('Histograms')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("wgan_fnp50_p0-8_01/figures/histograms.grayscale.svg", format="svg")
