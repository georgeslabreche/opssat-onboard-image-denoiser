# -*- coding: utf-8 -*-

import os
import pandas as pd
from skimage.metrics import *
from skimage import io, transform

# Define metric computation functions
def compute_mse(img1, img2):
  return mean_squared_error(img1, img2)

def compute_psnr(img1, img2):
  return peak_signal_noise_ratio(img1, img2)

def compute_ssim(img1, img2):
  return structural_similarity(img1, img2, multichannel=True, data_range=255, win_size=3)

# Load the reference image
reference_image_path = "wgan_fpn50_p0-8_01/images/sample.jpeg"
reference_image = io.imread(reference_image_path)
reference_image = transform.resize(reference_image, (224, 224), anti_aliasing=True)
reference_image = (reference_image * 255).astype('uint8')  # Convert to uint8

# Read the CSV file
csv_path = "wgan_fpn50_p0-8_01/metrics.csv"
df = pd.read_csv(csv_path)

# Calculate metrics for each image and update the CSV
for index, row in df.iterrows():
  patch_margin_pixels = int(row['patch_margin_pixels'])

  # Derive the denoised image name based on the patch_margin_pixels column
  denoised_image_name = f"sample.fpn50.p{patch_margin_pixels}.denoised.jpeg"
  denoised_image_path = os.path.join("wgan_fpn50_p0-8_01/images", denoised_image_name)
  
  # Read the denoised image
  denoised_image = io.imread(denoised_image_path)

  # Compute metrics
  mse = compute_mse(reference_image, denoised_image)
  psnr = compute_psnr(reference_image, denoised_image)
  ssim = compute_ssim(reference_image, denoised_image)

  # Update the CSV
  df.at[index, 'mse'] = mse
  df.at[index, 'psnr'] = psnr
  df.at[index, 'ssim'] = ssim

# Save the updated CSV
df.to_csv(csv_path, index=False)

# Pretty print the updated CSV
print(df)

# Round the values in the dataframe
columns_to_round = ['mse', 'psnr', 'ssim']
df[columns_to_round] = df[columns_to_round].round(3)

# Save the rounded CSV
rounded_csv_path = csv_path.replace(".csv", ".rounded.csv")
df.to_csv(rounded_csv_path, index=False)

# Pretty print the rounded CSV
print(df)