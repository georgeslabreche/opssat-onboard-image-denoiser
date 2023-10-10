# -*- coding: utf-8 -*-
"""
@Time : 2023/09/24 19:28
@Auth : Dr. Cesar Guzman
"""
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.metrics import *
from skimage import measure
import numpy as np
from PIL import Image



def plot_histograms_rgb(image_timestamp, original_image_resize_target, image_path_original, image_path_denoised, include_total, normalize_counts, save_fig):
  """Histogram for the RGB pixel values"""

  # Load the images using PIL
  image_original = Image.open(image_path_original)
  image_denoised = Image.open(image_path_denoised)

  if original_image_resize_target is not None:
    new_dimensions  = (original_image_resize_target, original_image_resize_target)
    image_original = image_original.resize(new_dimensions)

  # Convert the images into arrays
  data_original = np.array(image_original)
  data_denoised = np.array(image_denoised)

  # Plot histograms for each image
  plt.figure(figsize=(15, 7))

  for idx, data in enumerate([data_original, data_denoised], start=1):
    plt.subplot(1, 2, idx)

    # Extract the RGB channels
    red_channel = data[:, :, 0]
    green_channel = data[:, :, 1]
    blue_channel = data[:, :, 2]
    
    # Compute grayscale image (luminance) from RGB
    # This will be used as the histogram for the total (a.k.a. the grayscale or luminance representation)
    grayscale_image = 0.2989 * red_channel + 0.5870 * green_channel + 0.1140 * blue_channel

    # Plotting the histograms
    plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.5, rwidth=0.8, density=normalize_counts, label='Red Channel')
    plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.5, rwidth=0.8, density=normalize_counts, label='Green Channel')
    plt.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.5, rwidth=0.8, density=normalize_counts, label='Blue Channel')

    if include_total is True:
      plt.hist(grayscale_image.ravel(), bins=256, color='gray', alpha=0.5, rwidth=0.8, density=normalize_counts, label='Total (Grayscale)')

    title = 'Original RGB Histograms' if idx == 1 else 'Denoised RGB Histograms'
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    if normalize_counts:
      plt.ylabel('Probability Density')
    else:
      plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.legend(loc='upper right')

  plt.tight_layout()

  # Save the figure as an SVG
  if save_fig is True:
    plt.savefig(f"./figures/WGAN/FPN-50/histogram_original-vs-denoised/{image_timestamp}.rgb.svg", format="svg")
  else:
    plt.show()


def plot_histograms_grayscale(image_timestamp, original_image_resize_target, image_path_original, image_path_denoised, normalize_counts, save_fig):
  """Histogram for the total (i.e., the grayscale or luminance representation)"""

  # Load the images using OpenCV
  image_original = cv2.imread(image_path_original, cv2.IMREAD_GRAYSCALE)
  image_denoised = cv2.imread(image_path_denoised, cv2.IMREAD_GRAYSCALE)
  
  if original_image_resize_target is not None:
    new_dimensions  = (original_image_resize_target, original_image_resize_target)
    image_original = cv2.resize(image_original, new_dimensions)

  # Compute histograms
  hist_original = cv2.calcHist([image_original], [0], None, [256], [0, 256])
  hist_denoised = cv2.calcHist([image_denoised], [0], None, [256], [0, 256])

  # Normalize histograms for better visualization
  if normalize_counts is True:
    hist_original /= hist_original.sum()
    hist_denoised /= hist_denoised.sum()

  # Get non-zero ranges
  nz_original = np.where(hist_original > 0)
  nz_denoised = np.where(hist_denoised > 0)

  min_x = min(nz_original[0][0], nz_denoised[0][0])
  max_x = max(nz_original[0][-1], nz_denoised[0][-1])

  # Plot
  plt.figure(figsize=(10, 5))

  # Plot original image histogram
  plt.plot(hist_original, color='gray', label='Original Image')

  # Plot denoised image histogram
  plt.plot(hist_denoised, color='black', linestyle='--', label='Denoised Image')

  plt.title('Histograms of Original and Denoised Images')
  plt.xlim([min_x, max_x])  # Setting the x-axis limits
  plt.xlabel('Pixel Value')
  plt.ylabel('Normalized Count')
  plt.legend()

  # Save the figure as an SVG
  if save_fig:
    plt.savefig(f"./figures/WGAN/FPN-50/histogram_original-vs-denoised/{image_timestamp}.grayscale.svg", format="svg")
  else:
    plt.show()



def generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder, output_folder):
    
    # The original images are the default thumbnails retrieved from the spacecraft
    # This thumbnail is resized prior to onboard processing
    # We don't retrieve the resized version from the spacecraft so we have to resize it here
    original_image_resize_target = 224

    # Save the generated plots as SVG files
    save_fig = True

    # Include total (grayscale) histogram in RGB histogram
    include_total = False

    # Normalize the pixel count
    normalize_counts = False
    
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the rows in the CSV file
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        denoised_label = row['label_denoised']

        # Construct the paths for the original and denoised images
        original_path = os.path.join(original_folder, f"{denoised_label}/{timestamp}.jpeg")
        denoised_path = os.path.join(denoised_folder, f"{denoised_label}/{timestamp}.denoised.jpeg")

        
        # Generate the side-by-side histogram comparison plot
        # Plot RGB histogram
        plot_histograms_rgb(timestamp, original_image_resize_target, original_path, denoised_path, include_total, normalize_counts, save_fig)

        # Plot Total histogram (i.e., the grayscale or luminance representation)
        plot_histograms_grayscale(timestamp, original_image_resize_target, original_path, denoised_path, normalize_counts, save_fig)


# Example usage
csv_file = "./csv/results_classification-WGAN-FPN-50-short.csv"
csv_output_file = "./csv/results_classification-WGAN-FPN-50-metrics.csv"
original_folder = "./images/WGAN/FPN-50/"
denoised_folder = "./images/WGAN/FPN-50/"
output_folder = "./figures/WGAN/FPN-50/histogram_original-vs-denoised/"

generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder, output_folder)