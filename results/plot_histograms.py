import os
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from palettable.colorbrewer.diverging import RdYlBu_5

def plot_grayscale_histogram(original_image, denoised_image, fig_filepath, xlim=(0, 256), color=None, fill=False):
  # Plot the grayscale historgrams of original image + denoised
  plt.figure(figsize=(10, 4))

  def draw_histogram(image, label, xlim, color=None, fill=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    plt.plot(hist, label=label, color=color)
    if fill is True:
      plt.fill_between(np.arange(256), hist.ravel(), color=color, alpha=0.4)
    plt.xlim(xlim)

  # Plot grayscale histogram for the original and denoisd images
  draw_histogram(original_image, "Original", xlim, color, fill)
  draw_histogram(denoised_image, "Denoised", xlim, color, fill)

  plt.xlabel('Pixel Intensity')
  plt.ylabel('Frequency')
  plt.legend()
  plt.savefig(fig_filepath, format="svg")



def plot_rgb_histogram(image, fig_filepath, xlim=(0, 256), ylim=None):
  fig, axes = plt.subplots(3, 1, figsize=(4, 9))
  colors = ('r', 'g', 'b')
  color_names = ('Red', 'Green', 'Blue')

  for i, (col, color_name) in enumerate(zip(colors, color_names)):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    axes[i].plot(hist, color=col)
    axes[i].fill_between(np.arange(256), hist.ravel(), color=col, alpha=0.4)
    axes[i].set_xlim(xlim)
    if ylim:
      axes[i].set_ylim(ylim)
    axes[i].set_title(f'{color_name} Channel ', y=1.0, pad=-14, loc='right')
    axes[i].set_ylabel('Pixel Frequency')
    if i == 2:
      axes[i].set_xlabel('Pixel Intensity')
    plt.tight_layout()
    plt.savefig(fig_filepath, format="svg")


def plot_all_histograms(original_image_path, output_dirpath, xlim=(0, 256)):
  # Check if the path is a directory
  if os.path.isdir(original_image_path):
    # List all .jpeg images in the directory that do not contain "noised" or "denoised" in their filename
    image_files = glob.glob(os.path.join(original_image_path, '*.jpeg'))
    image_files = [f for f in image_files if "noised" not in f and "denoised" not in f]

  else:
    # If it's not a directory, use the provided single file
    image_files = [original_image_path]

  for img_path in image_files:
    # Load and resize the original image
    original_image = cv2.imread(img_path)
    original_image = cv2.resize(original_image, (224, 224))

    # Load the denoised image
    denoised_image_filepath = img_path.replace('.jpeg', '.denoised.jpeg')
    denoised_image = cv2.imread(denoised_image_filepath)

    # Find maximum y-value across both original and denoise image
    # Use this value as the max y-axis value for both histograms
    max_hist_value = 0
    for image in [original_image, denoised_image]:
      for i in range(3):
        hist = cv2.calcHist([image], [i], None, [xlim[1]], [xlim[0], xlim[1]])
        max_hist_value = max(max_hist_value, hist.max())

    # Construct the file path of the histogram svg files
    original_histogram_filepath = f"{output_dirpath}/{img_path.split('/')[-1].replace('.jpeg', '.histogram.rgb.svg')}"
    denoised_histogram_filepath = f"{output_dirpath}/{denoised_image_filepath.split('/')[-1].replace('.denoised.jpeg', '.histogram.rgb.denoised.svg')}"

    # Plot RGB histograms
    plot_rgb_histogram(original_image, fig_filepath=original_histogram_filepath, xlim=xlim, ylim=(0, max_hist_value))
    plot_rgb_histogram(denoised_image, fig_filepath=denoised_histogram_filepath, xlim=xlim, ylim=(0, max_hist_value))

    # Plot Grayscale histogram
    grayscale_histogram_filepath = f"{output_dirpath}/{img_path.split('/')[-1].replace('.jpeg', '.histogram.grayscale.svg')}"
    plot_grayscale_histogram(original_image, denoised_image, grayscale_histogram_filepath, xlim=xlim)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_original', '-i',  required=True,  type=str, help='The image file or folder path')
  parser.add_argument('--output_dir',     '-d',  required=True,  type=str, help='The output director where the generated histograms will be saved')
  parser.add_argument('--xlim0',          '-x0', required=False, type=int, default=0,   help='The x-axis lower limit')
  parser.add_argument('--xlim1',          '-x1', required=False, type=int, default=256, help='The x-axis upper limit')
  args = parser.parse_args()

  plot_all_histograms(args.image_original, args.output_dir, xlim=(args.xlim0, args.xlim1))
