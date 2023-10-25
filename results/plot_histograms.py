import os
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.diverging import RdYlBu_5

def plot_grayscale_histogram(image, label=None, xlim=(0, 256), color=None, fill=False):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist, label=label, color=color)
    if fill is True:
      plt.fill_between(np.arange(256), hist.ravel(), color=color, alpha=0.4)
    plt.xlim(xlim)


def plot_rgb_histogram(image, xlim=(0, 256), save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(4, 9))
    colors = ('r', 'g', 'b')
    color_names = ('Red', 'Green', 'Blue')
    
    for i, (col, color_name) in enumerate(zip(colors, color_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        axes[i].plot(hist, color=col)
        axes[i].fill_between(np.arange(256), hist.ravel(), color=col, alpha=0.4)
        axes[i].set_xlim(xlim)
        axes[i].set_title(f'{color_name} Channel')
        axes[i].set_ylabel('Pixel Frequency')
        if i == 2:
            axes[i].set_xlabel('Pixel Intensity')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    plt.show()


def plot_histogram(reference_folder_path, noise_image, output_three_channels, output_final, caption_histogram):

  # Load and resize the reference image
  reference_image_path = f"{reference_folder_path}/images/sample.jpeg"
  reference_image = cv2.imread(reference_image_path)
  reference_image = cv2.resize(reference_image, (224, 224))

  # Plot RGB histograms for the original image
  plot_rgb_histogram(reference_image, xlim=(20, 120), save_path=f"{reference_folder_path}/figures/histogram_rgb_original.svg")


  # Create another figure for the histograms of the WGAN p6 denoised image
  plt.figure(figsize=(10, 9))
  denoised_image_name = f"sample.{noise_image}.denoised.jpeg"
  denoised_image_path = os.path.join(f"{reference_folder_path}/images/", denoised_image_name)
  denoised_image = cv2.imread(denoised_image_path)

  # PLot RGB histograms
  temp = f"{reference_folder_path}/figures/{output_three_channels}"
  print (temp)
  plot_rgb_histogram(denoised_image, xlim=(20, 120), save_path=temp)




  # Plot the grayscale historgrams of original image + denoised
  plt.figure(figsize=(10, 4))

  # Define colors using the RdYlBu palette from palettable
  # FIXME: Use line types instead + legend

  # Plot grayscale histogram for the original image
  reference_image_gray = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
  reference_image_gray = cv2.resize(reference_image_gray, (224, 224))
  plot_grayscale_histogram(reference_image_gray, "Original image", xlim=(20, 120), fill=False)

  denoised_image_name = f"sample.{noise_image}.denoised.jpeg"
  denoised_image_path = os.path.join(f"{reference_folder_path}/images", denoised_image_name)
  denoised_image = cv2.imread(denoised_image_path, cv2.IMREAD_GRAYSCALE)
  denoised_image = cv2.resize(denoised_image, (224, 224))
  plot_grayscale_histogram(denoised_image, f'{caption_histogram} Denoised (margin is 6 pixels)', xlim=(20, 120), fill=False)



  '''
  # Create another figure with grayscale histograms of all denoised images
  # Read the CSV file
  csv_path = f"{reference_folder_path}/metrics.csv"
  df = pd.read_csv(csv_path)

  # For each patch_margin_pixels value in the CSV, plot the grayscale histogram for the corresponding denoised image
  for index, patch_margin_pixels in enumerate(df['patch_margin_pixels']):
    denoised_image_name = f"sample.fpn50.p{int(patch_margin_pixels)}.denoised.jpeg"
    denoised_image_path = os.path.join(f"{reference_folder_path}/images", denoised_image_name)
    denoised_image = cv2.imread(denoised_i
    plot_grayscale_histogram(denoised_image, f'p{int(patch_margin_pixels)}', xlim=(20, 120), fill=False)mage_path, cv2.IMREAD_GRAYSCALE)
    denoised_image = cv2.resize(denoised_image, (224, 224))
  '''

  plt.xlabel('Pixel Intensity')
  plt.ylabel('Frequency')
  plt.legend()
  plt.savefig(f"{reference_folder_path}/figures/{output_final}", format="svg")
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--reference_folder_path', required=True, type=str, help='Path to the reference folder')
  parser.add_argument('--noise_image', required=True, type=str, help='The type of noise and image of the results')
  parser.add_argument('--output_three_channels', required=True, type=str, help='The name of the output for three channels')
  parser.add_argument('--output_final', required=True, type=str, help='The final name of the file')
  parser.add_argument('--caption_histogram', required=True, type=str, help='The caption of the histogra')
  args = parser.parse_args()

  reference_image = plot_histogram(args.reference_folder_path, args.noise_image, args.output_three_channels, args.output_final, args.caption_histogram)