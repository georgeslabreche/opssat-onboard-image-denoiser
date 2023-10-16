# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.metrics import *
from skimage import measure
import numpy as np
import argparse


def compare_images(original_path, denoised_path, resize_original=True, resize_denoised=False):
    # Load the original and denoised images
    original_image = cv2.imread(original_path)
    denoised_image = cv2.imread(denoised_path)
    
    # Resize the original images to 224x224
    if resize_original is True:
      original_image = cv2.resize(original_image, (224, 224))

    # Resize the denoised images to 224x224
    if resize_denoised is True:
      denoised_image = cv2.resize(denoised_image, (224, 224))

    # Calculate the metrics
    psnr = peak_signal_noise_ratio(original_image, denoised_image)
    ssim = structural_similarity(original_image, denoised_image, multichannel=True, data_range=255, win_size=3)
    mse = mean_squared_error(original_image, denoised_image)
    
    # Return the results
    return psnr, ssim, mse


def calculate_metrics(csv_file, csv_output_file, original_folder, denoised_folder):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Lists to store the calculated metrics
    psnr_list = []
    ssim_list = []
    mse_list = []

    # Iterate over the rows in the CSV file
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        denoised_label = row['label_denoised']

        # Construct the paths for the original and denoised images
        original_path = os.path.join(original_folder, f"{denoised_label}/{timestamp}.jpeg")
        denoised_path = os.path.join(denoised_folder, f"{denoised_label}/{timestamp}.denoised.jpeg")

        # Compare the images and calculate the metrics
        psnr, ssim, mse = compare_images(original_path, denoised_path)

        # Append the metrics to the respective lists
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)

    # Add the metrics to the DataFrame
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df['MSE'] = mse_list

    # Save the new CSV file with the metrics
    df.to_csv(csv_output_file, index=False)

    print(f"Metrics CSV file saved: {csv_output_file}")


def process_results_preliminary():
  preliminary_path = "./preliminary"
  reference_image_path = os.path.join(preliminary_path, "sample.jpeg")
  results_csv_path = os.path.join(preliminary_path, "results.csv")

  # Initialize a DataFrame to store the results
  df = pd.DataFrame(columns=["filename", "psnr", "ssim", "mse"])

  # Loop through all .jpeg files in the preliminary folder
  for filename in os.listdir(preliminary_path):
    if filename.endswith(".jpeg") and filename != "sample.jpeg":
      image_path = os.path.join(preliminary_path, filename)

      # Calculate the metrics
      psnr, ssim, mse = compare_images(reference_image_path, image_path)

      # Append to 
      df.loc[len(df)] = [filename, psnr, ssim, mse]
  
  # Write the df as a csv file in ./prelimiary/results.csv
  df.to_csv(results_csv_path, index=False)


def process_results_flatsat():
  print("\nFor no particular reason, the FlatSat results have their own script in ./flatsat/calculate_metrics.py\n")


def process_results_spacecraft():
  # Paths
  csv_file = "./spacecraft/csv/results_classification-WGAN-FPN-50-short.csv"
  csv_output_file = "./spacecraft/csv/results_classification-WGAN-FPN-50-metrics.csv"
  original_folder = "./spacecraft/images/WGAN/FPN-50/"
  denoised_folder = "./spacecraft/images/WGAN/FPN-50/"

  # Calculate metrics
  calculate_metrics(csv_file, csv_output_file, original_folder, denoised_folder)


def parse_arguments():
  parser = argparse.ArgumentParser(description='Caculate similarity metrics of denoised images.')
  parser.add_argument('-t', '--target', choices=['p', 'f', 's'], required=True,
                      help='Target for processing: (p)reliminary, (f)latsat, or (s)pacecraft')
  return parser.parse_args()


def main():
  args = parse_arguments()

  if args.target == 'p': # preliminary results
    process_results_preliminary()
  elif args.target == 'f': # flatsat results
    process_results_flatsat()
  elif args.target == 's': # spacecraft results
    process_results_spacecraft()

if __name__ == "__main__":
    main()
