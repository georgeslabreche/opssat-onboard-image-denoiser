#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.metrics import *
from skimage import measure
import numpy as np
from skimage import io, transform
import argparse


def compare_images(original_path, denoised_path, resize_original=True, resize_denoised=False):
  # TODO : Should we convert it to uint8? (reference_image * 255).astype('uint8')  # Convert to uint8
  
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
  df['psnr'] = psnr_list
  df['ssim'] = ssim_list
  df['mse'] = mse_list

  # Save the new CSV file with the metrics
  df.to_csv(csv_output_file, index=False)

  print(f"Metrics CSV file saved: {csv_output_file}")



def calculate_metrics_flatsat(reference_folder_path, noise):

  # Load the reference image
  reference_image_path = f"{reference_folder_path}/images/sample.jpeg"

  # Read the CSV file
  csv_path = f"{reference_folder_path}/metrics.csv"
  df = pd.read_csv(csv_path)

  # Lists to store the calculated metrics
  psnr_list = []
  ssim_list = []
  mse_list = []

  # Calculate metrics for each image and update the CSV
  for index, row in df.iterrows():
    patch_margin_pixels = int(row['patch_margin_pixels'])

    # Construct the paths for the original and denoised images
    denoised_image_name = f"sample.{noise}.p{patch_margin_pixels}.denoised.jpeg"
    denoised_image_path = os.path.join(f"{reference_folder_path}/images", denoised_image_name)

    # Compute metrics
    psnr, ssim, mse = compare_images(reference_image_path, denoised_image_path)

    # Append the metrics to the respective lists
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    mse_list.append(mse)
  
  # Add the metrics to the DataFrame
  df['psnr'] = psnr_list
  df['ssim'] = ssim_list
  df['mse'] = mse_list

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

  print(f"Metrics CSV file saved: {csv_path} and rounded {rounded_csv_path}")

  # Pretty print the updated CSV
  print(df)

def process_results_preliminary_sample():
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

      # Append to the dataframe
      df.loc[len(df)] = [filename, psnr, ssim, mse]

  # Sort the dataframe by filename
  df = df.sort_values(by="filename")

  # Write the df as a csv file in ./prelimiary/results.csv
  df.to_csv(results_csv_path, index=False)


def process_results_preliminary_testset():

  # Process results for all model and input types
  model_types = ["AE", "WGAN"]
  input_types = ["full", "patch"]

  # Loop through results for all model and input types
  for mt in model_types:
    for it in input_types:

      # Build the paths
      original_dir_path = f"./preliminary/testset/original"
      denoised_dir_path = f"./preliminary/testset/denoised/{mt}/{it}"
      results_csv_filepath = f"./preliminary/testset/results_{mt.lower()}_{it}.csv"

      # Initialize a DataFrame to store the results
      df = pd.DataFrame(columns=["filename", "timestamp", "psnr", "ssim", "mse"])

      # Loop through all .jpeg files
      for image_filename in os.listdir(original_dir_path):
        if image_filename.endswith(".jpeg"):
          original_image_filepath = os.path.join(original_dir_path, image_filename)
          denoised_image_filepath = os.path.join(denoised_dir_path, image_filename)

          # Calculate the metrics
          psnr, ssim, mse = compare_images(original_image_filepath, denoised_image_filepath)

          # Append to the dataframe
          image_timestamp = image_filename.replace("img_msec_", "").replace("_1_", "").replace("_2_", "").replace("thumbnail.jpeg", "")
          df.loc[len(df)] = [image_filename, image_timestamp, psnr, ssim, mse]

      # Sort the dataframe by timestamp
      df = df.sort_values(by="timestamp")

      # Write the df as a csv file
      df.to_csv(results_csv_filepath, index=False)


def process_results_flatsat(reference_folder_path, noise):
  # FIXME: Move implementation from ./flatsat/calculate_metrics.py into this function
  #       Delete ./flatsat/calculate_metrics.py
  #       Make sure that CSVs in ./flatsat/wgan_fpn50_p0-8_01 are calculated from this function and not from ./flatsat/calculate_metrics.py
  print("\nFor no particular reason, the FlatSat results have their own script in ./flatsat/calculate_metrics.py\n")

  default_folder = "./flatsat"

  # Paths
  reference_folder_path = default_folder + reference_folder_path

  calculate_metrics_flatsat(reference_folder_path, noise)
  

def process_results_flatsat2():
  original_dir_path = "./spacecraft/images/WGAN/FPN-50/Earth"
  denoised_dir_path = "./flatsat/fm-wgans_to_em-ae/images"
  results_csv_filepath ="./flatsat/fm-wgans_to_em-ae/results.csv"
  
  # Initialize a DataFrame to store the results
  df = pd.DataFrame(columns=["filename", "timestamp", "psnr", "ssim", "mse"])
  
  # Loop through all .jpeg files
  for denoised_image_filename in os.listdir(denoised_dir_path):
    if denoised_image_filename.endswith(".jpeg"):
      original_image_filename = denoised_image_filename.replace(".denoised.", ".")

      original_image_filepath = os.path.join(original_dir_path, original_image_filename)
      denoised_image_filepath = os.path.join(denoised_dir_path, denoised_image_filename)

      # Calculate the metrics
      psnr, ssim, mse = compare_images(original_image_filepath, denoised_image_filepath)

      # Append to the dataframe
      df.loc[len(df)] = [original_image_filename, original_image_filename.replace(".jpeg", ""), psnr, ssim, mse]

  # Sort the dataframe by timestamp
  df = df.sort_values(by="timestamp")
    
  # Write the df as a csv file
  df.to_csv(results_csv_filepath, index=False)


def process_results_spacecraft(csv_file, csv_output_file, original_folder, denoised_folder):
  # FIXME: Sort by filename before saving
  
  default_folder = "./spacecraft"

  # Paths
  csv_file = default_folder + csv_file
  csv_output_file =  default_folder + csv_output_file
  original_folder = default_folder + original_folder
  denoised_folder = default_folder + denoised_folder

  print(csv_file)

  # Calculate metrics
  calculate_metrics(csv_file, csv_output_file, original_folder, denoised_folder)


def parse_arguments():
  parser = argparse.ArgumentParser(description='Caculate similarity metrics of denoised images.')
  parser.add_argument('-t', '--target', choices=['p', 't', 'f', 'f2', 's'], required=True,
                      help='Target for processing: (p)reliminary, (t)estset, (f)latsat, or (s)pacecraft')
  parser.add_argument('-ci', '--csv_input', required=False, type=str, help='CSV input file')
  parser.add_argument('-co', '--csv_output', required=False, type=str, help='CSV output file')
  parser.add_argument('-o', '--original_folder', required=False, type=str, help='Folder of the original images')
  parser.add_argument('-d', '--denoised_folder', required=False, type=str, help='Folder of the denoised images (normally the same as the original_folder)')

  # flatsat
  parser.add_argument('-f', '--reference_folder_path', required=False, type=str, help='Path to the reference folder')
  parser.add_argument('-n', '--noise', required=False, type=str, help='The type of noise of the results')
  return parser.parse_args()


def main():
  args = parse_arguments()

  if args.target == 'p': # preliminary sample results
    process_results_preliminary_sample()
  if args.target == 't': # preliminary testset results
    process_results_preliminary_testset()
  elif args.target == 'f': # flatsat results
    process_results_flatsat(args.reference_folder_path, args.noise)
  elif args.target == 'f2': # downlink wgans spacecraft results but process on the flatsat and with autoencoder
    process_results_flatsat2()
  elif args.target == 's': # spacecraft results
    process_results_spacecraft(args.csv_input, args.csv_output, args.original_folder, args.denoised_folder)
  

if __name__ == "__main__":
  main()
