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


def compare_images(original_path, denoised_path):
    # Load the original and denoised images
    original_image = cv2.imread(original_path)
    denoised_image = cv2.imread(denoised_path)
    
    # Resize the images to 224x224
    original_image = cv2.resize(original_image, (224, 224))
    denoised_image = cv2.resize(denoised_image, (224, 224))

    # Calculate the metrics
    psnr = peak_signal_noise_ratio(original_image, denoised_image)
    ssim = structural_similarity(original_image, denoised_image, multichannel=True, data_range=255, win_size=3)
    mse = mean_squared_error(original_image, denoised_image)
    return psnr, ssim, mse

def generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder):
    
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


    # Plotting the metrics
    plt.plot(psnr_list, label='PSNR')
    plt.plot(ssim_list, label='SSIM')
    plt.plot(mse_list, label='MSE')

    # Adding labels and title
    plt.xlabel('Image Index')
    plt.ylabel('Metric Value')
    plt.title('Comparison of PSNR, SSIM, and MSE')

    # Adding a legend
    plt.legend()

    # Save the plot to the output folder
    output_path = "./csv/results_classification-WGAN-FPN-50-metrics.svg"
    plt.savefig(output_path)


    # Plotting in log scale
    plt.close()

    # Creating an array of x-axis values
    x_values = np.arange(len(psnr_list))

    # Plotting the metrics with a logarithmic y-axis
    plt.semilogy(x_values, psnr_list, label='PSNR')
    plt.semilogy(x_values, ssim_list, label='SSIM')
    plt.semilogy(x_values, mse_list, label='MSE')

    # Adding labels and title
    plt.xlabel('Image Index')
    plt.ylabel('Metric Value (log scale)')
    plt.title('Comparison of PSNR, SSIM, and MSE')

    # Adding a legend
    plt.legend()

    # Save the plot to the output folder
    output_path = "./csv/results_classification-WGAN-FPN-50-metrics-log-scale.svg"
    plt.savefig(output_path)


# Example usage
csv_file = "./csv/results_classification-WGAN-FPN-50-short.csv"
csv_output_file = "./csv/results_classification-WGAN-FPN-50-metrics.csv"
original_folder = "./images/WGAN_FPN50/"
denoised_folder = "./images/WGAN_FPN50/"

generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder)