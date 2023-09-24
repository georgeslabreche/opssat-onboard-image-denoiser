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

def generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder, output_folder):
    
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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

        # Generate the side-by-side histogram comparison plot

        # Load the images using OpenCV
        original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        denoised_image = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)
            
        # Resize the images to 224x224
        original_image = cv2.resize(original_image, (224, 224))
        denoised_image = cv2.resize(denoised_image, (224, 224))

        # If the images are in color, convert them to grayscale
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        #denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
        
        # Compute histograms
        hist1 = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([denoised_image], [0], None, [256], [0, 256])

        # Normalize histograms for better visualization
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot image1 histogram
        ax[0].plot(hist1, color='black')
        ax[0].set_title('Original')
        ax[0].set_xlim([0, 256])
        ax[0].set_xlabel('Pixel Value')
        ax[0].set_ylabel('Normalized Count')

        # Plot image2 histogram
        ax[1].plot(hist2, color='black')
        ax[1].set_title('Denoised')
        ax[1].set_xlim([0, 256])
        ax[1].set_xlabel('Pixel Value')
        ax[1].set_ylabel('Normalized Count')

        plt.tight_layout()
        # plt.suptitle(f"Image Comparison - Label: Original\nPSNR: {psnr:.2f}, SSIM: {ssim:.2f}, MSE: {mse:.2f}")
        # plt.show()
        # Save the plot to the output folder
        output_path = os.path.join(output_folder, f"{timestamp}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Comparison plot saved: {output_path}")

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
    output_path = "results_classification-AE-FNP-50-metrics.png"
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
    output_path = "results_classification-AE-FNP-50-metrics-log-scale.png"
    plt.savefig(output_path)


# Example usage
csv_file = "results_classification-AE-FNP-50-short.csv"
csv_output_file = "results_classification-AE-FNP-50-metrics.csv"
original_folder = "../images/AE/FNP-50/"
denoised_folder = "../images/AE/FNP-50/"
output_folder = "../images/AE/FNP-50/histogram_original-vs-denoised/"

generate_comparison_plots(csv_file, csv_output_file, original_folder, denoised_folder, output_folder)