#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# Read the CSV files
df_wgans = pd.read_csv('../../spacecraft/csv/results_classification-WGAN-FPN-50-metrics.csv')
df_ae = pd.read_csv('results.csv')

# Rename the columns in each dataframe to avoid ambiguity
df_wgans.rename(columns={'PSNR': 'psnr_wgans', 'SSIM': 'ssim_wgans', 'MSE': 'mse_wgans'}, inplace=True)
df_ae.rename(columns={'psnr': 'psnr_ae', 'ssim': 'ssim_ae', 'mse': 'mse_ae'}, inplace=True)

# Merge the dataframes on the 'timestamp' column
df_merged = pd.merge(df_wgans, df_ae, on='timestamp')

# Filter for rows where {label_original,label_noised,label_denoised} = {Earth, Bad, Earth}
earth_bad_earth_filter = (df_merged['label_original'] == 'Earth') & \
                         (df_merged['label_noised'] == 'Bad') & \
                         (df_merged['label_denoised'] == 'Earth')

df_earth_bad_earth = df_merged[earth_bad_earth_filter]

# Remove rows with negative SSIM in WGANs
df_earth_bad_earth_no_outlier = df_earth_bad_earth[df_earth_bad_earth['ssim_wgans'] >= 0]


# Calculate separate average metrics for Earth, Bad, Earth scenario in both datasets
avg_psnr_earth_bad_earth_wgans = df_earth_bad_earth['psnr_wgans'].mean()
avg_ssim_earth_bad_earth_wgans = df_earth_bad_earth['ssim_wgans'].mean()
avg_mse_earth_bad_earth_wgans = df_earth_bad_earth['mse_wgans'].mean()

avg_psnr_earth_bad_earth_ae = df_earth_bad_earth['psnr_ae'].mean()
avg_ssim_earth_bad_earth_ae = df_earth_bad_earth['ssim_ae'].mean()
avg_mse_earth_bad_earth_ae = df_earth_bad_earth['mse_ae'].mean()

# Calculate averages without the outlier for WGANs
avg_psnr_earth_bad_earth_wgans_no_outlier = df_earth_bad_earth_no_outlier['psnr_wgans'].mean()
avg_ssim_earth_bad_earth_wgans_no_outlier = df_earth_bad_earth_no_outlier['ssim_wgans'].mean()
avg_mse_earth_bad_earth_wgans_no_outlier = df_earth_bad_earth_no_outlier['mse_wgans'].mean()

avg_psnr_earth_bad_earth_ae_no_outlier = df_earth_bad_earth_no_outlier['psnr_ae'].mean()
avg_ssim_earth_bad_earth_ae_no_outlier = df_earth_bad_earth_no_outlier['ssim_ae'].mean()
avg_mse_earth_bad_earth_ae_no_outlier = df_earth_bad_earth_no_outlier['mse_ae'].mean()

# Calculate separate average metrics for all rows in both datasets
avg_psnr_all_wgans = df_merged['psnr_wgans'].mean()
avg_ssim_all_wgans = df_merged['ssim_wgans'].mean()
avg_mse_all_wgans = df_merged['mse_wgans'].mean()

avg_psnr_all_ae = df_merged['psnr_ae'].mean()
avg_ssim_all_ae = df_merged['ssim_ae'].mean()
avg_mse_all_ae = df_merged['mse_ae'].mean()

# Results
results = {
  "Earth, Bad, Earth - WGANs": {
    "Average PSNR": avg_psnr_earth_bad_earth_wgans,
    "Average SSIM": avg_ssim_earth_bad_earth_wgans,
    "Average MSE": avg_mse_earth_bad_earth_wgans
  },
  "Earth, Bad, Earth - Autoencoder": {
    "Average PSNR": avg_psnr_earth_bad_earth_ae,
    "Average SSIM": avg_ssim_earth_bad_earth_ae,
    "Average MSE": avg_mse_earth_bad_earth_ae
  },
  "Earth, Bad, Earth - WGANs (No Outliers)": {
    "Average PSNR": avg_psnr_earth_bad_earth_wgans_no_outlier,
    "Average SSIM": avg_ssim_earth_bad_earth_wgans_no_outlier,
    "Average MSE": avg_mse_earth_bad_earth_wgans_no_outlier
  },
  "Earth, Bad, Earth - Autoencoder (No Outliers)": {
    "Average PSNR": avg_psnr_earth_bad_earth_ae_no_outlier,
    "Average SSIM": avg_ssim_earth_bad_earth_ae_no_outlier,
    "Average MSE": avg_mse_earth_bad_earth_ae_no_outlier
  },
  "All Rows - WGANs": {
    "Average PSNR": avg_psnr_all_wgans,
    "Average SSIM": avg_ssim_all_wgans,
    "Average MSE": avg_mse_all_wgans
  },
  "All Rows - Autoencoder": {
    "Average PSNR": avg_psnr_all_ae,
    "Average SSIM": avg_ssim_all_ae,
    "Average MSE": avg_mse_all_ae
  }
}

# Print the results and write to a file
with open('results_averages.txt', 'w') as file:
  for scenario, metrics in results.items():
    scenario_result = f"{scenario}:"
    print(scenario_result)
    file.write(scenario_result + "\n")

    for metric, value in metrics.items():
      metric_result = f"  {metric}: {value}"
      print(metric_result)
      file.write(metric_result + "\n")

    file.write("\n")
    print()