#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# Load the CSV files
df_ae = pd.read_csv('csv/results_classification-AE-FPN-50-metrics.csv')
df_wgans = pd.read_csv('csv/results_classification-WGAN-FPN-50-metrics.csv')

# Define a function to calculate averages
def calculate_averages(df, filter_condition=None):
  if filter_condition is not None:
    df = df.query(filter_condition)
  return {
    "Average PSNR": df['PSNR'].mean(),
    "Average SSIM": df['SSIM'].mean(),
    "Average MSE": df['MSE'].mean()
  }

# Filter out rows with negative SSIM in WGANs for Earth, Bad, Earth scenario
df_wgans_no_outliers = df_wgans[(df_wgans['label_original'] == 'Earth') & 
                                (df_wgans['label_noised'] == 'Bad') & 
                                (df_wgans['label_denoised'] == 'Earth') & 
                                (df_wgans['SSIM'] >= 0)]

# Scenarios and their filter conditions
scenarios = {
  "Earth, Bad, Earth - WGANs": ("df_wgans", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Earth'"),
  "Earth, Bad, Earth - WGANs (No Outliers)": ("df_wgans_no_outliers", None),
  "Earth, Bad, Earth - Autoencoder": ("df_ae", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Earth'"),
  "All Rows - WGANs": ("df_wgans", None),
  "All Rows - Autoencoder": ("df_ae", None)
}

# Calculate averages for each scenario
results = {}
for scenario, (df_name, condition) in scenarios.items():
  df = eval(df_name)
  results[scenario] = calculate_averages(df, condition)

# Print and write the results
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
