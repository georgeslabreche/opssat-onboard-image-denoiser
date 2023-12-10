#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# Load the CSV files
df_ae = pd.read_csv('csv/results_classification-AE-FPN-50-metrics.csv')
df_wgans = pd.read_csv('csv/results_classification-WGAN-FPN-50-metrics.csv')

# Define a function to calculate averages and sample size
def calculate_averages_and_size(df, filter_condition=None):
  if filter_condition is not None:
    df = df.query(filter_condition)
  return {
    "Sample Size": len(df),
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
  "WGANs: Earth → Bad → Earth": ("df_wgans", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Earth'"),
  "WGANs: Earth → Bad → Earth (No Outliers)": ("df_wgans_no_outliers", None),
  "WGANs: Earth → Bad → Bad": ("df_wgans", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Bad'"),
  "WGANs: Earth → Earth → Bad": ("df_wgans", "label_original == 'Earth' and label_noised == 'Earth' and label_denoised == 'Bad'"),
  "WGANs: Earth → Earth → Earth": ("df_wgans", "label_original == 'Earth' and label_noised == 'Earth' and label_denoised == 'Earth'"),
  "WGANs: All Sequences": ("df_wgans", None),
  "Autoencoder: Earth → Bad → Earth": ("df_ae", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Earth'"),
  "Autoencoder: Earth → Bad → Bad": ("df_ae", "label_original == 'Earth' and label_noised == 'Bad' and label_denoised == 'Bad'"),
  "Autoencoder: Earth → Earth → Bad": ("df_ae", "label_original == 'Earth' and label_noised == 'Earth' and label_denoised == 'Bad'"),
  "Autoencoder: Earth → Earth → Earth": ("df_ae", "label_original == 'Earth' and label_noised == 'Earth' and label_denoised == 'Earth'"),
  "Autoencoder: All Sequences": ("df_ae", None)
}

# Calculate averages and size for each scenario
results = {}
for scenario, (df_name, condition) in scenarios.items():
  df = eval(df_name)
  results[scenario] = calculate_averages_and_size(df, condition)

# Print and write the results
with open('results_averages.txt', 'w') as file:
  for scenario, metrics in results.items():
    scenario_result = f"{scenario}"
    print(scenario_result)
    file.write(scenario_result + "\n")

    for metric, value in metrics.items():
      metric_result = f"  {metric}: {value}"
      print(metric_result)
      file.write(metric_result + "\n")

    file.write("\n")
    print()