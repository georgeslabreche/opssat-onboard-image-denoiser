import pandas as pd

# read the csv file
df_ae_full = pd.read_csv('results_ae_full.csv')
df_ae_patch = pd.read_csv('results_ae_patch.csv')
df_wgan_full = pd.read_csv('results_wgan_full.csv')
df_wgan_patch = pd.read_csv('results_wgan_patch.csv')

# calculate the averages
avg_ae_full = df_ae_full[['psnr', 'ssim', 'mse']].mean()
avg_ae_patch = df_ae_patch[['psnr', 'ssim', 'mse']].mean()
avg_wgan_full = df_wgan_full[['psnr', 'ssim', 'mse']].mean()
avg_wgan_patch = df_wgan_patch[['psnr', 'ssim', 'mse']].mean()

# create a summary
summary = pd.DataFrame({
  'Autoencoder Full': avg_ae_full,
  'Autoencoder Patch': avg_ae_patch,
  'WGANs Full': avg_wgan_full,
  'WGANs Patch': avg_wgan_patch
}).T.reset_index() # transpose and reset index to turn the models into a column

# rename columns
summary.columns = ['Model', 'PSNR', 'SSIM', 'MSE']

# display and save the summary
print(summary)
summary.to_csv('results_averages.csv', index=False)
