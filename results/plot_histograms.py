import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# FIXME: The subplots for the original and denoised RGB histograms do not have the same x- and y- axis scale.
# Let's just manually fix this for the figures we end up including for publication.

def plot_histograms_rgb(image_timestamp, image_path_original, image_path_denoised, save_fig=True):
  # Load the images using PIL
  image_original = Image.open(image_path_original)
  image_denoised = Image.open(image_path_denoised)

  # Convert the images into arrays
  data_original = np.array(image_original)
  data_denoised = np.array(image_denoised)

  # Plot histograms for each image
  plt.figure(figsize=(15, 7))

  for idx, data in enumerate([data_original, data_denoised], start=1):
    plt.subplot(1, 2, idx)
    
    # Extract the RGB channels
    red_channel = data[:, :, 0]
    green_channel = data[:, :, 1]
    blue_channel = data[:, :, 2]

    # Plotting the Red, Green, and Blue histograms with normalized counts
    plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.5, rwidth=0.8, density=True, label='Red Channel')
    plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.5, rwidth=0.8, density=True, label='Green Channel')
    plt.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.5, rwidth=0.8, density=True, label='Blue Channel')

    title = 'Original RGB Histograms' if idx == 1 else 'Denoised RGB Histograms'
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density')
    plt.grid(axis='y', alpha=0.75)
    plt.legend(loc='upper right')

  plt.tight_layout()

  # Save the figure as an SVG
  if save_fig is True:
    plt.savefig(f"figures/image-histograms/{image_timestamp}.rgb.svg", format="svg")
  else:
    plt.show()
 
def plot_histograms_grayscale(image_timestamp, image_path_original, image_path_denoised, save_fig=True):
  # Load the images using OpenCV
  image_original = cv2.imread(image_path_original, cv2.IMREAD_GRAYSCALE)
  image_denoised = cv2.imread(image_path_denoised, cv2.IMREAD_GRAYSCALE)

  # Compute histograms
  hist_original = cv2.calcHist([image_original], [0], None, [256], [0, 256])
  hist_denoised = cv2.calcHist([image_denoised], [0], None, [256], [0, 256])

  # Normalize histograms for better visualization
  hist_original /= hist_original.sum()
  hist_denoised /= hist_denoised.sum()

  # Get non-zero ranges
  nz_original = np.where(hist_original > 0)
  nz_denoised = np.where(hist_denoised > 0)

  min_x = min(nz_original[0][0], nz_denoised[0][0])
  max_x = max(nz_original[0][-1], nz_denoised[0][-1])

  # Plot
  plt.figure(figsize=(10, 5))

  # Plot original image histogram
  plt.plot(hist_original, color='gray', label='Original Image')

  # Plot denoised image histogram
  plt.plot(hist_denoised, color='black', linestyle='--', label='Denoised Image')

  plt.title('Histograms of Original and Denoised Images')
  plt.xlim([min_x, max_x])  # Setting the x-axis limits
  plt.xlabel('Pixel Value')
  plt.ylabel('Normalized Count')
  plt.legend()

  # Save the figure as an SVG
  if save_fig:
    plt.savefig(f"figures/image-histograms/{image_timestamp}.grayscale.svg", format="svg")
  else:
    plt.show()

save_fig = True
ml_method = "AE"
noise_type = "FNP"
noise_factor = "50"
image_timestamp = 1694431981719

# Load the images
image_path_original = f"images/{ml_method}/{noise_type}-{noise_factor}/Earth/{image_timestamp}.jpeg"
image_path_denoised = f"images/{ml_method}/{noise_type}-{noise_factor}/Earth/{image_timestamp}.denoised.jpeg"

# Plot histograms
plot_histograms_rgb(image_timestamp, image_path_original, image_path_denoised, save_fig)
plot_histograms_grayscale(image_timestamp, image_path_original, image_path_denoised, save_fig)