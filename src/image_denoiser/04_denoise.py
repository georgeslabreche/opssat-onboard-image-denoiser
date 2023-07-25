#!/usr/bin/env python3

import os

# Somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
os.add_dll_directory('C:/Users/Subspace_Sig1/miniconda3/envs/denoiser/Library/bin')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import *
from PIL import Image

from sklearn.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity


# make sure constants are set as desired before executing this script
from constants import *

# load the denoiser model
denoiser = tf.keras.models.load_model(MODEL_PATH)

# check its architecture
denoiser.summary()

# print encoder and decoder summaries
if DENOISER_TYPE < 4:
  denoiser.encoder.summary()
  denoiser.decoder.summary()

# list the jpeg image files
list_image_files_jpeg = None
try:
  list_image_files_jpeg = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_VALIDATE + "/*.jpeg")
except:
  pass

# list the jpg image files
list_image_files_jpg = None
try:
  list_image_files_jpg = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_VALIDATE + "/*.jpg")
except:
  pass

# merge the list of image files
list_image_files = None
if None not in [list_image_files_jpeg, list_image_files_jpg]:
  list_image_files = list_image_files_jpeg.concatenate(list_image_files_jpg)
elif list_image_files_jpeg is not None:
  list_image_files = list_image_files_jpeg
elif list_image_files_jpg is not None:
  list_image_files = list_image_files_jpg
else:
  print(f"No images found in {DIR_PATH_IMAGERY_VALIDATE}")
  exit(1)

# get the number of files
num_files = len(list(list_image_files))
print("\nImages in the dataset:", num_files)

# load the images
input_data = list_image_files.map(lambda x: load_and_preprocess_image(x, resize=True))

# load an preprocess the images
x_images, x_images_noisy = zip(*list(input_data))

# convert tuples back into a single tensor
x_images_noisy = tf.stack(x_images_noisy)

# print the shape of the image inputs
print("Images shape:", np.shape(x_images_noisy))

# the decoded images
decoded_imgs = None

# denoise the noisy images into decoded images
if DENOISER_TYPE >= 4:
  # pass the noisy images through the denoiser model
  decoded_imgs = denoiser(x_images_noisy)
else:
  # encode the noisy images
  encoded_imgs = denoiser.encoder(x_images_noisy).numpy()

  # decode the encoded images
  decoded_imgs = denoiser.decoder(encoded_imgs).numpy()

  #predicted_noise = denoiser.predict(x_images_noisy)
  #decoded_imgs = x_images_noisy - predicted_noise


# function to plot 3 images in a row: 1) the original image, 2) the image with noise, and 3) the constructed image
def plot_images(images, titles):
  fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(len(images) * 5, 5))

  for i, image in enumerate(images):
    #axes[i].imshow(tf.squeeze(image), cmap='gray')
    axes[i].imshow(tf.squeeze(image))
    axes[i].set_title(titles[i], fontsize=14)
    axes[i].axis('off')

  plt.tight_layout()
  plt.show()

# plot both the noisy images and the denoised images produced by the denoiser autoencoder
for i in range(num_files):

  # evaluate the similarities between the original image and the denoised image by using the following metrics:

  # Mean Squared Error (MSE):
  #  - this is a common quantitative measure for comparing two images
  #  - it computes the average squared difference between the corresponding pixels in the two images
  #  - lower MSE values indicate closer match (0 indicates indentical images)
  mse = mean_squared_error(tf.reshape(x_images[i], [-1]), tf.reshape(x_images_noisy[i], [-1]))
  
  # Normalized Root Mean Squared Error (NRMSE):
  # - normalized by the maximum pixel intensity, making it scale invariant.
  # - this means that it provides a relative measure of the overall error between the two images, instead of an absolute measure
  # - lower NRMSE values indicate closer match (0 indicates indentical images)
  nrmse = normalized_root_mse(x_images[i].numpy(), decoded_imgs[i].numpy())
  
  # Peak Signal-to-Noise Ratio (PSNR):
  #  - this is a common measure that is especially used in image processing tasks
  #  - it measures the peak error
  #  - a higher PSNR indicates a closer match between the two images
  psnr = peak_signal_noise_ratio(x_images[i].numpy(), decoded_imgs[i].numpy())
  
  # Structural Similarity Index (SSIM):
  #  - this is a more advanced measure that considers changes in structural information, luminance, and contrast
  #  - a value of 1 indicates a perfect match
  # see: https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
  ssim = structural_similarity(x_images[i].numpy(), decoded_imgs[i].numpy(), multichannel=True, channel_axis=-1, data_range=1)
  
  # print results
  print(f"mse: {mse}, nrmse: {nrmse}, psnr: {psnr}, ssim: {ssim}")

  # plot the images
  images = [x_images[i], x_images_noisy[i], decoded_imgs[i]]
  titles = ["original", "original + noise", "reconstructed"]
  plot_images(images, titles)
