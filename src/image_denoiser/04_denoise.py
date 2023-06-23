#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import *

# make sure constants are set as desired before executing this script
from constants import *

# load the denoiser model
denoiser = tf.keras.models.load_model(MODEL_PATH)

# check its architecture
denoiser.summary()

# print encoder and decoder summaries
denoiser.encoder.summary()
denoiser.decoder.summary()

# list the image files
list_image_files = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_VALIDATE + "/*.jpeg")

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

# encode the noisy images
encoded_imgs = denoiser.encoder(x_images_noisy).numpy()

# decode the encoded images
decoded_imgs = denoiser.decoder(encoded_imgs).numpy()

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
  images = [x_images[i], x_images_noisy[i], decoded_imgs[i]]
  titles = ["original", "original + noise", "reconstructed"]
  plot_images(images, titles)
