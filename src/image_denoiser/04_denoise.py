#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# make sure constants are set as desired before executing this script
from constants import *

# function to resize and normalize the input images
def preprocess_image(file_path):

  # read image file
  image = tf.io.read_file(file_path)

  # decode image in desired channel
  image = tf.image.decode_jpeg(image, channels=DESIRED_CHANNELS)

  # resize image in desired
  if RESIZE_IMAGE is True:
    image = tf.image.resize(image, [DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH])
  else:
    # convert the image data to float32
    image = tf.cast(image, tf.float32)

  # normalization
  image /= 255.0

  # return the image
  return image

# load the denoiser model
denoiser = tf.keras.models.load_model(MODEL_PATH)

# check its architecture
denoiser.summary()

# print encoder and decoder summaries
denoiser.encoder.summary()
denoiser.decoder.summary()

# list the image files
list_image_files = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_VALIDATE + "/*.jpg")

# get the number of files
num_files = len(list(list_image_files))
print("\nImages in the dataset:", num_files)

# load the images
x_images = list_image_files.map(preprocess_image)

# convert the image to NumPy arrays
x_images = np.array(list(x_images))
print(x_images.shape)

# add the noise
x_images_noisy = x_images + NOISE_FACTOR * tf.random.normal(shape=x_images.shape)
x_images_noisy = tf.clip_by_value(x_images_noisy, clip_value_min=0., clip_value_max=1.)

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
