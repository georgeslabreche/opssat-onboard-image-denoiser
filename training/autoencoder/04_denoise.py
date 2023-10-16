#!/usr/bin/env python3

import os

# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
if True:
  os.add_dll_directory('C:/Users/Subspace_Sig1/miniconda3/envs/denoiser/Library/bin')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from utils import *
from PIL import Image

from sklearn.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity

# make sure constants are set as desired before executing this script
from constants import *

PREDICTION_DATA_SAMPLE_SIZE = 3

# create an argument parser
parser = argparse.ArgumentParser(description='Parse denoising parameters.')

# parse the model base name, noise type, and noise factor
parser.add_argument('-t', '--noisetype', type=int, help='the noise type')
parser.add_argument('-f', '--noisefactor', type=int, help='the noise factor')
parser.add_argument('-s', '--splitsize', type=int, help='the size of the split patches (e.g. 52 for 52x52')

# parse the arguments
args = parser.parse_args()

# construct the model name and path
noise_type_label = "fpn" if args.noisetype == 1 else "cfpn"
model_name = f"ae_{noise_type_label}{args.noisefactor}_{'p' if args.splitsize is not None else 'f'}"
model_path = MODEL_DIR + "/" + model_name
model_tflite_filepath = model_path + ".tflite"

# load the denoiser model
# FIXME: load the tflite file instead
denoiser = tf.keras.models.load_model(model_path)

# verbosity
print("Loaded model:", model_path)

# check its architecture
denoiser.summary()

# print encoder and decoder summaries
if DENOISER_TYPE < 4:
  denoiser.encoder.summary()
  denoiser.decoder.summary()


# the image data
image_data = None

# the image directory paths
original_image_dir_path = None
noisy_image_dir_path = None

if args.splitsize is None:
  original_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TEST_SET  + "/unnoised/original/*.jpeg"
  noisy_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TEST_SET  + "/noised/original/" + noise_type_label + "/" + str(args.noisefactor) + "/*.jpeg"
else:
  original_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TEST_SET  + f"/unnoised/split/{args.splitsize}/**/*.jpeg"
  noisy_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TEST_SET  + f"/noised/split/{args.splitsize}/" + noise_type_label + "/" + str(args.noisefactor) + "/**/*.jpeg"


if LOAD_NOISY_IMAGES_FROM_FILE is True:
  # Some verbosity
  print("Original images", original_image_dir_path)
  print("Noised images", noisy_image_dir_path)

  # list original image files
  original_image_files = tf.data.Dataset.list_files(original_image_dir_path)
  original_image_files = original_image_files.shuffle(buffer_size=10000).as_numpy_iterator()

  # list noisy image files
  noisy_image_files = tf.data.Dataset.list_files(noisy_image_dir_path)
  noisy_image_files = noisy_image_files.shuffle(buffer_size=10000).as_numpy_iterator()

  # sort both lists
  original_image_files = sorted([img_path for img_path in original_image_files])
  noisy_image_files = sorted([img_path for img_path in noisy_image_files])

  # convert to Dataset object
  original_image_files = tf.data.Dataset.from_tensor_slices(original_image_files)
  noisy_image_files = tf.data.Dataset.from_tensor_slices(noisy_image_files)

  # zip the two datasets together
  paired_dataset = tf.data.Dataset.zip((original_image_files, noisy_image_files))

  # take a subsample (to avoid OOM error)
  if PREDICTION_DATA_SAMPLE_SIZE is not None:
    paired_dataset = paired_dataset.take(PREDICTION_DATA_SAMPLE_SIZE)

  # get the image data
  image_data = paired_dataset.map(
    lambda original_path, noisy_path: load_and_preprocess_image_pair(
      original_path,
      noisy_path,
      resize_original = TRAINING_DATA_RESIZE_ORIGINAL_FROM_FILE if args.splitsize is None else False,
      resize_noisy    = TRAINING_DATA_RESIZE_NOISY_FROM_FILE if args.splitsize is None else False))

else:
  # Some verbosity
  print("Original images", original_image_dir_path)

  # only load the original images from files
  # the noise will be adding in-memory on the loaded originals
  original_image_files = tf.data.Dataset.list_files(original_image_dir_path)

  # load an preprocess the images
  # Setting apply_noise to False just means that we are reading the noisy images from pregenerated noisy image files rather than generating them
  # on the fly in-memory
  image_data = original_image_files.map(
    lambda x: load_and_preprocess_image(
      x,
      resize      = TRAINING_DATA_RESIZE_ORIGINAL_FROM_FILE if args.splitsize is None else False,
      apply_noise = True,
      noise_type  = args.noisetype,
      noise_factor= args.noisefactor))

# get the number of files
num_files = len(list(image_data))
print("Images in the dataset:", num_files)

# load an preprocess the images
x_images, x_images_noisy = zip(*list(image_data))

# convert tuples back into a single tensor
x_images_noisy = tf.stack(x_images_noisy)

# print the shape of the image inputs
print("Images shape:", np.shape(x_images_noisy))

# the decoded images
decoded_imgs = None

# denoise the noisy images into decoded images
# FIXME: support patched images (at the moment this only supports full images)

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
