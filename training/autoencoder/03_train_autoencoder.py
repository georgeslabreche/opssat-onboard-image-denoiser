#!/usr/bin/env python3

import os

# make sure constants are set as desired before training
from constants import *

# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
if GPU_ENABLED:
  os.add_dll_directory('C:/Users/Subspace_Sig1/miniconda3/envs/denoiser/Library/bin')

import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf # tensorflow
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping
from autoencoders import *

from utils import *
import argparse


# increase this when running on a proper ML training computer with GPU
# set to None to train with all available training data
TRAINING_DATA_SAMPLE_SIZE = None

# increase the training dataset size by rotating the images
# set to None for no rotations
NUMBER_OF_ROTATED_IMAGES_IN_TRAINING = None

# Print Tensorflow version
print(tf.__version__)

# create an argument parser
parser = argparse.ArgumentParser(description='Parse training parameters.')

# parse arguments
# for training with patch images -e is set to 10, with full images the default 5 is used
parser.add_argument('-t', '--noisetype', type=int, default=NOISE_TYPE, help='the noise type')
parser.add_argument('-f', '--noisefactor', type=int, default=NOISE_FACTOR, help='the noise factor')
parser.add_argument('-s', '--splitsize', type=int, help='the size of the split patches (e.g. 56 for 56x56)')
parser.add_argument('-e', '--earlystop', type=int, default=5, help='early stopping patience count')

# parse the arguments
args = parser.parse_args()

# check if TensorFlow was built with CUDA and GPU support
if GPU_ENABLED is True:

  # print some GPU stuff
  print("Built with CUDA: ", tf.test.is_built_with_cuda())
  print("Built with GPU support: ", tf.test.is_built_with_gpu_support())

  # Verbosity on the number of GPUs available
  print("Number GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# construct the model name and path
noise_type_label = "fpn" if args.noisetype == 1 else "cfpn"
model_name = f"ae_{noise_type_label}{args.noisefactor}_{'p' if args.splitsize is not None else 'f'}"
model_path = MODEL_DIR + "/" + model_name
model_tflite_filepath = model_path + ".tflite"

# verbosity
print(f"The trained model will be saved as {model_tflite_filepath}")

# delete model folder if it exists already
if os.path.exists(model_path):
  shutil.rmtree(model_path)

# delete model tflite file if it exists already
if os.path.exists(model_tflite_filepath):
  os.remove(model_tflite_filepath)


# function to rotate the image 90 degrees counter clockwise
def rot90_image(image, image_noisy):
  return tf.image.rot90(image, 1), tf.image.rot90(image_noisy, 1)

# the image data
image_data = None

original_image_dir_path = None
noisy_image_dir_path = None

if args.splitsize is None:
  original_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TRAINING_SET + "/unnoised/original/*.jpeg"
  noisy_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TRAINING_SET + "/noised/original/" + noise_type_label + "/" + str(args.noisefactor) + "/*.jpeg"
else:
  original_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TRAINING_SET + f"/unnoised/split/{args.splitsize}/**/*.jpeg"
  noisy_image_dir_path = DIR_PATH_IMAGES_OPSSAT_TRAINING_SET + f"/noised/split/{args.splitsize}/" + noise_type_label + "/" + str(args.noisefactor) + "/**/*.jpeg"


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
  if TRAINING_DATA_SAMPLE_SIZE is not None:
    paired_dataset = paired_dataset.take(TRAINING_DATA_SAMPLE_SIZE)

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

# rotate the images to create more training data
if NUMBER_OF_ROTATED_IMAGES_IN_TRAINING is not None:
  
  # the original size of the training dataset
  len1 = len(list(image_data))
  print("Increase training dataset by rotating images:")

  # rotate the images
  # and append to the original image dataset
  if NUMBER_OF_ROTATED_IMAGES_IN_TRAINING in [1, 2, 3]: 
    image_data_rot90 = image_data.map(rot90_image)
    image_data = image_data.concatenate(image_data_rot90)
    len2 = len(list(image_data))
    print(f"  {len1} --> {len2}")
  
  if NUMBER_OF_ROTATED_IMAGES_IN_TRAINING in [2, 3]:
    image_data_rot180 = image_data_rot90.map(rot90_image)
    image_data = image_data.concatenate(image_data_rot180)
    len3 = len(list(image_data))
    print(f"  {len2} --> {len3}")
  
  if NUMBER_OF_ROTATED_IMAGES_IN_TRAINING == 3:
    image_data_rot270 = image_data_rot180.map(rot90_image)
    image_data = image_data.concatenate(image_data_rot270)
    len4 = len(list(image_data))
    print(f"  {len3} --> {len4}")

# calculate the number of files in the training dataset to use as training data
# the rest will be used as test data
num_train = int(len(list(image_data)) * TRAIN_RATIO)

# shuffle
image_data = image_data.shuffle(buffer_size=3000)

# split the image data between train and test data
train_data = image_data.take(num_train)
validation_data  = image_data.skip(num_train)

# convert the train and test datasets to NumPy arrays
x_train, x_train_noisy = zip(*list(train_data))
x_validate, x_validate_noisy = zip(*list(validation_data))

# convert tuples back into a single tensor
x_train           = tf.stack(x_train)
x_train_noisy     = tf.stack(x_train_noisy)
x_validate        = tf.stack(x_validate)
x_validate_noisy  = tf.stack(x_validate_noisy)

# print the shapes of the train and validation datasets
print("Train images shape:",               np.shape(x_train))
print("Train images shape (noisy):",       np.shape(x_train_noisy))
print("Validation images shape:",          np.shape(x_validate))
print("Validation images shape (noisy):",  np.shape(x_validate_noisy))


# plot the first 10 images
# first row: original
# second row: original + noise
if DISPLAY_TEST_NOISE:
  n = 10
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
  
    ax = plt.subplot(2, n, i + 1)
    plt.title("original")
    plt.imshow(tf.squeeze(x_validate[i]))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original + noise
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_validate_noisy[i]))
    #plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

  # show the plot
  plt.show()


# instanciate the desired denoiser autoencoder
# todo: make scalable with Factory Pattern
denoiser = None
if DENOISER_TYPE == 1:
  denoiser = DenoiseAutoencoderNaive(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
elif DENOISER_TYPE == 2:
  denoiser = DenoiseAutoencoderSimple(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
elif DENOISER_TYPE == 3:
  denoiser = DenoiseAutoencoderComplex(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
elif DENOISER_TYPE == 4:
  denoiser = DenoiseAutoencoderSkipConnection()
elif DENOISER_TYPE == 5:
  denoiser = DenoiseAutoencoderVGG16()
elif DENOISER_TYPE == 6:
  denoiser = DenoiseAutoencoderSkipConnectionVGG16()
elif DENOISER_TYPE == 7:
  denoiser = DenoiseAutoencoderMobileNetV2()
elif DENOISER_TYPE == 8:
  denoiser = DenoiseAutoencoderSkipConnectionMobileNetV2()
else:
  print(f"Error: unsupported denoiser encoder typel: {DENOISER_TYPE}")
  quit()

# compile
denoiser.compile(optimizer='adam', loss=losses.MeanSquaredError())

# define the early stopping criteria
early_stop = EarlyStopping(monitor='val_loss', patience=args.earlystop, verbose=1)

# train the denoiser autoencoder
# use the training dataset
history = denoiser.fit(x_train_noisy, x_train,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  shuffle=True,
  validation_data=(x_validate_noisy, x_validate),
  callbacks=[early_stop])

# save the model
denoiser.save(model_path)

# convert the model to a tflite mode and save
converter = tf.lite.TFLiteConverter.from_keras_model(denoiser)
tflite_denoiser = converter.convert()

# save the tflite model
with open(model_tflite_filepath, 'wb') as f:
  f.write(tflite_denoiser)

# print encoder and decoder summaries
if DENOISER_TYPE < 4:
  denoiser.encoder.summary()
  denoiser.decoder.summary()
  

# plot training & validation loss values
if DISPLAY_TRAINING_AND_VALIDATION_LOSS_VALUES:
  plt.figure(figsize=(12, 6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.show()
  
# plot some denoised validation images
if DISPLAY_VALIDATE_NOISE:

  # decoded images:
  decoded_imgs = None

  # take the first 4 items
  number_of_test_denoise_to_plot = 4
  x_validate_noisy = tf.random.shuffle(x_validate_noisy)
  x_validate_noisy = x_validate_noisy[:number_of_test_denoise_to_plot]

  # denoise the test noisy images into decoded images
  if DENOISER_TYPE >= 4:
    # pass the noisy images through the denoiser model
    decoded_imgs = denoiser(x_validate_noisy)
  else:
    # encode the noisy images from the test set
    encoded_imgs = denoiser.encoder(x_validate_noisy).numpy()

    # decode the encoded images
    decoded_imgs = denoiser.decoder(encoded_imgs).numpy()

  # plot both the noisy images and the denoised images produced by the denoiser autoencoder
  n = number_of_test_denoise_to_plot
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_validate_noisy[i]))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    #plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

  # show the reconstruction
  plt.show()