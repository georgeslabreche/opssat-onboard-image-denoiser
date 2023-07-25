#!/usr/bin/env python3

import os

# Somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
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

# make sure constants are set as desired before training
from constants import *


# increase this when running on a proper ML training computer with GPU
# set to None to train with all available training data
TRAINING_DATA_SAMPLE_SIZE = None

# resize training data
TRAINING_DATA_RESIZE = True

# increase the training dataset size by rotating the images
# set to None for no rotations
NUMBER_OF_ROTATED_IMAGES_IN_TRAINING = None


# Print Tensorflow version
print(tf.__version__)

# Check if Tensorflow was built with CUDA and GPU support
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Built with GPU support: ", tf.test.is_built_with_gpu_support())

# Verbosity on the number of GPUs available
print("Number GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Delete model folder if it exists already
if os.path.exists(MODEL_PATH):
  shutil.rmtree(MODEL_PATH)

# Delete model tflite file if it exists already
if os.path.exists(MODEL_PATH + ".tflite"):
  os.remove(MODEL_PATH + ".tflite")


# function to rotate the image 90 degrees counter clockwise
def rot90_image(image, image_noisy):
  return tf.image.rot90(image, 1), tf.image.rot90(image_noisy, 1)

# list the image files
list_image_files = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_TRAIN + "/*.jpeg")

# get the number of files
num_files = len(list(list_image_files))
print("Images in the dataset:", num_files)

# Take a subsample only in dev environment.
if TRAINING_DATA_SAMPLE_SIZE is not None:
  list_image_files = list_image_files.take(TRAINING_DATA_SAMPLE_SIZE)

# get the number of files
num_files = len(list(list_image_files))
print("Images sampled from the dataset:", num_files)

# load an preprocess the images
image_data = list_image_files.map(lambda x: load_and_preprocess_image(x, resize=TRAINING_DATA_RESIZE))

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
test_data  = image_data.skip(num_train)

# convert the train and test datasets to NumPy arrays
x_train, x_train_noisy = zip(*list(train_data))
x_test, x_test_noisy = zip(*list(test_data))

# convert tuples back into a single tensor
x_train       = tf.stack(x_train)
x_train_noisy = tf.stack(x_train_noisy)
x_test        = tf.stack(x_test)
x_test_noisy  = tf.stack(x_test_noisy)

# print the shapes of the train and test datasets
print("Train images shape:",         np.shape(x_train))
print("Train images shape (noisy):", np.shape(x_train_noisy))
print("Test images shape:",          np.shape(x_test))
print("Test images shape (noisy):",  np.shape(x_test_noisy))


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
    plt.imshow(tf.squeeze(x_test[i]))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original + noise
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
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
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# train the denoiser autoencoder
# use the training dataset
history = denoiser.fit(x_train_noisy, x_train,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  shuffle=True,
  validation_data=(x_test_noisy, x_test),
  callbacks=[early_stop])

# save the model
denoiser.save(MODEL_PATH)

# convert the model to a tflite mode and save
converter = tf.lite.TFLiteConverter.from_keras_model(denoiser)
tflite_denoiser = converter.convert()

# save the tflite model
with open(MODEL_PATH + '.tflite', 'wb') as f:
  f.write(tflite_denoiser)

# print encoder and decoder summaries
if DENOISER_TYPE < 4:
  denoiser.encoder.summary()
  denoiser.decoder.summary()
  
  
# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
  
# plot some denoised test images

# decoded images:
decoded_imgs = None

# take the first 10 items
number_of_test_denoise_to_plot = 4
x_test_noisy = tf.random.shuffle(x_test_noisy)
x_test_noisy = x_test_noisy[:number_of_test_denoise_to_plot]

# denoise the test noisy images into decoded images
if DENOISER_TYPE >= 4:
  # pass the noisy images through the denoiser model
  decoded_imgs = denoiser(x_test_noisy)
else:
  # encode the noisy images from the test set
  encoded_imgs = denoiser.encoder(x_test_noisy).numpy()

  # decode the encoded images
  decoded_imgs = denoiser.decoder(encoded_imgs).numpy()

# plot both the noisy images and the denoised images produced by the denoiser autoencoder
n = number_of_test_denoise_to_plot
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original + noise
  ax = plt.subplot(2, n, i + 1)
  plt.title("original + noise")
  plt.imshow(tf.squeeze(x_test_noisy[i]))
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