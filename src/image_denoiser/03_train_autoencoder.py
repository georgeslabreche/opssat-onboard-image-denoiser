#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf # tensorflow
from tensorflow.keras import layers, losses
from autoencoders import NaiveDenoiser, SimpleDenoiser
from utils import *

# make sure constants are set as desired before training
from constants import *


# increase this when running on a proper ML training computer with GPU
# set to None to train with all available training data
TRAINING_DATA_SAMPLE_SIZE = 3000

# list the image files
list_image_files = tf.data.Dataset.list_files(DIR_PATH_IMAGERY_TRAIN + "/*.jpg")

# get the number of files
num_files = len(list(list_image_files))
print("Images in the dataset:", num_files)

# shuffle the dataset
list_image_files = list_image_files.shuffle(buffer_size=1000, seed=42)

# Take a subsample only in dev environment.
if TRAINING_DATA_SAMPLE_SIZE is not None:
  list_image_files = list_image_files.take(TRAINING_DATA_SAMPLE_SIZE)

  # get the number of files
  num_files = len(list(list_image_files))
  print("Images sampled from the dataset:", num_files)

# calculate the number of files in the training set
num_train = int(num_files * TRAIN_RATIO)

# separate the image file path datasets into training and testing sets
list_image_files_train = list_image_files.take(num_train)
list_image_files_test  = list_image_files.skip(num_train)

# load an preprocess the images
train_data = list_image_files_train.map(lambda x: load_and_preprocess_image(x, resize=False))
test_data  = list_image_files_test.map(lambda x: load_and_preprocess_image(x, resize=False))

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
  denoiser = NaiveDenoiser(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
elif DENOISER_TYPE == 2:
  denoiser = SimpleDenoiser(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
else:
  print(f"Error: unsupported denoiser encoder typel: {DENOISER_TYPE}")
  quit()

# compile
denoiser.compile(optimizer='adam', loss=losses.MeanSquaredError())

# train the denoiser autoencoder
# use the training dataset
denoiser.fit(x_train_noisy, x_train,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  shuffle=True,
  validation_data=(x_test_noisy, x_test))

# save the model
denoiser.save(MODEL_PATH)

# convert the model to a tflite mode and save
converter = tf.lite.TFLiteConverter.from_keras_model(denoiser)
tflite_denoiser = converter.convert()

# save the tflite model
with open(MODEL_PATH + '.tflite', 'wb') as f:
  f.write(tflite_denoiser)

# print encoder and decoder summaries
denoiser.encoder.summary()
denoiser.decoder.summary()

# encode the noisy images from the test set
encoded_imgs = denoiser.encoder(x_test_noisy).numpy()

# decode the encoded images
decoded_imgs = denoiser.decoder(encoded_imgs).numpy()

# plot both the noisy images and the denoised images produced by the denoiser autoencoder
n = 10
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
