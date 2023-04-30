import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, losses

# make sure constants are set as desired before training
from constants import *

# autoencoders
from autoencoders import NaiveDenoiser, SimpleDenoiser

# function to resize and normalize the input images
def preprocess_image(file_path):

  # read image file
  image = tf.io.read_file(file_path)

  # decode image in desired channel
  image = tf.image.decode_jpeg(image, channels=DESIRED_CHANNELS)

  # resize image in desired 
  image = tf.image.resize(image, [DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH])
  
  # normalization
  image /= 255.0

  return image


# list the image files
list_image_files = tf.data.Dataset.list_files(DIR_PATH_IMAGES_EARTH + "/*.jpeg")

# get the number of files
num_files = len(list(list_image_files))
print("Number of images files in the dataset:", num_files)

# shuffle the dataset
list_image_files = list_image_files.shuffle(buffer_size=1000, seed=42)

# calculate the number of files in the training set
num_train = int(num_files * TRAIN_RATIO)

# separate the dataset into training and testing sets
train_data = list_image_files.take(num_train)
test_data = list_image_files.skip(num_train)

# load the images for both training and testing datasets
train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

# convert the train and test datasets to NumPy arrays
x_train = np.array(list(train_data))
x_test = np.array(list(test_data))

# print the shapes of the train and test datasets
print("Train images shape:", x_train.shape)
print("Test images shape:", x_test.shape)

# adding random noise to the images using a normal distribution
x_train_noisy = x_train + NOISE_FACTOR * tf.random.normal(shape=x_train.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)

x_test_noisy = x_test + NOISE_FACTOR * tf.random.normal(shape=x_test.shape)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# another way of adding noise
# use NumPy's random normal distribution centered at 0.5 with a standard deviation of 0.5
#train_noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
#x_train_noisy = np.clip(x_train + train_noise, 0, 1)

# another way of adding noise, using numpy
#test_noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
#x_test_noisy = np.clip(x_test + test_noise, 0, 1)


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
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original + noise
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
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
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

# show the reconstruction
plt.show()
