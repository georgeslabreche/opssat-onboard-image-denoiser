import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# make sure constants are set as desired before training
from constants import *


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

# adding random noise to the images
x_train_noisy = x_train + NOISE_FACTOR * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + NOISE_FACTOR * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


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


# define a convolutional autoencoder
# use Conv2D layers in the encoder and Conv2DTranspose layers in the decoder
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()

    # Conv2D layers in the encoder that applies two convolutional layers with downsampling
    # the strides=2 parameter indicates that:
    #  - the convolution operation moves by two pixels at a time
    #  - i.e. the layer downsamples the input by a factor of 2
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)),
      #layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
    ])

    # Conv2DTranspose layers in the decoder
    # the layers upsample the input by a factor of 2 (strides=2)
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      #layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# the denoiser autoencoder
denoiser = Denoise()

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
