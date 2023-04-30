import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# flag to display or not the test noisy images
DISPLAY_TEST_NOISE = False

# training epochs
EPOCHS = 10

# import the dataset to omit the modifications made earlier
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)

# adding random noise to the images
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

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
  plt.show()

# define a convolutional autoencoder
# use Conv2D layers in the encoder and Conv2DTranspose layers in the decoder
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()

    # Conv2D layers in the encoder
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    # Conv2DTranspose layers in the decoder
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
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
  shuffle=True,
  validation_data=(x_test_noisy, x_test))

denoiser.save('models/denoiser')

# convert the model to a tflite mode and save
converter = tf.lite.TFLiteConverter.from_keras_model(denoiser)
tflite_denoiser = converter.convert()

# save the tflite model
with open('models/denoiser.tflite', 'wb') as f:
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
plt.show()