import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# define a convolutional autoencoder for a naive denoiser
# use Conv2D layers in the encoder and Conv2DTranspose layers in the decoder
# encoder and decoder taken from: https://www.tensorflow.org/tutorials/generative/autoencoder
class NaiveDenoiser(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(NaiveDenoiser, self).__init__()
    
    # the shape of the image input
    self.input_height = input_height
    self.input_width = input_width
    self.input_channels = input_channels

    # Conv2D layers in the encoder that applies two convolutional layers with downsampling
    # the strides=2 parameter indicates that:
    #  - the convolution operation moves by two pixels at a time
    #  - i.e. the layer downsamples the input by a factor of 2
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(self.input_height, self.input_width, self.input_channels)),
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


# a denoiser that's not so naive but still simple
# encoder and decoder implementation taken from https://keras.io/examples/vision/autoencoder/
class SimpleDenoiser(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(SimpleDenoiser, self).__init__()

    # the shape of the image input
    self.input_height = input_height
    self.input_width = input_width
    self.input_channels = input_channels

    # encoder
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(self.input_height, self.input_width, self.input_channels)),
      layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2), padding="same"),
      layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2), padding="same")
    ])

    # decoder
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
      layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
      layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
