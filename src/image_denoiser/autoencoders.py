import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# define a convolutional autoencoder for a naive denoiser
# use Conv2D layers in the encoder and Conv2DTranspose layers in the decoder
# encoder and decoder taken from: https://www.tensorflow.org/tutorials/generative/autoencoder
class DenoiseNaiveAutoencoder(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseNaiveAutoencoder, self).__init__()
    
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
class DenoiseSimpleAutoencoder(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseSimpleAutoencoder, self).__init__()

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


class DenoiseComplexAutoencoder(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseComplexAutoencoder, self).__init__()

    # the shape of the image input
    self.input_height = input_height
    self.input_width = input_width
    self.input_channels = input_channels

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(self.input_height, self.input_width, self.input_channels)),
      layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  

# a denoiser that uses skip connections
class DenoiseSkipAutoencoder(Model):
  def __init__(self):
    super(DenoiseSkipAutoencoder, self).__init__()

    # encoder
    self.encoder_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    self.encoder_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    self.encoder_conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')

    # used for downsampling
    self.pool = layers.MaxPooling2D((2, 2), strides=2)

    # decoder
    self.decoder_conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
    self.decoder_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    self.decoder_conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')

    # used for upsampling
    self.upsample = layers.UpSampling2D((2, 2))

    # final output layer
    self.final_conv = layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')

  def call(self, x):
    # encoder
    conv1 = self.encoder_conv1(x)
    x = self.pool(conv1)
    
    conv2 = self.encoder_conv2(x)
    x = self.pool(conv2)
    
    x = self.encoder_conv3(x)
    
    # decoder
    x = self.upsample(x)
    x = tf.concat([x, conv2], axis=-1) # skip connection
    x = self.decoder_conv1(x)
    
    x = self.upsample(x)
    x = tf.concat([x, conv1], axis=-1) # skip connection
    x = self.decoder_conv2(x)

    x = self.decoder_conv3(x)

    return self.final_conv(x)