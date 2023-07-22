import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2

# define a convolutional autoencoder for a naive denoiser
# use Conv2D layers in the encoder and Conv2DTranspose layers in the decoder
# encoder and decoder taken from: https://www.tensorflow.org/tutorials/generative/autoencoder
class DenoiseAutoencoderNaive(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseAutoencoderNaive, self).__init__()
    
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
class DenoiseAutoencoderSimple(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseAutoencoderSimple, self).__init__()

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


class DenoiseAutoencoderComplex(Model):
  def __init__(self, input_height, input_width, input_channels):
    super(DenoiseAutoencoderComplex, self).__init__()

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
class DenoiseAutoencoderSkipConnection(Model):
  def __init__(self):
    super(DenoiseAutoencoderSkipConnection, self).__init__()

    # encoder
    self.encoder_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
    self.encoder_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    self.encoder_conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    self.encoder_conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')

    self.pool = layers.MaxPooling2D((2, 2), strides=2) # Used for downsampling

    # decoder
    self.decoder_conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
    self.decoder_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    self.decoder_conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    self.decoder_conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')

    self.upsample = layers.UpSampling2D((2, 2)) # Used for upsampling

    # final output layer
    self.final_conv = layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')

  def call(self, x):
    # encoder
    conv1 = self.encoder_conv1(x)
    x = self.pool(conv1)

    conv2 = self.encoder_conv2(x)
    x = self.pool(conv2)

    conv3 = self.encoder_conv3(x)
    x = self.pool(conv3)

    x = self.encoder_conv4(x)

    # decoder
    x = self.upsample(x)
    x = tf.concat([x, conv3], axis=-1) # skip connection
    x = self.decoder_conv1(x)

    x = self.upsample(x)
    x = tf.concat([x, conv2], axis=-1) # skip connection
    x = self.decoder_conv2(x)

    x = self.upsample(x)
    x = tf.concat([x, conv1], axis=-1) # skip connection
    x = self.decoder_conv3(x)

    x = self.decoder_conv4(x)

    return self.final_conv(x)


class DenoiseAutoencoderVGG16(Model):
  def __init__(self):
    super(DenoiseAutoencoderVGG16, self).__init__()

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    self.encoder = Model(base_model.input, [base_model.get_layer('block1_conv2').output, 
                                                    base_model.get_layer('block2_conv2').output,
                                                    base_model.get_layer('block3_conv3').output,
                                                    base_model.get_layer('block4_conv3').output,
                                                    base_model.get_layer('block5_conv3').output])

    self.upsample = layers.UpSampling2D(size=(2, 2))  # define the upsampling operation

    self.upsample1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
    self.upsample2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
    self.upsample3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
    self.upsample4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
    self.final_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')

  def call(self, inputs):
    encoded1, encoded2, encoded3, encoded4, encoded5 = self.encoder(inputs)

    x1 = self.upsample1(encoded5)
    x2 = self.upsample2(x1)
    x3 = self.upsample3(x2)
    x4 = self.upsample4(x3)
    decoded = self.final_layer(x4)

    return decoded


class DenoiseAutoencoderSkipConnectionVGG16(Model):
  def __init__(self):
    super(DenoiseAutoencoderSkipConnectionVGG16, self).__init__()

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    # Fixme: not needed? Just do self.encoder = base_model?
    self.encoder = Model(base_model.input, [base_model.get_layer('block1_conv2').output, 
                                                    base_model.get_layer('block2_conv2').output,
                                                    base_model.get_layer('block3_conv3').output,
                                                    base_model.get_layer('block4_conv3').output,
                                                    base_model.get_layer('block5_conv3').output])

    # fixme: not needed
    self.upsample = layers.UpSampling2D(size=(2, 2))  # define the upsampling operation

    self.upsample1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
    self.upsample2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
    self.upsample3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
    self.upsample4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
    self.final_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')

  def call(self, inputs):
    encoded1, encoded2, encoded3, encoded4, encoded5 = self.encoder(inputs)

    x1 = layers.Concatenate()([self.upsample1(encoded5), encoded4])
    x2 = layers.Concatenate()([self.upsample2(x1), encoded3])
    x3 = layers.Concatenate()([self.upsample3(x2), encoded2])
    x4 = layers.Concatenate()([self.upsample4(x3), encoded1])
    decoded = self.final_layer(x4)

    return decoded


class DenoiseAutoencoderMobileNetV2(Model):
  def __init__(self, **kwargs):
    super(DenoiseAutoencoderMobileNetV2, self).__init__(**kwargs)
    # Load MobileNetV2 as encoder
    self.encoder = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # Freeze the encoder
    self.encoder.trainable = False
    # The layers for the decoder
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
      layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid'),  # RGB channels
      layers.UpSampling2D()  # Additional upsampling layer
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# FIXME: this doesn't work
class DenoiseAutoencoderSkipConnectionMobileNetV2(Model):
  def __init__(self, **kwargs):
    super(DenoiseAutoencoderSkipConnectionMobileNetV2, self).__init__(**kwargs)

    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Redefining encoder model with those layers
    # fixme: use base_model?
    self.encoder = Model(base_model.input,[
      base_model.get_layer('block_2_project').output,
      base_model.get_layer('block_5_project').output,
      base_model.get_layer('block_9_project').output,
      base_model.get_layer('block_12_project').output,
      base_model.get_layer('block_15_project').output,
      base_model.get_layer('block_16_project').output])


    # Freeze the encoder
    self.encoder.trainable = False

    self.upsample1 = layers.Conv2DTranspose(320, (3, 3), strides=2, padding='same', activation='relu')
    self.upsample2 = layers.Conv2DTranspose(160, (3, 3), strides=2, padding='same', activation='relu')
    self.upsample3 = layers.Conv2DTranspose(96, (3, 3), strides=2, padding='same', activation='relu')
    self.upsample4 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')
    self.upsample5 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')
    self.upsample6 = layers.Conv2DTranspose(24, (3, 3), strides=2, padding='same', activation='relu')
    self.final_layer = layers.Conv2D(3, (3, 3), strides=2, padding='same', activation='sigmoid')

    self.upsample_skip = layers.UpSampling2D()  # For resizing skip connection layers to match upsampled layers

    self.upsample_skip_2x = layers.UpSampling2D(size=(2, 2))  # Upsamples by a factor of 2
    self.upsample_skip_4x = layers.UpSampling2D(size=(4, 4))  # Upsamples by a factor of 4

  def call(self, inputs):
    # Encoder
    encoded1, encoded2, encoded3, encoded4, encoded5, encoded6 = self.encoder(inputs)

    x1 = layers.Concatenate()([self.upsample1(encoded6), self.upsample_skip_2x(encoded5)])
    x2 = layers.Concatenate()([self.upsample2(x1), self.upsample_skip_2x(encoded4)])
    x3 = layers.Concatenate()([self.upsample3(x2), self.upsample_skip_4x(encoded3)])
    x4 = layers.Concatenate()([self.upsample4(x3), self.upsample_skip_4x(encoded2)])
    x5 = layers.Concatenate()([self.upsample5(x4), self.upsample_skip_4x(encoded1)])
    decoded = self.final_layer(self.upsample_skip_2x(x5))

    return decoded
  
