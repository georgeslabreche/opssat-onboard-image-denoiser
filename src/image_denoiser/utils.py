import tensorflow as tf
from noiser import generate_fixed_noise_pattern, add_noise_to_image

# make sure constants are set as desired before training
from constants import *


def load_and_preprocess_image(path, resize=False):
  ''' load and preprocess image '''

  # read image
  image = tf.io.read_file(path)

  # decode image in desired channel
  image = tf.image.decode_jpeg(image, channels=DESIRED_CHANNELS)

  # resize the images
  if resize is True:
    image  = tf.image.resize(image, [DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH])

  # noise pattern
  noise_pattern = None

  # pattern size for the FNP noise
  if NOISE_TYPE in [1, 2]:
    pattern_size = (DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS) if NOISE_TYPE == 1 else (DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
    noise_pattern = generate_fixed_noise_pattern(pattern_size, NOISE_FACTOR)

  # add noise to image
  image_noisy = add_noise_to_image(image, noise_pattern, NOISE_FACTOR, NOISE_TYPE)

  # convert the images data to float32
  image       = tf.cast(image, tf.float32)
  image_noisy = tf.cast(image_noisy, tf.float32)

  # normalize the images to [0,1] range
  image /= 255.0
  image_noisy /= 255.0

  # return the image and noisy image
  return image, image_noisy