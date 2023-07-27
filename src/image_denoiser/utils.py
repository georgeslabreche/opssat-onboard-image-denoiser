import os
import tensorflow as tf
from noiser import generate_fixed_noise_pattern, add_noise_to_image

# make sure constants are set as desired before training
from constants import *


def load_and_preprocess_image(path, resize=False, apply_noise=False, noise_type=1, noise_factor=50):
  ''' load and preprocess image 
 
    @param path: variable is a TensorFlow tensor
    @param resize: the image resize flag
    @param noise_type: the noise type, 1 for NFP and 2 for cNFP
    @param noise_factor: the noise factor
  '''

  # read image
  image = tf.io.read_file(path)

  # decode image in desired channel
  image = tf.image.decode_jpeg(image, channels=DESIRED_CHANNELS)

  # resize the images
  if resize is True:
    image  = tf.image.resize(image, [DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH])

  # generate noisy images in-memory
  image_noisy = None
  if apply_noise is True:

    # noise pattern
    noise_pattern = None
  
    # pattern size for the FNP noise
    if noise_type in [1, 2]:
      pattern_size = (DESIRED_INPUT_HEIGHT, DESIRED_INPUT_WIDTH, DESIRED_CHANNELS) if noise_type == 1 else (DESIRED_INPUT_WIDTH, DESIRED_CHANNELS)
      noise_pattern = generate_fixed_noise_pattern(pattern_size, noise_factor)

    # add noise to image
    image_noisy = add_noise_to_image(image, noise_pattern, noise_factor, noise_type)

    # convert the images data to float32
    image_noisy = tf.cast(image_noisy, tf.float32)

    # normalize the images to [0,1] range
    image_noisy /= 255.0


  # convert the original images data to float32
  image = tf.cast(image, tf.float32)

  # normalize the original images to [0,1] range
  image /= 255.0

  # return the image and noisy image
  return image, image_noisy


def load_and_preprocess_image_pair(original_path, noisy_path, resize_original=False, resize_noisy=False):
  # load and preprocess the original image
  original_image, _ = load_and_preprocess_image(original_path, resize_original)

  # load and preprocess the noisy image
  noisy_image, _ = load_and_preprocess_image(noisy_path, resize_noisy)

  return original_image, noisy_image
