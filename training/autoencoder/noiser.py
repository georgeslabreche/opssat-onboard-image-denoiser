import tensorflow as tf
import numpy as np
import math
import time


def generate_gaussian_noise(mean, std_dev):
  ''' generate Gaussian noise '''
  u1 = np.random.uniform(0, 1)
  u2 = np.random.uniform(0, 1)

  # the Box-Muller transform
  rand_std_normal = math.sqrt(-2.0 * np.log(u1)) * math.cos(2.0 * math.pi * u2)

  # random normal distribution(mean, stdDev^2)
  rand_normal = mean + std_dev * rand_std_normal

  return rand_normal


def generate_fixed_pattern_noise(size, noise_factor):
  ''' generate a fixed-pattern noise (FPN) '''
  np.random.seed(1) 
  return (noise_factor * np.random.rand(*size)).astype(np.uint8)


def add_noise_to_image(image, noise_pattern, noise_factor, noise_type):
  ''' add noise to image '''

  if noise_type == 0:
    # Gaussian noise
    np.random.seed(int(time.time()))
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0)
    image = tf.cast(image, tf.float32) + noise_factor * noise

  elif noise_type in [1, 2]:
    # fixed pattern noise (FPN) or column fixed pattern noise (FPN)
    image = image + noise_pattern
  else:
    print(f'Invalid noise type: {noise_type}')
    exit(1)

  # clamp the values to [0, 255]
  image = tf.clip_by_value(image, 0, 255)

  # return the noisy image
  return tf.cast(image, tf.uint8)
