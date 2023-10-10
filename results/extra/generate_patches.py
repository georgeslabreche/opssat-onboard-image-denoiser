#!venv/bin/python3

import os
import cv2
import numpy as np
import random
import math
import shutil

# Constants
BATCH_SIZE = 64
BATCH_SHAPE = [BATCH_SIZE, 224, 224, 3]
PATCH_NUM = 50
PATCH_SIZE = BATCH_SHAPE[1] // 4
PATCH_SHAPE = [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3]

# Output directories
output_dir_training_original = "patches/training/original"
output_dir_training_noised = "patches/training/noised"

output_dir_denoising_original = "patches/denoising/original"
output_dir_denoising_noised = "patches/denoising/noised"


def generate_training_patches(raw, noise, patch_num=PATCH_NUM, patch_size=PATCH_SHAPE[1]):
  out_raw = []
  out_noise = []
  max_x_y = raw.shape[1] - patch_size

  for n in range(raw.shape[0]):
    for pn in range(patch_num):
      rx = random.randint(0, max_x_y)
      ry = random.randint(0, max_x_y)
      rf = random.choice([-1, 0, 1, None])
      
      if rf is not None:
        raw_patch = cv2.flip(raw[n], rf)[rx:rx + patch_size, ry:ry + patch_size, :]
        noise_patch = cv2.flip(noise[n], rf)[rx:rx + patch_size, ry:ry + patch_size, :]
      else:
        raw_patch = raw[n][rx:rx + patch_size, ry:ry + patch_size, :]
        noise_patch = noise[n][rx:rx + patch_size, ry:ry + patch_size, :]

      out_raw.append(raw_patch)
      out_noise.append(noise_patch)

      # Save patches with x, y coordinates
      patch_name = f"{n}_x{rx}_y{ry}.jpg"

      # Ensure directories exist
      if not os.path.exists(output_dir_training_original):
        os.makedirs(output_dir_training_original)

      if not os.path.exists(output_dir_training_noised):
        os.makedirs(output_dir_training_noised)

      # Write patches
      cv2.imwrite(os.path.join(output_dir_training_original, patch_name), raw_patch)
      cv2.imwrite(os.path.join(output_dir_training_noised, patch_name), noise_patch)


def safe_get_pixel_value(img, x, y, c, width, height, channels):
  # Ensure the coordinates are within the image boundaries
  x = max(0, min(x, width - 1))
  y = max(0, min(y, height - 1))
  return img[y, x, c]


def generate_denoising_patches(img, patch_size, patch_margin, output_dir, red_out=False):
  input_height, input_width, channels = img.shape
  
  step_w = patch_size - 2 * patch_margin
  step_h = patch_size - 2 * patch_margin
  
  pwidth_max = 1 + math.ceil((input_width - patch_size) / float(step_w))
  pheight_max = 1 + math.ceil((input_height - patch_size) / float(step_h))
  
  if not os.path.exists(f"{output_dir}/{patch_margin}/{'red' if red_out is True else 'nored'}"):
    os.makedirs(f"{output_dir}/{patch_margin}/{'red' if red_out is True else 'nored'}")
    
  idx = 0
  for w in range(pwidth_max):
    for h in range(pheight_max):
      start_i = w * step_w
      start_j = h * step_h
      
      if w == pwidth_max - 1:
        start_i = input_width - patch_size
      if h == pheight_max - 1:
        start_j = input_height - patch_size
        
      patch = np.zeros((patch_size, patch_size, channels), dtype=np.uint8)
      
      for j in range(start_j, start_j + patch_size):
        for i in range(start_i, start_i + patch_size):
          for k in range(channels):
            pixel_val = safe_get_pixel_value(img, i, j, k, input_width, input_height, channels)
            patch[j - start_j, i - start_i, k] = pixel_val
            
            # Red-out the pixels that will be discarded
            if red_out and ((j - start_j < patch_margin) or (j - start_j >= patch_size - patch_margin) or
                           (i - start_i < patch_margin) or (i - start_i >= patch_size - patch_margin)):
              patch[j - start_j, i - start_i] = [0, 0, 255]

      cv2.imwrite(f"{output_dir}/{patch_margin}/{'red' if red_out is True else 'nored'}/{idx}_x{start_i}_y{start_j}.jpg", patch)
      idx += 1

# Remove the patches folder if it exists
if os.path.exists("patches"):
  shutil.rmtree("patches")

# Load the images
original_img = cv2.imread("sample.wb.jpeg")
original_img = cv2.resize(original_img, (224, 224))
noised_img = cv2.imread("images/samples/sample.wb.cfpn100.noised.jpeg")

# Get training patches
original_images = np.array([original_img])
noised_images = np.array([noised_img])
generate_training_patches(original_images, noised_images)

# Get denoising patches
for patch_margin in [0]:
  generate_denoising_patches(original_img, PATCH_SIZE, patch_margin, output_dir_denoising_original, red_out=False)
  generate_denoising_patches(noised_img, PATCH_SIZE, patch_margin, output_dir_denoising_noised, red_out=False)

for patch_margin in [6, 8]:
  generate_denoising_patches(original_img, PATCH_SIZE, patch_margin, output_dir_denoising_original, red_out=True)
  generate_denoising_patches(noised_img, PATCH_SIZE, patch_margin, output_dir_denoising_noised, red_out=True)

  generate_denoising_patches(original_img, PATCH_SIZE, patch_margin, output_dir_denoising_original, red_out=False)
  generate_denoising_patches(noised_img, PATCH_SIZE, patch_margin, output_dir_denoising_noised, red_out=False)
