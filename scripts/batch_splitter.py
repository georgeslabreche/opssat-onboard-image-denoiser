#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this script is for splitting images into patches
it resizes images to and then splits them into patches
each patch is saved as a separate image
"""

import argparse
import os
from PIL import Image

# initialize the ArgumentParser object
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('-r', '--resize', type=int, help="Resize the image input, e.g. 224 for 224x224")
parser.add_argument('-s', '--size', type=int, help="Target size of the split outputs, e.g. 56 for 56x56")
parser.add_argument('-v', '--validation', action='store_true', default=False, help="If set, process validation set")
parser.add_argument('-n', '--noised', action='store_true', default=False, help="If set, process noised images")


# parse the arguments
args = parser.parse_args()

INPUT_SIZE = args.resize
SPLIT_SIZE = args.size

# check that arguments have been set
if SPLIT_SIZE is None:
  print("Missing required argument: --size")
  exit(1)

# directory containing the jpeg images
dir_img = f"../data/opssat/{'validate' if args.validation else 'earth'}/{'noised' if args.noised else 'unnoised'}/original"

# create directory that will contain the split images
dir_split_img = f"../data/opssat/{'validate' if args.validation else 'earth'}/{'noised' if args.noised else 'unnoised'}/split/{SPLIT_SIZE}"
os.makedirs(dir_split_img, exist_ok=True)

# function to split image into patches and save them
def split_image(img_path, input_size, split_size, save_dir):
  img = Image.open(img_path)
  img_name = os.path.splitext(os.path.basename(img_path))[0]
  
  # resize image to before splitting
  if input_size is not None:
    img = img.resize((input_size, input_size))

  width, height = img.size

  # keep track of x,y coordinates of the patches
  # use them in the filename
  x_coord = 0
  y_coord = 0
  
  for i in range(0, width, split_size):
    for j in range(0, height, split_size):
      box = (i, j, i+split_size, j+split_size)
      patch = img.crop(box)
      
      # make a directory for the original image if it doesn't exist
      img_dir = os.path.join(save_dir, img_name)
      os.makedirs(img_dir, exist_ok=True)
      
      # create patch filename
      patch_name = f"{img_name}_{x_coord}_{y_coord}.jpeg"
      patch_path = os.path.join(img_dir, patch_name)
      
      # save patch to file
      patch.save(patch_path)
      
      # increment y-coordinate
      y_coord += 1
      
    # increment x-coordinate
    x_coord += 1
    
    # reset y-coordinate:
    y_coord = 0

# iterate over each image in dir_img and split it into patches
def iterate_images(dir_img, dir_split_img):
  for filename in os.listdir(dir_img):
    if filename.endswith('.jpeg'):
      img_path = os.path.join(dir_img, filename)
      split_image(img_path, INPUT_SIZE, SPLIT_SIZE, dir_split_img)

if args.noised:
  for noise_type in ['fnp', 'cfnp']:
    for noise_factor in [50, 100, 150, 200]:

      # redefine path of the noised images
      noised_dir_img = f"{dir_img}/{noise_type}/{noise_factor}"
      noised_dir_split_img = f"{dir_split_img}/{noise_type}/{noise_factor}"
      
      # iterate over each image in dir_img and split it into patches
      iterate_images(noised_dir_img, noised_dir_split_img)

else:
  # iterate over each image in dir_img and split it into patches
  iterate_images(dir_img, dir_split_img)
