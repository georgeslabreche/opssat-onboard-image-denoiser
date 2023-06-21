#!/usr/bin/env python3

import os
import rasterio
from rasterio.merge import merge
import numpy as np
from PIL import Image
from constants import *

# target date
TARGET_DATE = None #20180610

def merge_bands(red_band_tif, green_band_tif, blue_band_tif, save_as_jpeg=True):
  ''' merge the RGB bands tif file into a single tif file '''

  # get the parent directory path
  dir_parent_path = os.path.dirname(red_band_tif)

  # get the parent directory name so it can be used as the filename for the merged RGB tif
  ext = 'jpg' if save_as_jpeg else 'tif'
  merged_filename = f'{os.path.basename(dir_parent_path)}.{ext}'
  merged_filepath = f'{DIR_PATH_IMAGERY_TRAIN}/{merged_filename}'

  # open the bands and read them as numpy arrays
  with rasterio.open(red_band_tif) as src:
    red_band = src.read(1)
  with rasterio.open(green_band_tif) as src:
    green_band = src.read(1)
  with rasterio.open(blue_band_tif) as src:
    blue_band = src.read(1)

  # define a function to scale the bands
  def scale_band(band):
    return ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)

  # scale the bands
  red_band_scaled = scale_band(red_band)
  green_band_scaled = scale_band(green_band)
  blue_band_scaled = scale_band(blue_band)

  # stacked bands
  # use np.dstack for jpeg and np.stack for tif
  stacked_bands = \
    np.dstack((red_band_scaled, green_band_scaled, blue_band_scaled)) \
    if save_as_jpeg \
    else np.stack((red_band_scaled, green_band_scaled, blue_band_scaled))

  # calculate the average pixel value
  # we want to skip images that are too dark
  average_pixel_value = np.mean(stacked_bands)

  # set a threshold for the average pixel value below which an image is considered 'too dark'
  threshold = 20

  # skip the image if the average pixel value is less or equal than the threshold
  if average_pixel_value <= threshold:
    print(f'Skipped due to low brightness {merged_filename}')
    return

  # save the image
  print(f'Write {merged_filename}')

  # save as jpeg
  if save_as_jpeg is True:
    # write
    Image.fromarray(stacked_bands).save(merged_filepath)

  else: # save as tif

    # copy the metadata from one of the input files
    with rasterio.open(red_band_tif) as src:
      meta = src.meta

    # update the metadata to reflect the number of layers
    meta.update(count=3, dtype=np.uint8)

    # Write the stacked bands to a new file
    with rasterio.open(merged_filepath, 'w', **meta) as dst:
      for i, band in enumerate(stacked_bands, start=1):
        dst.write(band, i)


# walk through each directory in the parent directory
for dirpath, dirnames, filenames in os.walk(DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA):

  # only process for desired date (if a target date is set)
  if TARGET_DATE is None or str(TARGET_DATE) in dirpath:

    # check if all files (B04.tif, B03.tif, B02.tif) exist in the directory
    if 'B04.tif' in filenames and 'B03.tif' in filenames and 'B02.tif' in filenames:

      # get file path of the RGB tif files
      red_b04_tif = os.path.join(dirpath, 'B04.tif')
      green_b03_tif = os.path.join(dirpath, 'B03.tif')
      blue_b02_tif = os.path.join(dirpath, 'B02.tif')

      # merge the RGB bands into a single tif
      merge_bands(red_b04_tif, green_b03_tif, blue_b02_tif)

    else:
      print(f'Skipped {dirpath}')