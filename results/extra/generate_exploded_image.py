#!venv/bin/python3

import os
import cv2
import numpy as np
import shutil

def merge_patches_into_exploded_image(patches_folder, output_filename, spacing=10, replace_red=False):
  # Get all patch filenames in the given folder
  patch_files = [f for f in os.listdir(patches_folder) if f.endswith('.jpg')]

  # Extract idx and x coordinates from filenames for sorting and placement
  def extract_values(filename):
    idx = int(filename.split('_')[0])
    x = int(filename.split('_')[1].split('x')[1])
    return idx, x

  patch_files.sort(key=extract_values)

  # Read the first patch to get its dimensions
  sample_patch = cv2.imread(os.path.join(patches_folder, patch_files[0]))
  patch_height, patch_width = sample_patch.shape[:2]

  # Extract unique x-values and sort them
  unique_x_values = sorted(list(set([extract_values(f)[1] for f in patch_files])))

  # Calculate the number of patches in a row based on the unique x-values
  patches_in_row = len(unique_x_values)

  # Determine the canvas size without extra spacing on the right and bottom
  canvas_width = (patch_width * patches_in_row) + (spacing * (patches_in_row - 1))
  canvas_height = (patch_height * patches_in_row) + (spacing * (patches_in_row - 1))
  canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

  # Dictionary to keep track of the current y position for each x-value
  current_y_for_x = {x_val: 0 for x_val in unique_x_values}

  for patch_file in patch_files:
    # Get the idx and x values from the filename
    _, x_value = extract_values(patch_file)

    grid_x = unique_x_values.index(x_value)
    grid_y = current_y_for_x[x_value]

    # Update the current y position for the given x-value
    current_y_for_x[x_value] += 1

    # Determine the effective position on the canvas, considering the spacing
    effective_x = grid_x * (patch_width + spacing)
    effective_y = grid_y * (patch_height + spacing)

    # Place the patch on the canvas
    patch_img = cv2.imread(os.path.join(patches_folder, patch_file))
    
    # Replace red pixels with white if flag is set
    if replace_red:
      red_pixels = (patch_img[:,:,0] < 50) & (patch_img[:,:,1] < 50) & (patch_img[:,:,2] > 200)
      patch_img[red_pixels] = [255, 255, 255]

    canvas[effective_y:effective_y+patch_height, effective_x:effective_x+patch_width] = patch_img

  # Save the resulting canvas
  cv2.imwrite(output_filename, canvas)
    
# Remove the images output folder if it exists
if os.path.exists("images/patches"):
  shutil.rmtree("images/patches")

if not os.path.exists("images/patches"):
  os.makedirs("images/patches")

# Do it
for p in [0]:
  merge_patches_into_exploded_image(f"patches/denoising/original/{p}/nored/", f"images/patches/sample.exploded.p{p}.wb.jpg", spacing=10-p, replace_red=False)
  merge_patches_into_exploded_image(f"patches/denoising/noised/{p}/nored/", f"images/patches/sample.noised.exploded.p{p}.wb.jpg", spacing=10-p, replace_red=False)

for p in [6, 8]:
  
  merge_patches_into_exploded_image(f"patches/denoising/original/{p}/nored/", f"images/patches/sample.exploded.p{p}.margin-original.wb.jpg", spacing=20, replace_red=False)
  merge_patches_into_exploded_image(f"patches/denoising/noised/{p}/nored/", f"images/patches/sample.noised.exploded.p{p}.margin-original.wb.jpg", spacing=20, replace_red=False)
  merge_patches_into_exploded_image(f"patches/denoising/original/{p}/red/", f"images/patches/sample.exploded.p{p}.margin-red.wb.jpg", spacing=20, replace_red=False)
  merge_patches_into_exploded_image(f"patches/denoising/noised/{p}/red/", f"images/patches/sample.noised.exploded.p{p}.margin-red.wb.jpg", spacing=20, replace_red=False)
  merge_patches_into_exploded_image(f"patches/denoising/original/{p}/red/", f"images/patches/sample.exploded.p{p}.margin-white.wb.jpg", spacing=20, replace_red=True)
  merge_patches_into_exploded_image(f"patches/denoising/noised/{p}/red/", f"images/patches/sample.noised.exploded.p{p}.margin-white.wb.jpg", spacing=20, replace_red=True)
