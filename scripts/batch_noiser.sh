#!/bin/bash

# this script applies noise to the training images
# the resulting noisy images are used to train denoiser models

# the file path the the noiser executable binary
noiser_bin="../tools/noiser/build/noiser"

if [ ! -f "$noiser_bin" ]; then
  echo "noiser executable binary does not exist"
  echo "you must build it first"
  echo "see instructions in tools/noiser/BUILD.md"
  exit 1
fi

# directory containing the jpeg images
dir_img="$1/unnoised/original"

# create directory that will contain the noisy images
dir_noisy_parent="$1/noised/original"
mkdir -p $dir_noisy_parent

# target size
resize="224x224"

# the noise type label
noise_type_label=""

# apply different noise types
for noise_type in 1 2; do

  # ternary-like operation to set the noise type string
  [[ $noise_type -eq 1 ]] && noise_type_label="fpn" || noise_type_label="cfpn"

  # apply different noise factors
  for noise_factor in 50 100 150 200; do
    echo "applying $noise_type_label at noise factor $noise_factor..."
  
    # create the noise type and factor directory if it doesn't exist already
    dir_noisy_dest=$dir_noisy_parent/$noise_type_label/$noise_factor
    mkdir -p $dir_noisy_dest

    # loop over all jpeg files
    for img_file in "$dir_img"/*.jpeg; do

      # get the basename of the image file
      img_file_basename=$(basename "$img_file")

      # execute the noiser on the current image
      "$noiser_bin" -i "$img_file" -n $noise_factor -t $noise_type -w 1 -r $resize -q 100 > /dev/null 2>&1

      # move the noisy image to the target directory
      filename_base=$(basename "$img_file" .jpeg)
      filename_noisy="${filename_base}.noised.jpeg"
      mv $dir_img/$filename_noisy $dir_noisy_dest/$img_file_basename
    done
  done
done

# great success
echo "Qapla'"
exit 0