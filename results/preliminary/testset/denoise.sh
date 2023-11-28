#!/bin/bash

# Output directories that will contain the denoised images
mkdir -p denoised/AE/full
mkdir -p denoised/AE/patch
mkdir -p denoised/WGAN/full
mkdir -p denoised/WGAN/patch

# Define model and input types
model_types=("AE" "WGAN")
input_types=("full" "patch")

# Denoise for all model and input types
for mt in "${model_types[@]}"; do
  for it in "${input_types[@]}"; do

    # Build the paths
    noised_dir_path="./noised"
    denoised_dir_path="./denoised/${mt}/${it}"
    results_csv_filepath="./results_${mt,,}_${it}.csv"

    # Loop through all .jpeg files in noised directory
    for image_filepath in "$noised_dir_path"/*.jpeg; do

      # Build the model file path
      if [ "$mt" == "WGAN" ]; then
        model_dir="wgans"
        model_prefix="wgan"
      else
        model_dir="autoencoders"
        model_prefix="ae"
      fi

      if [ "$it" == "full" ]; then
        model_suffix="f"
      else
        model_suffix="p"
      fi
      model_filepath="./../../../models/${model_dir}/${model_prefix}_fpn50_${model_suffix}.tflite"

      # Build the denoiser args
      denoiser_args="-i $image_filepath -r 224x224 -m $model_filepath -o $denoised_dir_path/$(basename "$image_filepath") -q 100"

      if [ "$it" == "patch" ]; then
        denoiser_args="$denoiser_args -p 56x56 -g 6"
      fi

      # Assign the command to a variable
      denoiser_command="./../../../tools/denoiser/build/denoiser $denoiser_args"

      # Print the command
      echo -e "\n\n$denoiser_command"

      # Execute the command using eval
      eval "$denoiser_command"
    done
  done
done
