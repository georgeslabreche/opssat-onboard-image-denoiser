#!/bin/sh

# the experiment directory
exp_dir=/home/exp253

# the noise parameters
noise_type="$1"
noise_factor="$2"

# ternary-like operation to set the noise type string
[ ${noise_type} -eq 1 ] && noise_type_label="fpn" || noise_type_label="cfpn"

# the model parameters
model_type="$3"
model_input_full_or_patch="$4"
model_filepath=${exp_dir}/models/${model_type}_${noise_type_label}${noise_factor}_${model_input_full_or_patch}.tflite

# the directory for the test results
toGround_dir=${exp_dir}/toGround

# the file paths of the executable binaries used to noise and denoise and image
bin_noiser=${exp_dir}/noiser
bin_denoiser=${exp_dir}/denoiser

# the file path of the sample image
img_filepath_original=${exp_dir}/sample.jpeg

# the file path of the noised image
img_filepath_noised=${toGround_dir}/sample.${noise_type_label}${noise_factor}.noised.jpeg

# the file path of the denoised image
img_filepath_denoised=${toGround_dir}/sample.${noise_type_label}${noise_factor}.denoised.jpeg

# apply noise to sample image
cmd_noiser="${bin_noiser} -i ${img_filepath_original} -r 224x224 -t ${noise_type} -n ${noise_factor} -o ${img_filepath_noised} -q 100"
echo "${cmd_noiser}"
eval "time ${cmd_noiser}"

# catch the exit code
exit_code=$?

# check the exit code of the noiser command
if [ $exit_code -ne 0 ]; then
  echo "Noiser command failed with exit code $exit_code"
  exit 1
fi

# denoise the image using the available models
cmd_denoiser="${bin_denoiser} -i ${img_filepath_noised} -m ${model_filepath} -o ${img_filepath_denoised} -q 100"
if [ "$model_type" = "wgan" ]; then
  cmd_denoiser="${cmd_denoiser} -p 56x56 -g 6"
fi
echo "${cmd_denoiser}"
eval "time ${cmd_denoiser}"

# catch the exit code
exit_code=$?

# check the exit code of the denoiser command
if [ $exit_code -ne 0 ]; then
  echo "Denoiser command failed with exit code $exit_code"
  exit 1
fi

# list the test artifacts
ls -larth ${toGround_dir}

# the test was a great success!
echo "Qapla'"
exit 0