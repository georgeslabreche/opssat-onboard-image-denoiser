#!/bin/sh

# this script denoises the pale blue dot

# the experiment directory
exp_dir=/home/exp253

# the noise parameters
noise_type_label="fpn"
noise_factor=50

# the model parameters
model_type="wgan"
model_input_full_or_patch="p"
model_filepath=${exp_dir}/models/${model_type}_${noise_type_label}${noise_factor}_${model_input_full_or_patch}.tflite

# the directory for the test results
toGround_dir=${exp_dir}/toGround

# the file paths of denoiser executable
bin_denoiser=${exp_dir}/denoiser

# the file path of the blue dot image
img_filepath_noised=${exp_dir}/a_pale_blue_dot.jpeg

# the file path of the denoised blue dot image
img_filepath_denoised=${toGround_dir}/a_pale_blue_dot.${model_type}.${noise_type_label}${noise_factor}.denoised.jpeg

# denoise the image
cmd_denoiser="${bin_denoiser} -i ${img_filepath_noised} -m ${model_filepath} -o ${img_filepath_denoised} -q 100 -p 56x56 -g 6"
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