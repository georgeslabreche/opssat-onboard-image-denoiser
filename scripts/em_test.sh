#!/bin/sh

# this script is used for EM validation
# it use used to run the noiser and denoiser binary executables on the EM
# only run this script on the EM to test that the executables work as expected

# the experiment directory
exp_dir=/home/exp253

# the directory for the test results
results_dir=${exp_dir}/results
rm -rf ${results_dir}
mkdir ${results_dir}

# the file paths of the executable binaries used to noise and denoise and image
bin_noiser=${exp_dir}/noiser
bin_denoiser=${exp_dir}/denoiser

# the file path of the sample image
img_filepath_original=${exp_dir}/sample.jpeg

# the noise type label
noise_type_label=""

for noise_type in 1 2; do
  for noise_factor in 50 100 150 200; do
    # ternary-like operation to set the noise type string
    [ ${noise_type} -eq 1 ] && noise_type_label="fnp" || noise_type_label="cfnp"

    # verbosity
    echo "" # new line
    echo "Testing ${noise_type_label} noise at factor ${noise_factor}"

    # the file path of the denoiser model
    model_filepath=${exp_dir}/models/denoiser_ae_${noise_type_label}${noise_factor}.tflite

    # the file path of the noised image
    img_filepath_noised=${results_dir}/sample.${noise_type_label}.${noise_factor}.noised.jpeg

    # the file path of the denoised image
    img_filepath_denoised=${results_dir}/sample.${noise_type_label}.${noise_factor}.denoised.jpeg

    # apply noise to the image
    ${bin_noiser} -i ${img_filepath_original} -r 224x224 -t ${noise_type} -n ${noise_factor} -o ${img_filepath_noised} -q 100

    # check the exit code of the noiser command
    if [ $? -ne 0 ]; then
      echo "Noiser command failed with exit code $?"
      exit 1
    else
      # wait before denoising the image that was just noised
      # the wait time only exist to have a nice gap in the CPU and memory usage plots
      sleep 1

      # new line
      echo ""
    fi

    # denoise the image using the available models
    ${bin_denoiser} -i ${img_filepath_noised} -m ${model_filepath} -o ${img_filepath_denoised} -q 100

    # check the exit code of the denoiser command
    if [ $? -ne 0 ]; then
      echo "Denoiser command failed with exit code $?"
      exit 1
    else
      # wait before processing the next image
      # the wait time only exist to have a nice gap in the CPU and memory usage plots
      sleep 2

      # new line
      echo ""
    fi
  done
done

# great success!
echo "" # new line
echo "Qapla'"
exit 0
