#!/bin/sh

# this script is used for EM validation
# it use used to run the noiser and denoiser binary executables on the EM
# only run this script on the EM to test that the executables work as expected

# check that an argument for the model type is given
if [ "$#" -ne 1 ]; then
  echo "Error: Exactly one argument is require: 'wgans' or 'autoencoders'."
  exit 1
fi

# check that the model type value is correct
MODEL_TYPE="$1"
if [ "$MODEL_TYPE" != "wgans" ] && [ "$MODEL_TYPE" != "autoencoders" ]; then
  echo "Error: Argument must be either 'wgans' or 'autoencoders'."
  exit 1
fi

# the experiment directory
exp_dir=/home/exp253

# the directory for the test results
results_parent_dir=${exp_dir}/results
results_dir=${results_parent_dir}/${MODEL_TYPE}
rm -rf ${results_dir}
mkdir -p ${results_dir}

# the file paths of the executable binaries used to noise and denoise and image
bin_noiser=${exp_dir}/noiser
bin_denoiser=${exp_dir}/denoiser

# the file path of the sample image
img_filepath_original=${exp_dir}/sample.jpeg

# the noise type label
noise_type_label=""

# verbosity
echo "Testing ${MODEL_TYPE}:"

for noise_type in 1 2; do
  for noise_factor in 50; do
    # ternary-like operation to set the noise type string
    [ ${noise_type} -eq 1 ] && noise_type_label="fnp" || noise_type_label="cfnp"

    # verbosity
    echo "" # new line
    echo "${noise_type_label} at factor ${noise_factor}"

    # the file path of the denoiser model
    if [ "$MODEL_TYPE" = "autoencoders" ]; then
      model_filepath=${exp_dir}/models/ae_${noise_type_label}${noise_factor}_f.tflite
    elif [ "$MODEL_TYPE" = "wgans" ]; then
      model_filepath=${exp_dir}/models/wgan_${noise_type_label}${noise_factor}_p.tflite
    fi

    # the file path of the noised image
    img_filepath_noised=${results_dir}/sample.${noise_type_label}.${noise_factor}.noised.jpeg

    # the file path of the denoised image
    img_filepath_denoised=${results_dir}/sample.${noise_type_label}.${noise_factor}.denoised.jpeg

    # apply noise to the image
    cmd_noiser="${bin_noiser} -i ${img_filepath_original} -r 224x224 -t ${noise_type} -n ${noise_factor} -o ${img_filepath_noised} -q 100"
    echo "${cmd_noiser}"
    eval "time ${cmd_noiser}"

    # cache the exit code
    exit_code=$?

    # check the exit code of the noiser command
    if [ $exit_code -ne 0 ]; then
      echo "Noiser command failed with exit code $exit_code"
      exit 1
    else
      # wait before denoising the image that was just noised
      # the wait time only exists to have a nice gap in the CPU and memory usage plots
      sleep 1

      # new line
      echo ""
    fi

    # denoise the image using the available models
    cmd_denoiser="${bin_denoiser} -i ${img_filepath_noised} -m ${model_filepath} -o ${img_filepath_denoised} -q 100"
    if [ "$MODEL_TYPE" = "wgans" ]; then
      cmd_denoiser="${cmd_denoiser} -p 56x56 -g 6"
    fi
    echo "${cmd_denoiser}"
    eval "time ${cmd_denoiser}"

    # cache the exit code immediately
    exit_code=$?

    # check the exit code of the denoiser command
    if [ $exit_code -ne 0 ]; then
      echo "Denoiser command failed with exit code $exit_code"
      exit 1
    else
      # wait before processing the next image
      # the wait time only exists to have a nice gap in the CPU and memory usage plots
      sleep 2

      # new line
      echo ""
    fi
  done
done

# tar the results
tar -cvzf exp253_${MODEL_TYPE}_results.tar.gz ${results_parent_dir}/

# great success!
echo "" # new line
echo "Qapla'"
exit 0
