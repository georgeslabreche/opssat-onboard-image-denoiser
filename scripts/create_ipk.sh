#!/usr/bin/env bash

# Creates the ipk file to install this project into the SEPP onboard the OPS-SAT spacecraft.

# The project directory path.
# Remove the scripts folder in case this bash script is being executed from the scripts folder
# instead of from the project root folder.
project_dir=$(pwd)
project_dir=${project_dir/scripts/""}

# The files that will be packaged into the ipk
bin_noiser=${project_dir}/tools/noiser/build/noiser
bin_denoiser=${project_dir}/tools/denoiser/build/denoiser
smartcam_config=${project_dir}/smartcam/config.ini

# Check that the required files exist
if [ ! -f "$bin_noiser" ] || [ ! -f "$bin_denoiser" ]; then
  echo "missing build files"
  echo "you must build the required tools first"
  echo "see instructions in tools/noiser/BUILD.md"
  exit 1
fi

# Check the 'noiser' executable binary
noiser_info=$(file $bin_noiser)
if [[ $noiser_info != *"ARM"* ]]; then
  echo "The noiser was not compiled for the spacecraft:"
  file $noiser_info
  exit 1
fi

# Check the 'denoiser' executable binary
denoiser_info=$(file $bin_denoiser)
if [[ $denoiser_info != *"ARM"* ]]; then
  echo "The denoiser was not compiled for the spacecraft:"
  file $denoiser_info
  exit 1
fi

# Extract the package name, version, and architecture from the control file.
PKG_NAME=$(sed -n -e '/^Package/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_VER=$(sed -n -e '/^Version/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_ARCH=$(sed -n -e '/^Architecture/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)

# The ipk filename will depend on the target environment: EM (flatsat) or FM (spacecraft).
# For the EM, just affix the filename with "em".
IPK_FILENAME=""

# Deployment directory paths.
deploy_dir=${project_dir}/deploy
deploy_home_dir=${deploy_dir}/home
deploy_exp_dir=${deploy_home_dir}/${PKG_NAME}
deploy_models_dir=${deploy_exp_dir}/models

# Clean and initialize the deploy folder.
rm -rf ${deploy_dir}
mkdir -p ${deploy_models_dir}

# The project can be packaged for the spacecraft (no bash command options) or for the EM (us the 'em' option).
if [ "$1" == "" ]; then
  IPK_FILENAME=${PKG_NAME}_${PKG_VER}_${PKG_ARCH}.ipk
  echo "Create ${IPK_FILENAME} for the spacecraft"

  # The test script.
  cp ${project_dir}/scripts/test_fm.sh ${deploy_exp_dir}

elif [ "$1" == "em" ]; then
  # If packaging for the EM then include some files needed for testing.
  IPK_FILENAME=${PKG_NAME}_${PKG_VER}_${PKG_ARCH}_em.ipk
  echo "Create ${IPK_FILENAME} for the EM"

  # The test script.
  cp ${project_dir}/scripts/test_em.sh ${deploy_exp_dir}
else
  # If not deploying for spacecraft nor the EM then an invalid parameter was given.
  echo "Error: invalid option"
  rm -rf ${deploy_dir}
  exit 1
fi

# The sample images.
cp ${project_dir}/scripts/sample.jpeg ${deploy_exp_dir}
cp ${project_dir}/scripts/sample.wb.jpeg ${deploy_exp_dir}

# A test script that denoises the pale blue dot.
cp ${project_dir}/scripts/a_pale_blue_dot.jpeg ${deploy_exp_dir}
cp ${project_dir}/scripts/test_bd.sh ${deploy_exp_dir}

# Copy files that that will be packaged into the ipk.
cp ${bin_noiser} ${deploy_exp_dir}
cp ${bin_denoiser} ${deploy_exp_dir}
cp ${smartcam_config} ${deploy_exp_dir}

# Create the toGround directory
mkdir ${deploy_exp_dir}/toGround

# Copy the pre-trained models.
# Keep the ipk under 10 MB for spacecraft uplink.
if [ "$1" == "em" ]; then
  cp -R ${project_dir}/models/autoencoders/*_fpn50_*.tflite ${deploy_models_dir}
  cp -R ${project_dir}/models/wgans/*_fpn50_*.tflite ${deploy_models_dir}
else
  cp -R ${project_dir}/models/autoencoders/ae_cfpn200_f.tflite ${deploy_models_dir}
  cp -R ${project_dir}/models/wgans/wgan_fpn50_p.tflite ${deploy_models_dir}
fi

# Create the label files.
# These are only required because the SmartCam expects them.
echo "noised" > ${deploy_exp_dir}/noiser.txt
echo "denoised" > ${deploy_exp_dir}/denoiser.txt

# Create the data tar file.
cd ${deploy_dir}
tar -czvf data.tar.gz home --numeric-owner --group=0 --owner=0

# Create the control tar file.
cd ${project_dir}/sepp_package/CONTROL
tar -czvf ${deploy_dir}/control.tar.gz control postinst postrm preinst prerm --numeric-owner --group=0 --owner=0
cp debian-binary ${deploy_dir}

# Create the ipk file.
cd ${deploy_dir}
ar rv ${IPK_FILENAME} control.tar.gz data.tar.gz debian-binary
echo "Created ${IPK_FILENAME}"

# Cleanup.
echo "Cleaning"

# Delete the tar files.
rm -f ${deploy_dir}/data.tar.gz
rm -f ${deploy_dir}/control.tar.gz
rm -f ${deploy_dir}/debian-binary

# Delete the home directory.
rm -rf ${deploy_home_dir}

# Done
echo "Qapla'"
