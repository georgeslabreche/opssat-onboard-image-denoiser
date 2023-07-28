#!/usr/bin/env bash

# Creates the ipk file to install this project into the SEPP onboard the OPS-SAT spacecraft.

# The project directory path.
# Remove the scripts folder in case this bash script is being executed from the scripts folder
# instead of from the project root folder.
project_dir=$(pwd)
project_dir=${project_dir/scripts/""}

# The experiment id.
exp_id=253

# The files that will be packaged into the ipk
bin_noiser=${project_dir}/tools/noiser/build/noiser
bin_denoiser=${project_dir}/tools/denoiser/build/denoiser
lib_tensorflowlite_c=${project_dir}/tools/tflite/build/libtensorflowlite_c.so

# Check that the required files exist
if [ ! -f "$bin_noiser" ] || [ ! -f "$bin_denoiser" ] || [ ! -f "$lib_tensorflowlite_c" ]; then
  echo "missing build files"
  echo "you must build the required tools first"
  echo "see instructions in tools/noiser/BUILD.md"
  exit 1
fi

# Extract the package name, version, and architecture from the control file.
PKG_NAME=$(sed -n -e '/^Package/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_VER=$(sed -n -e '/^Version/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_ARCH=$(sed -n -e '/^Architecture/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)

# Build the ipk filename.
IPK_FILENAME=${PKG_NAME}_${PKG_VER}_${PKG_ARCH}.ipk

# Deployment directory paths.
deploy_dir=${project_dir}/deploy
deploy_home_dir=${deploy_dir}/home
deploy_exp_dir=${deploy_home_dir}/exp${exp_id}
deploy_models_dir=${deploy_exp_dir}/models

# Clean and initialize the deploy folder.
rm -rf ${deploy_dir}
mkdir -p ${deploy_models_dir}

# The project can be packaged for the spacecraft (no bash command options) or for the EM (us the 'em' option).
if [ "$1" == "" ]; then
  echo "Create ${IPK_FILENAME} for the spacecraft"
elif [ "$1" == "em" ]; then
  # If packaging for the EM then include some files needed for testing
  echo "Create ${IPK_FILENAME} for the EM"

  # The sample image
  cp ${project_dir}/scripts/sample.jpeg ${deploy_exp_dir}

  # The test script
  cp ${project_dir}/scripts/em_test.sh ${deploy_exp_dir}
else
  # If not deploying for spacecraft nor the EM then an invalid parameter was given.
  echo "Error: invalid option"
  rm -rf ${deploy_dir}
  exit 1
fi

# Copy files that that will be packaged into the ipk
cp ${bin_noiser} ${deploy_exp_dir}
cp ${bin_denoiser} ${deploy_exp_dir}
cp ${lib_tensorflowlite_c} ${deploy_exp_dir}
cp -R ${project_dir}/models/*.tflite ${deploy_models_dir}

# Create the data tar file.
cd ${deploy_dir}
tar -czvf data.tar.gz home --numeric-owner --group=0 --owner=0

# Create the control tar file.
cd ${project_dir}/sepp_package/CONTROL && 
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
