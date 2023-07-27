# Compilter target parameter
# Otions are "dev" or "arm"
# Defaults to "dev" if no options are given
param (
  [string]$TARGET = "dev"
)

# Define Docker image and container names
$IMAGE_NAME = "denoiser-builder"

# Build the Docker image
# Change the TARGET to arm to build for the spacecraft
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME -f Dockerfile.denoiser .

# Run the Docker container
$CONTAINER_ID = $(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
if (!(Test-Path "denoiser/build")) {
  New-Item -ItemType Directory -Force -Path "denoiser/build"
}

# Copy the compiled executable and the TensorFlow Lite C library
docker cp ${CONTAINER_ID}:/denoiser/denoiser denoiser/build/
docker cp ${CONTAINER_ID}:/denoiser/libtensorflowlite_c.so denoiser/build/

# Success!
Write-Host " Qapla'"