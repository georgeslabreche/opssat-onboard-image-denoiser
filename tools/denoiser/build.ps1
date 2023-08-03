# Compilter target parameter
# Default target compiler is the local gcc/g++ dev environment
# Options are "dev" or "arm"
param (
  [string]$TARGET = "dev"
)

# Define Docker image and container names
$IMAGE_NAME = "denoiser-builder"

# Build the Docker image
# Set the TARGET to arm to build for the spacecraft
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

# Run the Docker container
$CONTAINER_ID = $(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
if (!(Test-Path "build")) {
  New-Item -ItemType Directory -Force -Path "build"
}

# Copy the compiled executable
docker cp ${CONTAINER_ID}:/denoiser/build/denoiser build/

# Success!
Write-Host " Qapla' $CONTAINER_ID"