#!/bin/bash

# Default target compiler to "dev" if not provided as a command line argument
# Options are "dev" or "arm"
TARGET=${1:-dev}

# Define Docker image and container names
IMAGE_NAME="denoiser-builder"

# Build the Docker image
# Change the TARGET to arm to build for the spacecraft
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME -f Dockerfile.denoiser .

# Run the Docker container
CONTAINER_ID=$(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
mkdir -p denoiser/build

# Copy the compiled executable and the TensorFlow Lite C library
docker cp $CONTAINER_ID:/denoiser/denoiser denoiser/build/
docker cp $CONTAINER_ID:/denoiser/libtensorflowlite_c.so denoiser/build/

# Success!
echo " Qapla'"
