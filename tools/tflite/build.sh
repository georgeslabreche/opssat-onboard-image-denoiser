#!/bin/bash

# Default target compiler to "dev" if not provided as a command line argument
# Options are "dev" or "arm"
TARGET=${1:-dev}

# Define Docker image and container names
IMAGE_NAME="tflite-builder"

# Build the Docker image
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

# Run the Docker container
CONTAINER_ID=$(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
mkdir -p build

# Copy the TensorFlow Lite C library from the Docker container to host system
docker cp $CONTAINER_ID:/tensorflow/bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so build/

# Success!
echo " Qapla'"
