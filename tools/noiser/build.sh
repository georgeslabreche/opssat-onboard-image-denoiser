#!/bin/bash

# Default target compiler to "dev" if not provided as a command line argument
# Options are "dev" or "arm"
TARGET=${1:-dev}

# Define Docker image and container names
IMAGE_NAME="noiser-builder"

# Build the Docker image
# Change the TARGET to arm to build for the spacecraft
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

# Run the Docker container
CONTAINER_ID=$(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
mkdir -p build

# Copy the compiled executable
docker cp $CONTAINER_ID:/noiser/noiser build/

# Success!
echo " Qapla'"
