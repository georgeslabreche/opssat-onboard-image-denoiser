# Compilter target parameter
# Otions are "dev" or "arm"
# Defaults to "dev" if no options are given
param (
  [string]$TARGET = "dev"
)

# Define Docker image and container names
$IMAGE_NAME = "tflite-builder"

docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

# Run the Docker container
$CONTAINER_ID = $(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
if (!(Test-Path "build")) {
  New-Item -ItemType Directory -Force -Path "build"
}

# Copy the TensorFlow Lite C library from the Docker container to host system
docker cp ${CONTAINER_ID}:/tensorflow/bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so build/

# Success!
Write-Output " Qapla'"
