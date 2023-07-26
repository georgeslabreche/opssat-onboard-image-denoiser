# Compilter target parameter
# Otions are "dev" or "arm"
# Defaults to "dev" if no options are given
param (
  [string]$TARGET = "dev"
)

# Define Docker image and container names
$IMAGE_NAME = "noiser-builder"

# Build the Docker image
# Change the TARGET to arm to build for the spacecraft
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

# Run the Docker container
$CONTAINER_ID = $(docker run -d $IMAGE_NAME)

# Create the build directory if it doesn't exist already
if (!(Test-Path "build")) {
  New-Item -ItemType Directory -Force -Path "build"
}

# Copy the compiled executable
docker cp ${CONTAINER_ID}:/noiser/noiser build/

# Success!
Write-Host " Qapla'"