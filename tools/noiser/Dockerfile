# Define the build stage
FROM debian:buster AS build

# Maintainer
LABEL maintainer="georges@tanagraspace.com"

# Install the required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    make

# Copy the denoiser source code
COPY . /noiser

# Set the working directory
WORKDIR /noiser

# The argument for the compiler target
ARG TARGET=dev

# Compile the denoiser
RUN make clean && make TARGET=$TARGET

# Make the compiled binary executable
RUN chmod +x noiser