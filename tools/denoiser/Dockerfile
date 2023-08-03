# Use a suitable base image
FROM debian:buster

# Maintainer
LABEL maintainer="georges@tanagraspace.com"

# Install the required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    git \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    wget

# Install the version of CMake expected by TensorFlow
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0.tar.gz && \
    tar -xvzf cmake-3.16.0.tar.gz && \
    cd cmake-3.16.0 && \
    ./bootstrap && \
    make && \
    make install && \
    export PATH=/usr/local/bin:$PATH

# Copy the denoiser source code and files
COPY src /denoiser/src
COPY stb /denoiser/stb
COPY tensorflow /denoiser/tensorflow
COPY CMakeLists.txt /denoiser

# Set the working directory
WORKDIR /denoiser

# Create build directory
RUN mkdir build

# The argument for the compiler target
ARG TARGET=dev

# Re-configure with the target and build the project
RUN cd build && cmake -DTARGET=$TARGET .. && cmake --build .

# Make executable the compiled binary
RUN chmod +x build/denoiser
