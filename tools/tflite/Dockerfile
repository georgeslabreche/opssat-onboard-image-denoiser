# Base image
FROM ubuntu:18.04

# Maintainer
LABEL maintainer="georges@tanagraspace.com"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends apt-utils 2>&1

# Install necessary build tools
RUN apt-get install -y git \
    python3 \
    python3-pip \
    curl \
    unzip \
    wget \
    build-essential \
    dos2unix

# Install NumPy
RUN pip3 install numpy

# Symbolic link for Bazel when it invokes Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Bazel 3.1.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
RUN chmod +x bazel-3.1.0-installer-linux-x86_64.sh
RUN ./bazel-3.1.0-installer-linux-x86_64.sh

# Copy the TensorFlow directory
COPY tensorflow /tensorflow

# Windows-style line endings (CRLF) instead of Unix-style line endings (LF)
# This is in case the host machine is a Windows environment
RUN find /tensorflow -type f -exec dos2unix {} \;

# Set the working directory
WORKDIR /tensorflow

# Configure the build
RUN yes "" | ./configure

# The argument for the compiler target
ARG TARGET=dev

# Build the TensorFlow Lite static library
RUN if [ "$TARGET" = "arm" ] ; then bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:tensorflowlite_c ; else bazel build -c opt //tensorflow/lite/c:tensorflowlite_c ; fi

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

# Uncomment this if we want to bash into the image with `docker exec -it <container_id_or_name> bash`
#CMD ["tail", "-f", "/dev/null"]