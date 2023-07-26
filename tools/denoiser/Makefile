TARGET = dev

# Compiler options.
CC_DEV = g++
CC_ARM = /usr/bin/arm-linux-gnueabihf-g++

# Target compiler environment.
ifeq ($(TARGET),arm)
	CC = $(CC_ARM)
else
	CC = $(CC_DEV)
endif

# Header includes.
INCLUDEPATH = -Istb -I/tensorflow

# Flags.
CFLAGS = -Wall -O3 -std=c++14 -Wl,-rpath='$$ORIGIN'

# Library flags.
LIBFLAGS = -L. -ltensorflowlite_c

# Source directory and files.
SOURCEDIR = src
HEADERS := $(wildcard $(SOURCEDIR)/*.hpp)
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)

# Target output.
BUILDTARGET = denoiser

all:
	$(CC) $(CFLAGS) $(INCLUDEPATH) $(HEADERS) $(SOURCES) -o $(BUILDTARGET) $(LIBFLAGS)

clean:
	rm -f $(BUILDTARGET)
