TARGET = dev

# Compiler options.
CC_DEV = g++
CC_ARM = /usr/bin/arm-linux-gnueabihf-g++

# Header includes.
INCLUDEPATH = -Istb

# Flags.
CFLAGS = -Wall -static -Os -std=c++14

# Library flags
LIBFLAGS = -lm

# Source directory and files.
SOURCEDIR = src
HEADERS := $(wildcard $(SOURCEDIR)/*.hpp)
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)

# Target output.
BUILDTARGET = noiser

# Target compiler environment.
ifeq ($(TARGET),arm)
	CC = $(CC_ARM)
else
	CC = $(CC_DEV)
endif

all:
	$(CC) $(CFLAGS) $(INCLUDEPATH) $(HEADERS) $(SOURCES) -o $(BUILDTARGET) $(LIBFLAGS)

clean:
	rm -f $(BUILDTARGET)