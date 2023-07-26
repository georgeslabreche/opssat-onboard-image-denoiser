## Build
- Prerequisite: build the TensorFlow Lite C shared library, see [here](../tflite/BUILD.md)
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`
- Change directory to the parent directory `cd ..`
- Docker build and run: `./build-denoiser.sh`
- Cross-compile for ARM: `./build-denoiser.sh arm`

The build needs to be triggered from the parent directory because we need to resolved a dependency with the TensorFlow Lice C shared library in the [tools/tflite](../tflite/) folder.

## Usage
```bash
cd build
./denoiser -?
```