## Build
- Initialize and update the stb Git submodule: `git submodule init . && git submodule update .`
- Docker build and run: `./build.sh`
- Cross-compile for ARM: `./build.sh arm`

## Take a break
Building the denoiser will take a while because the Docker environment needs to compile CMake 3.16 and TensorFlow Lite from source. Seize the opportunity for an extended coffee break.

## Usage
```bash
cd build
./denoiser -?
```