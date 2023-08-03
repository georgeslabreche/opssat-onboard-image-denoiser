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

## Hax
The TensorFlow Lite C++ library is built with `XNNPACK` enabled. This results in a one time `INFO` log message when running the denoiser. The logging is harcoded in `xnnpack_delegate.cc`. A [fix to disable it](https://github.com/tensorflow/tensorflow/issues/58050#issuecomment-1623290091) is only available in a more recent version of the library which is incompatible with the spacecraft's eLinux environment. The workaround is to simply edit `xnnpack_delegate.cc` and commenting out the following prior to building the project: 

```cpp
TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                     "Created TensorFlow Lite XNNPACK delegate for CPU.");
```

Createinga fork of TensorFlow just for this fix seemed overkill.