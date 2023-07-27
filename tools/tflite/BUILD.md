## Build
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`
- Docker build and run: `./build.sh`
- Cross-compile for ARM: `./build.sh arm`

Note that the TensorFlow submodule uses the last revision prior to the SmartCam fork [(from Oct 28, 2020)](https://github.com/georgeslabreche/tensorflow-opssat-smartcam/commit/669993ebe8534eac877eec61225925cff737eac2). That revision corresponds to a few commits ahead of [Rekease v2.1.2 (Sep 24, 2020)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.2) and is confirmed to work on the onboard the spacecraft.
