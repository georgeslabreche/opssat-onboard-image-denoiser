# Install Instuctions
This program uses a Makefile.

## Build
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`.
- Compile with `make`.
- Can also compile for ARM architecture with `make TARGET=arm`.

## Usage
```bash
./noiser <image_filepath> <noise_factor> <noise_type>
```

- **image_filepath**: The filepath of the image input.
- **noise_type**: The noise factor to determine how much noise to apply (e.g. 150).
- **noise_type**: The noise type is a flag where 0 means Gaussian noise, 1 means FPN (for CCD noise simulation), and 2 means column FPN (for CMOS noise simulation).
