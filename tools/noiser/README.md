# Noiser
- Add noise to an image (applies a normal distribution).
- Uses the [stb library](https://github.com/georgeslabreche/stb) to read the image input and write the noisy image output.
- The input and output image format is in jpeg.

## Build
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`.
- Compile with `make`.
- Can also compile for ARM architecture with `make TARGET=arm`.

## Usage
```bash
./noiser <image_filepath> <noise_factor>
```

## Samples
Sample outputs with doubling noise factors (NF) from 25 to 400:

<table>
  <tr>
    <td><img src="sample.jpeg" width="200"/><br>Original</td>
    <td><img src="samples/sample.noisy.025.jpeg" width="200"/><br>NF 25</td>
    <td><img src="samples/sample.noisy.050.jpeg" width="200"/><br>NF 50</td>
  </tr>
  <tr>
    <td><img src="samples/sample.noisy.100.jpeg" width="200"/><br>NF 100</td>
    <td><img src="samples/sample.noisy.200.jpeg" width="200"/><br>NF 200</td>
    <td><img src="samples/sample.noisy.400.jpeg" width="200"/><br>NF 400</td>
  </tr>
</table>