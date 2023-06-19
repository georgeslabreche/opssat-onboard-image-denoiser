# Noiser
- Add noise to an image, either a fixed noise pattern (FPN) or a Gaussian distribution.
- Uses the [stb library](https://github.com/georgeslabreche/stb) to read the image input and write the noisy image output.
- The input and output image format is in jpeg.

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
- **noise_type**: The noise type is a flag where 0 means Gaussian noise and 1 means fixed pattern noise.

## Samples
Sample outputs with doubling Gaussian noise factors (NF) from 25 to 400:

<table>
  <tr>
    <td><img src="sample.jpeg" width="200"/><br>Original</td>
    <td><img src="samples/sample.noisy.gaussian.025.jpeg" width="200"/><br>NF 25</td>
    <td><img src="samples/sample.noisy.gaussian.050.jpeg" width="200"/><br>NF 50</td>
  </tr>
  <tr>
    <td><img src="samples/sample.noisy.gaussian.100.jpeg" width="200"/><br>NF 100</td>
    <td><img src="samples/sample.noisy.gaussian.200.jpeg" width="200"/><br>NF 200</td>
    <td><img src="samples/sample.noisy.gaussian.400.jpeg" width="200"/><br>NF 400</td>
  </tr>
</table>

Sample outputs with doubling fixed noise pattern:


<table>
  <tr>
    <td><img src="sample.jpeg" width="200"/><br>Original</td>
    <td><img src="samples/sample.noisy.fpn.025.jpeg" width="200"/><br>NF 25</td>
    <td><img src="samples/sample.noisy.fpn.050.jpeg" width="200"/><br>NF 50</td>
  </tr>
  <tr>
    <td><img src="samples/sample.noisy.fpn.100.jpeg" width="200"/><br>NF 100</td>
    <td><img src="samples/sample.noisy.fpn.200.jpeg" width="200"/><br>NF 200</td>
    <td><img src="samples/sample.noisy.fpn.400.jpeg" width="200"/><br>NF 400</td>
  </tr>
</table>


**Why does the same noise factor value results in more noise for Gaussian distributions compared to fixed noise pattern?**

When generating _fixed noise pattern_ with a uniform distribution, all values within the specified range hold equal probability of being chosen. The resultant noise values added to each pixel will fall within the boundaries dictated by the noise factor, with all values being equally probable.

In contrast, generating _Gaussian noise_ produces values following a Gaussian or normal distribution. This distribution exhibits a bell curve shape, where values near the mean (set to 0 in the provided code) hold a higher likelihood of occurrence than values further away. In this case, the noise factor represents the standard deviation of the distribution. An increased noise factor enhances the spread of the distribution, leading to a broader range of noise values generated.

Consequently, a specific noise factor may result in more visible noise in the case of Gaussian noise generation as compared to fixed pattern noise. Although the range of noise values is technically equivalent for both, Gaussian noise generation is more prone to generating values that differ significantly from 0 (the mean), which results in more noticeable noise.