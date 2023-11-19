# Memory Overview
These test runs were conducted to demonstrate the influence of image input sizes on memory allocation during denoising. Memory allocation is much greater with single full image inputs than with multiple patched image inputs. The memory allocation is greater when denoising full images with WGANs than with Autoencoder whereas it remains approximately the same for both with patched image inputs. The patched approach is thus a more scalable solution that can be applied to larger images without the risk of running out of memory.

## Denoising Order
The folder names are confusing so here's a list of the order in which the models were tested for these runs.

### Run 1
1. Full image Autoencoder.
2. Patched image Autoencoder
3. Full image WGANs.
4. Patched image WGANs.

### Run 2
1. Patched image Autoencoder
2. Full image Autoencoder.
3. Patched image WGANs.
4. Full image WGANs.

## Execution Times
Execution times in the `exec_times.csv` files were measured using the `time` command.
