## What was done
1. Grabbed the images denoised by WGANs onboard the spacecraft.
2. Denoised them on the FlatSat with Autoencoder.
3. Calculated the PSNR, SSIM, and MSE metrics when denoised with Autoencoder.

## But why?
So that we cam compare the denoising performance of WGANs vs Autoencoder on a common image dataset that was acquired by the spacecraft. It doesn't matter if WGANs denoising was done onboard the spacecraft and Autoencoder denoising was done in the FlatSat; it's the same hardware and software.