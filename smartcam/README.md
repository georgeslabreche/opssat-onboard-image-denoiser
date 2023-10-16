# SmartCam Integration
This experiment leverages the OPS-SAT SmartCam's [image classification/processing pipeline](https://github.com/georgeslabreche/opssat-smartcam#33-building-an-image-classification-pipeline) to apply the noiser and denoiser on images acquired by the spacecraft.

## Scenarios
The experiment will run a series of scenarios representing two types of noise at different levels of noise factor. Noiser and denoiser configurations in the SmartCam's [config.ini](config.ini) must be updated from one scenario to another.

### Fixed-Pattern Noising/Denoising

Noise Factor 50:
```ini
noiser.args   = -r 224x224 -t 1 -n 50 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_fpn50.tflite
```

Noise Factor 100:
```ini
noiser.args   = -r 224x224 -t 1 -n 100 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_fpn100.tflite
```

Noise Factor 150:
```ini
noiser.args   = -r 224x224 -t 1 -n 150 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_fpn150.tflite
```

Noise Factor 200:
```ini
noiser.args   = -r 224x224 -t 1 -n 200 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_fpn200.tflite
```

### Column Fixed-Pattern Noising/Denoising

Noise Factor 50:
```ini
noiser.args   = -r 224x224 -t 2 -n 50 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_cfpn50.tflite
```

Noise Factor 100:
```ini
noiser.args   = -r 224x224 -t 2 -n 100 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_cfpn100.tflite
```

Noise Factor 150:
```ini
noiser.args   = -r 224x224 -t 2 -n 150 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_cfpn150.tflite
```

Noise Factor 200:
```ini
noiser.args   = -r 224x224 -t 2 -n 200 -q 100
denoiser.args = -q 100 -m /home/exp253/models/denoiser_ae_cfpn200.tflite
```

## SpaceShell Examples
Updating the SmarCam's config.ini from fixed-pattern noise pattern factor 50 to factor 100:

```bash
sed -i 's/-n 50/-n 100/g' /home/exp1000/config.ini
sed -i 's/denoiser_ae_fpn50.tflite/denoiser_ae_fpn100.tflite/g' /home/exp1000/config.ini
cat /home/exp1000/config.ini
```

From fixed-pattern noise factor 200 to column fixed-pattern noise factor 50:

```bash
sed -i 's/-n 200/-n 50/g' /home/exp1000/config.ini
sed -i 's/denoiser_ae_fpn200.tflite/denoiser_ae_cfpn50.tflite/g' /home/exp1000/config.ini
cat /home/exp1000/config.ini
```