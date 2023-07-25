# Install Instructions
Instructions on how to install the application to train and apply the denoiser autoencoder model. Training images are not provided.

## Installation
Install and create the virtual environment:
```bash
sudo apt install python3-venv
pip install virtualenv
virtualenv venv
```

Activate the virtual environment in Linux:
```bash
source venv/bin/activate
```

In Windows:
```
.\venv\Scripts\activate
```

Make sure you are in the last version:
```bash
python3 -m pip install --upgrade build
```

Install Pytest:
```bash
pip install pytest
```

Make sure that you pass all the test:
```bash
pytest
```

Compile the last version:
```bash
python3 -m build
```

Install using pip:
```bash
pip install dist/image_denoiser_cguz-0.0.1-py3-none-any.whl
```

Deactivate the virtual environment:
```bash
deactivate
```

In Windows with Conda:
```powershell
conda create -n denoiser
conda activate denoiser
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc=11.3
pip install -r requirements.txt
```

## Configure MLHub
- Training data are fetched from [mlhub.earth](https://mlhub.earth/).
- Create an account and generate an API Key.
- Configure the API key in your development environment:

```bash
$ mlhub configure
API Key: Enter your API key here...
Wrote profile to /Users/youruser/.mlhub/profile
```

## Execution
Activate the virtual environment and afger editing `constants.py` with the desired training parameters:

```bash
python src/image_denoiser/01_fetch_imagery.py
python src/image_denoiser/02_merge_bands.py
python src/image_denoiser/03_train_autoencoder.py
python src/image_denoiser/04_denoise.py
```
