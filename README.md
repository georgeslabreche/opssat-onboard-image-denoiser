# OPS-SAT Image Denoiser
Denoising OPS-SAT images with TensorFlow Autoencoders.

## Geting started
Install the virtual environment:
```bash
pip install virtualenv
virtualenv venv
```

Activate the environment in Linux:
```bash
source venv/bin/activate
```

In Windows:
```
.\venv\Scripts\activate
```

Install the application's Python package dependencies:
```
pip install -r requirements.txt
```

Run the application
```bash
python denoiser.py
```

Deactivate the environment:
```bash
deactivate
```

## Implementation
Mostly following and adapting TensorFlow's [Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder).
