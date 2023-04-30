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
Mostly following and adapting TensorFlow's [Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder). There's another example by Keras, [here](https://keras.io/examples/vision/autoencoder/).

## Execution
1. Use `find_bad_jpegs.py` to identify corrupt images that will break the training (get rid of those images, if they exist).
2. Edit `constants.py` with the desired training parameters.
3. Train the model with `train_denoiser.py`.
4. Test the model on some images with `denoise.py`.
