import os
import subprocess

# this script triggers training for autoencoder denoising models
# a model is trained for each noise type and noise factor pair
# make sure that:
#  1. all the training images exists (original and noised ones)
#  2. the python virtual environment is active

# change directory
os.chdir("..")

# the python script that triggers the training
training_script = "src/image_denoiser/03_train_autoencoder.py"

# the noise type label
noise_type_label = ""

# apply different noise types
for noise_type in [1, 2]:

  # ternary-like operation to set the noise type string
  if noise_type == 1:
    noise_type_label = "fnp"
  else:
    noise_type_label = "cfnp"

  # apply different noise factors
  for noise_factor in [50, 100, 150, 200]:
    # some verbosity
    print(f"training {noise_type_label} denoiser autoencoder for noise factor {noise_factor}...")
    
    # execute the python training script for the current noise type and noise factor
    subprocess.run(['python', training_script, '-m', 'denoiser_ae', '-t', str(noise_type), '-f', str(noise_factor)])

# change directory back into the scripts folder
os.chdir("scripts")

# great success
print("Qapla'")
