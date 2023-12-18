import os
import requests
from tensorflow.keras.models import model_from_json
import tensorflow as tf


# Target input shape
input_shape = (224, 224, 1)

# Model file names
model_file_h5 = "model.h5"
model_file_json = "model.json"

# URL of the files to be downloaded
model_file_url_h5 = f"https://github.com/cszn/DnCNN/raw/master/TrainingCodes/dncnn_keras/models/DnCNN_sigma25/{model_file_h5}"
model_file_url_json = f"https://raw.githubusercontent.com/cszn/DnCNN/master/TrainingCodes/dncnn_keras/models/DnCNN_sigma25/{model_file_json}"

# Flag to indicate if we optimize the model coversion to TFLite
#       Optimized: results in smaller TFLite file size but slower to run the interpreter
#   Not Optimized: results in larger TFLite file size but faster to run the interpreter
optimize = False


def get_file(url, filename):
  # Sending a GET request to the URL
  response = requests.get(url)

  # Checking if the request was successful
  if response.status_code == 200:
    # Writing the content of the response to a file
    with open(filename, "wb") as file:
      file.write(response.content)
  else:
    print("Download failed. Status code: " + str(response.status_code))


# Get files
get_file(model_file_url_h5, model_file_h5)
get_file(model_file_url_json, model_file_json)

# Load JSON model architecture
json_file = open(model_file_json, 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load model weights
loaded_model.load_weights(model_file_h5)

# Define the input layer
input_layer = tf.keras.layers.Input(shape=input_shape)

# Replace the input layer of the loaded model with the new input shape
loaded_model = tf.keras.Model(inputs=input_layer, outputs=loaded_model(input_layer))

# Convert the model to TFLite format with the specific input shape
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# Some optional conversion parameters
if optimize is True:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Convert!
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("dncnn.tflite", "wb") as f:
  f.write(tflite_model)


# Delete the h5 and json model files
os.remove(model_file_h5)
os.remove(model_file_json)

# Great success!
print("Qapla'")