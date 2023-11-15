import numpy as np
import cv2

PATH="./test-images/"

def compare_images(original_image, denoised_image):
  # Normalize the pixel values
  original_image = original_image.astype('float32') / 255.0
  denoised_image = denoised_image.astype('float32') / 255.0

  # Calculate the difference between the normalized pixel values
  difference = original_image - denoised_image

  # Calculate the mean squared error (MSE) of the difference
  mse = np.mean((difference * difference))

  # If the MSE is greater than a certain threshold, then there is a problem with the model
  if mse > 0.01:
    print('There is a problem with the model. The MSE is', mse)
  else:
    print('The model is working correctly. The MSE is', mse)

# Load the original image
original_image = cv2.imread(PATH+'original.jpeg')

# Load the denoised image
denoised_image = cv2.imread(PATH+'denoised.jpeg')

# Compare the images
compare_images(original_image, denoised_image)