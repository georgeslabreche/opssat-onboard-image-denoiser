# path where the trained model will be saved (and loaded)
MODEL_PATH = "models/v4/denoiser"

# location of the image data
DIR_PATH_IMAGES_EARTH = "D:/Pictures/OPSSAT/opssat-thumbnails/01_unprocessed/earth"
DIR_PATH_IMAGES_EDGE = "D:/Pictures/OPSSAT/opssat-thumbnails/01_unprocessed/edge"
DIR_PATH_IMAGES_TEST = "D:/Pictures/OPSSAT/opssat-thumbnails/01_unprocessed/test"

# flag to display or not the test noisy images
DISPLAY_TEST_NOISE = True

# some hyperparameters
EPOCHS = 10
BATCH_SIZE = 32

# set the proportion of data to use for training and testing
TRAIN_RATIO = 0.8

# greyscale the images to go easy on processing needs
DESIRED_CHANNELS = 1

# noise factor image
NOISE_FACTOR = 0.03

# resize the image
#DESIRED_INPUT_HEIGHT = 614
#DESIRED_INPUT_WIDTH = 583

#DESIRED_INPUT_HEIGHT = 300
#DESIRED_INPUT_WIDTH = 300

DESIRED_INPUT_HEIGHT = 256
DESIRED_INPUT_WIDTH = 256

#DESIRED_INPUT_HEIGHT = 512
#DESIRED_INPUT_WIDTH = 512