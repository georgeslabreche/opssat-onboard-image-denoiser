# GPU flag
GPU_ENABLED = True

# path where the trained model will be saved (and loaded)
MODEL_NAME = "model"
MODEL_DIR = "../../models/autoencoders"

# directory path for the imagery archive that will be downloaded
DIR_PATH_IMAGES_DOWNLOAD = "../../data"

# location of the image data
DIR_PATH_IMAGES_OPSSAT_TRAINING_SET = f"{DIR_PATH_IMAGES_DOWNLOAD}/opssat/earth/training-set"
DIR_PATH_IMAGES_OPSSAT_TEST_SET     = f"{DIR_PATH_IMAGES_DOWNLOAD}/opssat/earth/test-set"

# flag to display or not the test noisy images
DISPLAY_TEST_NOISE = False

# flag to display or not training & validation loss values
DISPLAY_TRAINING_AND_VALIDATION_LOSS_VALUES = False

# flag to display or not the validation noisy to denoise images
DISPLAY_VALIDATE_NOISE = False


# seveal autoencoders were experimented with:
# - 1 for DenoiseAutoencoderNaive
# - 2 for DenoiseAutoencoderSimple
# - 3 for DenoiseAutoencoderComplex
# - 4 for DenoiseAutoencoderSkipConnection                 --> finalist - 1st place (~1 MB)
# - 5 for DenoiseAutoencoderVGG16                          --> garbage
# - 6 for DenoiseAutoencoderSkipConnectionVGG16            --> finalist - 2nd place (~60 MB) ---> try without freezing the pre-trained model
#   7 for DenoiseAutoencoderMobileNetV2                    --> garbage
# - 8 for DenoiseAutoencoderSkipConnectionMobileNetV2
# see autoencoders.py for implementation
DENOISER_TYPE = 4

# some hyperparameters
EPOCHS = 100
BATCH_SIZE = 16

# set the proportion of data to use for training and testing
TRAIN_RATIO = 0.9

# greyscale the images to go easy on processing needs
DESIRED_CHANNELS = 3

# we can either add the noise to the loaded original files
# or load prenoised images from the filesystem (and then match them with the originals via filename)
LOAD_NOISY_IMAGES_FROM_FILE = True

# resize training data
TRAINING_DATA_RESIZE_ORIGINAL_FROM_FILE = True
TRAINING_DATA_RESIZE_NOISY_FROM_FILE    = False

# the noise type:
#  0 for Gaussian noise
#  1 for fixed-pattern noise (for CCD noise simulation)
#  2 for column fixed-pattern noise (for CMOS noise simulation)
NOISE_TYPE = 1

# the noise factor to determine how much noise to apply (e.g. 150)
NOISE_FACTOR = 50

# For transfer learning: MobileNetV2 and VGG16
# Also matches the dimension expected by the onboard CNN image classifier (SmartCam)
DESIRED_INPUT_HEIGHT = 224
DESIRED_INPUT_WIDTH = 224