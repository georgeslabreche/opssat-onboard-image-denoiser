'''
the dataset ids for Landsat imagery:

  ref_landcovernet_af_v1: LandCoverNet Africa
  ref_landcovernet_as_v1: LandCoverNet Asia
  ref_landcovernet_au_v1: LandCoverNet Australia
  ref_landcovernet_eu_v1: LandCoverNet Europe
  ref_landcovernet_na_v1: LandCoverNet North America
  ref_landcovernet_sa_v1: LandCoverNet South America
'''
LANDCOVERNET_DATASET_ID_EU = 'ref_landcovernet_eu_v1'
LANDCOVERNET_DATASET_ID_NA = 'ref_landcovernet_na_v1'

# imagery source
LANDCOVERNET_IMAGERY_SOURCE_LANDSAT8 = 'landsat_8'

# path where the trained model will be saved (and loaded)
MODEL_NAME = "model"
MODEL_DIR = "./models"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

# directory path for the imagery archive that will be downloaded
DIR_PATH_IMAGES_DOWNLOAD = './data'

# location of the image data
DIR_PATH_IMAGES_OPSSAT_EARTH = "./data/opssat/earth"
DIR_PATH_IMAGES_OPSSAT_EDGE  = "./data/opssat/edge"
DIR_PATH_IMAGES_OPSSAT_VALIDATE = "./data/opssat/validate"
DIR_PATH_IMAGES_OPSSAT_WHITEBALANCED = "./data/opssat/wb"

# a pale blue dot
DIR_PATH_IMAGES_PALEBLUEDOT = "./data/pale_blue_dot"

# FIXME: use relative paths
DIR_INIT = "/home/cguzman/git/cguz/opssat-image-denoiser/"

# location of the image data
DIR_PATH_IMAGES_EARTH = DIR_INIT + "./data/OPSSAT/opssat-thumbnails/01_unprocessed/earth"
DIR_PATH_IMAGES_EDGE = DIR_INIT + "./data/OPSSAT/opssat-thumbnails/01_unprocessed/edge"
DIR_PATH_IMAGES_TEST = DIR_INIT + "./data/OPSSAT/opssat-thumbnails/01_unprocessed/test"
# parent directory path of the imagery
DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA = f'{DIR_PATH_IMAGES_DOWNLOAD}/{LANDCOVERNET_DATASET_ID_NA}_source_{LANDCOVERNET_IMAGERY_SOURCE_LANDSAT8}'

# target directory for the merged RGB bands imagery
DIR_PATH_IMAGERY_LANDSAT8_TRAIN = f'{DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA}_rgb/train'

# director for validation
DIR_PATH_IMAGERY_LANDSAT8_VALIDATE = f'{DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA}_rgb/validate'

# where to save the fetched images (in the training data folder)
# also where to read the training images for training
#DIR_PATH_IMAGERY_TRAIN = DIR_PATH_IMAGERY_LANDSAT8_TRAIN     # train using landsat training images
DIR_PATH_IMAGERY_TRAIN = DIR_PATH_IMAGES_OPSSAT_EARTH         # train using ops-sat training images

# directory containing imagery to validate the model
#DIR_PATH_IMAGERY_VALIDATE = DIR_PATH_IMAGERY_LANDSAT8_VALIDATE
DIR_PATH_IMAGERY_VALIDATE = DIR_PATH_IMAGES_OPSSAT_VALIDATE
#DIR_PATH_IMAGERY_VALIDATE = DIR_PATH_IMAGES_OPSSAT_WHITEBALANCED
#DIR_PATH_IMAGERY_VALIDATE = DIR_PATH_IMAGES_PALEBLUEDOT

# flag to display or not the test noisy images
DISPLAY_TEST_NOISE = False

# flag to display or not training & validation loss values
DISPLAY_TRAINING_AND_VALIDATION_LOSS_VALUES = False

# flag to display or not the validation noisy to denoise images
DISPLAY_VALIDATE_NOISE = False


# two types of denoisers autoencoders are implemented:
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

#DESIRED_INPUT_HEIGHT = 614
#DESIRED_INPUT_WIDTH = 583

#DESIRED_INPUT_HEIGHT = 300
#DESIRED_INPUT_WIDTH = 300

#DESIRED_INPUT_HEIGHT = 256
#DESIRED_INPUT_WIDTH = 256

# For transfer learning: MobileNetV2 and VGG16
DESIRED_INPUT_HEIGHT = 224
DESIRED_INPUT_WIDTH = 224

#DESIRED_INPUT_HEIGHT = 512
#DESIRED_INPUT_WIDTH = 512