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
MODEL_PATH = "./models/landsat8v1"

# directory path for the imagery archive that will be downloaded
DIR_PATH_IMAGES_DOWNLOAD = './data'

# location of the image data
DIR_PATH_IMAGES_OPSSAT_EARTH = "./data/opssat/earth"
DIR_PATH_IMAGES_OPSSAT_EDGE  = "./data/opssat/edge"
DIR_PATH_IMAGES_OPSSAT_VALIDATE  = "./data/opssat/validate"

# parent directory path of the imagery
DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA = f'{DIR_PATH_IMAGES_DOWNLOAD}/{LANDCOVERNET_DATASET_ID_NA}_source_{LANDCOVERNET_IMAGERY_SOURCE_LANDSAT8}'

# target directory for the merged RGB bands imagery
DIR_PATH_IMAGERY_LANDSAT8_TRAIN = f'{DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA}_rgb/train'

# director for validation
DIR_PATH_IMAGERY_LANDSAT8_VALIDATE = f'{DIR_PATH_IMAGERY_INPUT_LANDSAT8_NA}_rgb/validate'

# where to save the fetched imaged (in the training data folder)
DIR_PATH_IMAGERY_TRAIN = DIR_PATH_IMAGERY_LANDSAT8_TRAIN
DIR_PATH_IMAGERY_VALIDATE = DIR_PATH_IMAGERY_LANDSAT8_VALIDATE

# flag to display or not the test noisy images
DISPLAY_TEST_NOISE = True

# two types of denoisers are implemented:
# - 1 for NaiveDenoiser
# - 2 for SimpleDenoiser
# see autoencoders.py for implementation
DENOISER_TYPE = 1

# some hyperparameters
EPOCHS = 10
BATCH_SIZE = 32

# set the proportion of data to use for training and testing
TRAIN_RATIO = 0.9

# greyscale the images to go easy on processing needs
DESIRED_CHANNELS = 3

# noise factor image
NOISE_FACTOR = 0.2

# resize the image
RESIZE_IMAGE = False

#DESIRED_INPUT_HEIGHT = 614
#DESIRED_INPUT_WIDTH = 583

#DESIRED_INPUT_HEIGHT = 300
#DESIRED_INPUT_WIDTH = 300

DESIRED_INPUT_HEIGHT = 256
DESIRED_INPUT_WIDTH = 256

#DESIRED_INPUT_HEIGHT = 512
#DESIRED_INPUT_WIDTH = 512