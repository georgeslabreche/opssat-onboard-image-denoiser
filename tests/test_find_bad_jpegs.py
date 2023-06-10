from image_denoiser import find_bad_jpegs
from image_denoiser import constants
import os
    
def test_find_bad_jpegs():

    # go through the image files and check for invalid ones
    for f in os.listdir(constants.DIR_PATH_IMAGES_EARTH):
        jpeg = find_bad_jpegs.JPEG(constants.DIR_PATH_IMAGES_EARTH + "/" + f)

        try:
            jpeg.try_open()
        except Exception as e:
            assert False

        try:
            jpeg.decode()
        except Exception as e:
            assert False
        
    assert True