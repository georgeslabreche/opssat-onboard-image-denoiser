from struct import unpack
from PIL import Image
import os
import constants

# find bad jpeg image files
# uses two approachs: 1) try to open the image with PIL and 2) scan the image and check markers
# some code taken from: https://github.com/tensorflow/tpu/issues/455

marker_mapping = {
  0xffd8: "Start of Image",
  0xffe0: "Application Default Header",
  0xffdb: "Quantization Table",
  0xffc0: "Start of Frame",
  0xffc4: "Define Huffman Table",
  0xffda: "Start of Scan",
  0xffd9: "End of Image"
}

class JPEG:
  def __init__(self, image_file):
    self.img_file = image_file

    with open(image_file, 'rb') as f:
      self.img_data = f.read()

  def decode(self):
    data = self.img_data
    while(True):
      marker, = unpack(">H", data[0:2])
      # print(marker_mapping.get(marker))
      if marker == 0xffd8:
        data = data[2:]
      elif marker == 0xffd9:
        return
      elif marker == 0xffda:
        data = data[-2:]
      else:
        lenchunk, = unpack(">H", data[2:4])
        data = data[2+lenchunk:]
      if len(data)==0:
        raise Exception()

  # find the corrupt jpeg image files in a given folder
  # this approach uses PIL
  def try_open(self):
    try:
      im = Image.open(self.img_file)
    except:
      raise Exception()


# go through the image files and check for invalid ones
for f in os.listdir(constants.DIR_PATH_IMAGES_EARTH):
  jpeg = JPEG(constants.DIR_PATH_IMAGES_EARTH + "/" + f)

  try:
    jpeg.try_open()
  except Exception as e:
    print(f + ': ' + str(e))

  try:
    jpeg.decode()
  except Exception as e:
    print(f + ': ' + str(e))