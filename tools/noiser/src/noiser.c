#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

double generate_random_normal() {
  // use the Box-Muller transform for generating normally distributed random numbers
  double u = (double) rand() / (RAND_MAX + 1.0);
  double v = (double) rand() / (RAND_MAX + 1.0);
  return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

void add_noise_to_image(unsigned char* image, int size, double noise_factor) {

  // for every pixel value
  for(int i = 0; i < size; ++i) {
    int noisy_value = (int) ((double) image[i] + noise_factor * generate_random_normal());

    // make sure the new value is within the valid range for an 8-bit color component
    noisy_value = noisy_value < 0 ? 0 : noisy_value;
    noisy_value = noisy_value > 255 ? 255 : noisy_value;

    // rewrite image bugger as noisy image buffer
    image[i] = (unsigned char) noisy_value;
  }
}

char* create_output_filename(char* input_filename) {
  char* last_dot = strrchr(input_filename, '.');
  if(last_dot == NULL) return NULL;

  // base filename length + ".noisy" + extension + null character
  size_t base_length = last_dot - input_filename;
  char* output_filename = malloc(base_length + 7 + strlen(last_dot) + 1);

  strncpy(output_filename, input_filename, base_length);
  output_filename[base_length] = '\0';

  strcat(output_filename, ".noisy");
  strcat(output_filename, last_dot);

  return output_filename;
}

int main(int argc, char *argv[]) {

  // check for expected number of arguments
  if(argc != 3) {
    printf("Usage: %s <image_filepath> <noise_factor>\n", argv[0]);
    return -1;
  }

  // read the image
  int width, height, channels;
  unsigned char* image = stbi_load(argv[1], &width, &height, &channels, 0);
  
  // error check
  if(image == NULL) {
    printf("Error in loading the image\n");
    return -1;
  }

  // add noise to the image
  int size = width * height * channels;
  double noise_factor = atof(argv[2]);
  add_noise_to_image(image, size, noise_factor);

  // create output filename
  char* output_filename = create_output_filename(argv[1]);
  if(output_filename == NULL) {
    printf("Error in creating the output filename\n");
    return -1;
  }

  // write the noisy image
  stbi_write_jpg(output_filename, width, height, channels, image, 100);

  // deallocate resources
  free(output_filename);
  stbi_image_free(image);

  return 0;
}