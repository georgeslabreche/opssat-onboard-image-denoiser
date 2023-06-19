#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// function to generate Gaussian noise
double generate_gaussian_noise(double mean, double std_dev) {

  // uniformly distributed random number between 0 and 1
  double u1 = (double) rand() / RAND_MAX;
  double u2 = (double) rand() / RAND_MAX;

  // the Box-Muller transform
  double rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

  // random normal distribution(mean, stdDev^2)
  double rand_normal = mean + std_dev * rand_std_normal;

  return rand_normal;
}

// function to generate a fixed noise pattern (FNP)
void generate_fixed_noise_pattern(unsigned char* noise_pattern, int size, double noise_factor) {

  // use a fixed seed for deterministic random numbers
  srand(1); 

  // generate noise pattern
  for(int i = 0; i < size; ++i) {
    noise_pattern[i] = (unsigned char) (noise_factor * rand() / RAND_MAX);
  }
}

// function to add noise to image
void add_noise_to_image(unsigned char* image, unsigned char* noise_pattern, int size, double noise_factor, int noise_type) {

  // use current time as seed for random number generator
  srand(time(NULL)); 

  for(int i = 0; i < size; ++i) {
    int noisy_value = 0;

    if (noise_type == 0) { 
      // Gaussian noise
      noisy_value = (int) ((double) image[i] + noise_factor * generate_gaussian_noise(0, 1));
    } else {
      // fixed pattern noise (FPN)
      noisy_value = (int) ((double) image[i] + (double) noise_pattern[i]);
    }

    // clamp the noisy_value to [0, 255]
    noisy_value = noisy_value < 0 ? 0 : noisy_value;
    noisy_value = noisy_value > 255 ? 255 : noisy_value;

    // re-use the image input buffer as the image output buffer
    image[i] = (unsigned char) noisy_value;
  }
}

// function to create the output filename
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

// the main function
// usage: ./noiser <image_filepath> <noise_factor> <noise_type>
// arguments:
//  - image_filepath: The filepath of the image input.
//  - noise_type: The noise factor to determine how much noise to apply (e.g. 150).
//  - noise_type: The noise type is a flag where 0 means Gaussian noise and 1 means fixed pattern noise.
int main(int argc, char *argv[]) {

  // check for expected number of arguments
  if(argc != 4) {
    printf("Usage: %s <image> <noise_factor> <noise_type>\n", argv[0]);
    return -1;
  }

  // create output filename
  char* output_filename = create_output_filename(argv[1]);
  if(output_filename == NULL) {
    printf("Error in creating the output filename\n");
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
  int noise_type = atoi(argv[3]);

  // allocate a buffer for the noise pattern
  unsigned char* noise_pattern = malloc(size * sizeof(unsigned char));
  if(noise_pattern == NULL) {
    printf("Error in allocating memory for the noise pattern\n");
    return -1;
  }

  // generate fixed pattern noise (FPN)
  if (noise_type == 1) {
    generate_fixed_noise_pattern(noise_pattern, size, noise_factor);
  }

  // add noise to the image
  add_noise_to_image(image, noise_pattern, size, noise_factor, noise_type);

  // write the noisy image
  stbi_write_jpg(output_filename, width, height, channels, image, 100);

  // deallocate resources
  free(output_filename);
  free(noise_pattern);
  stbi_image_free(image);

  return 0;
}