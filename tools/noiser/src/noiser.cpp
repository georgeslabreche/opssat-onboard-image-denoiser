#include <iostream>
#include <vector>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* define the name of this program for convenience purposes when printing instructions */
#define PROGRAM_NAME                                                                           "noiser"

/* indicate if we are building for the OPS-SAT spacecraft or not */
#define TARGET_BUILD_OPSSAT                                                                           1

/* default jpeg write quality */
#define DEFAULT_JPEG_WRITE_QUALITY                                                                  100

/* define convenience macros */
#define streq(s1,s2)    (!strcmp ((s1), (s2)))



// --------------------------------------------------------------------------
// parse the program options

int parse_options(int argc, char **argv,
    int *argv_index_input, int *argv_index_write_mode, int *argv_index_write_quality,
    int *argv_index_noise_factor, int *argv_index_noise_type,
    int *argv_index_resize)
{
  int argn;
  for (argn = 1; argn < argc; argn++)
  {
    if (streq (argv [argn], "--help")
    ||  streq (argv [argn], "-?"))
    {
      printf("%s [options] ...", PROGRAM_NAME);
      printf("\n  --input    / -i       the file path of the input image");
      printf("\n  --write    / -w       the write mode of the output image (optional)"
              "\n\t0 - do not write a new image (equivalent to not specifying --write)"
              "\n\t1 - write a new image as a new file"
              "\n\t2 - write a new image that overwrites the input image file"
              "\n\t3 - same as option 2 but backs up the original input image"
            );
      printf("\n  --quality  / -q       the jpeg output quality (optional, from 1 to 100)");
      printf("\n  --noise    / -n       the noise factor (e.g. 50, 100, 150...)");
      printf("\n  --type     / -t       the noise type (0 for Gaussian noise, 1 for FPN, and 2 for column FPN)");
      printf("\n  --resize   / -r       resize the input image (e.g. 224x224)");
      printf("\n  --help     / -?       this information\n");
      
      /* program error exit code */
      /* 11 	EAGAIN 	Try again */
      return EAGAIN;
    }
    else
    if (streq (argv[argn], "--input")
    ||  streq (argv[argn], "-i"))
      *argv_index_input = ++argn;
    else
    if (streq (argv[argn], "--write")
    ||  streq (argv[argn], "-w"))
      *argv_index_write_mode = ++argn;
    else
    if (streq (argv [argn], "--quality")
    ||  streq (argv [argn], "-q"))
      *argv_index_write_quality = ++argn;
    else
    if (streq (argv[argn], "--noise")
    ||  streq (argv[argn], "-n"))
      *argv_index_noise_factor = ++argn;
    else
    if (streq (argv[argn], "--type")
    ||  streq (argv[argn], "-t"))
      *argv_index_noise_type = ++argn;
    else
    if (streq (argv[argn], "--resize")
    ||  streq (argv[argn], "-r"))
      *argv_index_resize = ++argn;
    else
    {
      /* print error message */
      printf("Unknown option %s. Get help: ./%s -?\n", argv[argn], PROGRAM_NAME);

      /* program error exit code */
      /* 22 	EINVAL 	Invalid argument */
      return EINVAL;
    }
  }


  // --------------------------------------------------------------------------
  // check that image input was given

  if(*argv_index_input == -1)
  {
    /* print error message */
    printf("No image input path specified. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 	EINVAL 	Invalid argument */
    return EINVAL;
  }

  // --------------------------------------------------------------------------
  // check that the noise factor value was given

  if(*argv_index_noise_factor == -1)
  {
    /* print error message */
    printf("No noise factor specified. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 	EINVAL 	Invalid argument */
    return EINVAL;
  }

  // --------------------------------------------------------------------------
  // check that the noise type value was given

  if(*argv_index_noise_type == -1)
  {
    /* print error message */
    printf("No noise type specified. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 	EINVAL 	Invalid argument */
    return EINVAL;
  }

  /* success */
  return 0;
}


// --------------------------------------------------------------------------
// build file name output string (the file name of the output image that will be written)

int build_image_output_filename(int write_mode, char* inimg_filename, char *outimg_filename, char *image_file_ext)
{
  switch(write_mode)
  {
    case 1: /* write a new image as a new file */
      /* create new file name for the output image file */
      strncpy(outimg_filename, inimg_filename, strcspn(inimg_filename, "."));
      strcat(outimg_filename, ".noisy.");
      strcat(outimg_filename, image_file_ext);

      break;

    case 2: /* write a new image that overwrites the input image file */
      /* use existing input image file name as the the output image file name */
      strcpy(outimg_filename, inimg_filename);
      
      break;

    case 3: /*  write a new image that overwrites the input image file but back up the original input image */
      char inimg_filename_new[100] = {0};
      strncpy(inimg_filename_new, inimg_filename, strcspn(inimg_filename, "."));
      strcat(inimg_filename_new, ".original.");
      strcat(inimg_filename_new, image_file_ext);
      rename(inimg_filename, inimg_filename_new);

      /* use existing input image file name as the output image file name */
      strcpy(outimg_filename, inimg_filename);

      break;
  }

  /* success */
  return 0;
}

// --------------------------------------------------------------------------
// function to generate Gaussian noise

double generate_gaussian_noise(double mean, double std_dev)
{
  /* uniformly distributed random number between 0 and 1 */
  double u1 = (double) rand() / RAND_MAX;
  double u2 = (double) rand() / RAND_MAX;

  /* the Box-Muller transform */
  double rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

  /* random normal distribution(mean, stdDev^2) */
  double rand_normal = mean + std_dev * rand_std_normal;

  return rand_normal;
}

// --------------------------------------------------------------------------
// function to generate a fixed noise pattern (FNP)

void generate_fixed_noise_pattern(unsigned char* noise_pattern, int size, double noise_factor)
{

  /* use a fixed seed for deterministic random numbers */
  srand(1);

  /* generate noise pattern */
  for(int i = 0; i < size; ++i)
  {
    noise_pattern[i] = (unsigned char) (noise_factor * rand() / RAND_MAX);
  }
}

// --------------------------------------------------------------------------
// function to add noise to image

void add_noise_to_image(unsigned char* image, unsigned char* noise_pattern, int width, int height, int channels, double noise_factor, int noise_type)
{

  /* use current time as seed for random number generator */
  srand(time(NULL)); 

  for(int y = 0; y < height; ++y)
  {
    for(int x = 0; x < width; ++x)
    {
      for(int c = 0; c < channels; ++c)
      {
        int i = (y * width * channels) + (x * channels) + c;
        int noisy_value = 0;

        if (noise_type == 0)
        {
          /* Gaussian noise */
          noisy_value = (int) ((double) image[i] + noise_factor * generate_gaussian_noise(0, 1));
        }
        else if (noise_type == 1)
        {
          /* fixed pattern noise (FPN) */
          noisy_value = (int) ((double) image[i] + (double) noise_pattern[i]);
        }
        else if (noise_type == 2)
        {
          /* column fixed pattern noise */
          noisy_value = (int) ((double) image[i] + (double) noise_pattern[x * channels + c]);
        }

        /* clamp the noisy_value to [0, 255] */
        noisy_value = noisy_value < 0 ? 0 : noisy_value;
        noisy_value = noisy_value > 255 ? 255 : noisy_value;

        /* re-use the image input buffer as the image output buffer */
        image[i] = (unsigned char) noisy_value;
      }
    }
  }
}


// --------------------------------------------------------------------------
// the main function

int main(int argc, char **argv)
{
  /* the return code */
  int rc = 0;

  /* get provider host and port from command arguments */
  int argv_index_input = -1;
  int argv_index_write_mode = -1;
  int argv_index_write_quality = -1;
  int argv_index_noise_factor = -1;
  int argv_index_noise_type = -1;
  int argv_index_resize = -1;

  /* parse the program options */
  rc = parse_options(argc, argv,
    &argv_index_input, &argv_index_write_mode, &argv_index_write_quality, &argv_index_noise_factor, &argv_index_noise_type, &argv_index_resize);

  /* error check */
  if(rc != 0)
  {
    /* there was an error */
    /* end of program */
    return rc;
  }


  /* the filename of the input image */
  char *inimg_filename = argv[argv_index_input];

  /* get the extension of the image file */
  char image_file_ext[10] = {0};
  strncpy(image_file_ext,
    inimg_filename + strcspn(inimg_filename, ".") + 1,
    strlen(inimg_filename) - strcspn(inimg_filename, "."));


  /* check that write mode was given and if not set it to default value */
  /* the write mode variable */
  int write_mode;

  /* check that write mode was given and if not set it to default value */
  if(argv_index_write_mode == -1)
  {
    /* default write mode is 0 */
    write_mode = 0;
  }
  else
  {
    /* cast given write mode option to integer */
    write_mode = atoi(argv[argv_index_write_mode]);

    /* check that the write mode value is valid */
    if(write_mode < 0 || write_mode > 3)
    {
      /* print error message */
      printf("Invalid write mode option. Get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 	EINVAL 	Invalid argument */
      return EINVAL;
    }
  }

  /* set the default jpeg write quality */
  int jpeg_write_quality = DEFAULT_JPEG_WRITE_QUALITY;

  /* overwrite the default jpeg write quality if a target value is given */
  if(argv_index_write_quality != -1)
  {
     /* cast given write quality option to integer */
    jpeg_write_quality = atoi(argv[argv_index_write_quality]);

    /* check that the write quality value is valid */
    if(jpeg_write_quality <= 0 || jpeg_write_quality > 100)
    {
      /* print error message */
      printf("Invalid write quality option. Get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 	EINVAL 	Invalid argument */
      return EINVAL;
    }
  }

  /* check that the noise factor value is valid */
  double noise_factor = atof(argv[argv_index_noise_factor]);
  if(noise_factor <= 0)
  {
    /* print error message */
    printf("Invalid noise factor value option. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 	EINVAL 	Invalid argument */
    return EINVAL;
  }

  /* check that the noise type value is valid */
  int noise_type = atoi(argv[argv_index_noise_type]);
  if(noise_type < 0 || noise_type > 2)
  {
    /* print error message */
    printf("Invalid noise type value option. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 	EINVAL 	Invalid argument */
    return EINVAL;
  }

  /* check that the image resize was given */
  int resize_width, resize_height;
  if(argv_index_resize != -1)
  {
    char *resize = argv[argv_index_resize];

    /* extract the resize values from the string and assign them to variables */
    if (sscanf(resize, "%dx%d", &resize_width, &resize_height) != 2)
    {
      /* print error message */
      printf("Invalid resize value option. Get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 	EINVAL 	Invalid argument */
      return EINVAL;
    }
  }

  /* read the image */
  int input_width, input_height, channels;
  unsigned char* img_buffer = stbi_load(inimg_filename, &input_width, &input_height, &channels, 0);

  /* resize the image */
  if(argv_index_resize != -1)
  {
    /* dynamically allocate the resized image buffer so that it can later overwrite the original image buffer */
    int img_buffer_resized_size = resize_width * resize_height * channels;
    unsigned char* img_buffer_resized = (unsigned char*) malloc(img_buffer_resized_size * sizeof(unsigned char));

    /* downsample the image i.e., resize the image to a smaller dimension */
    rc = stbir_resize_uint8(img_buffer, input_width, input_height, 0, img_buffer_resized, resize_width, resize_height, 0, channels);

    /* error check */
    /* confusingly, stb result is 1 for success and 0 in case of an error */
    if(rc != 1)
    {
      /* free the input image data buffers */
      stbi_image_free(img_buffer);
      stbi_image_free(img_buffer_resized);

      /* end of program */
      /* return 1 for error, not the rc error value of 0 set by stb */
      return 1;
    }

    /* free the memory for the original img_buffer and point it to img_buffer_resized */
    stbi_image_free(img_buffer);
    img_buffer = img_buffer_resized;

    /* reset the input image dimensions to the resized value */
    input_width = resize_width;
    input_height = resize_height;
  }

  /* build file name output string (the file name of the output image that will be written) */
  char outimg_filename[100] = {0};
  rc = build_image_output_filename(write_mode, inimg_filename, outimg_filename, image_file_ext);

  /* error check */
  if(rc != 0)
  {
    /* there was an error */

    /* free the input image data buffer */
    stbi_image_free(img_buffer);

    /* end of program */
    return rc;
  }

  /* the size of the image buffer data */
  int img_buffer_size = input_width * input_height * channels;

  /* buffer for the noise pattern */
  unsigned char noise_pattern[img_buffer_size];

  /* generate fixed pattern noise (FPN) */
  if (noise_type == 1 || noise_type == 2)
  {
    int pattern_size = noise_type == 1 ? img_buffer_size : input_width * channels;
    generate_fixed_noise_pattern(noise_pattern, pattern_size, noise_factor);
  }

  /* add noise to the image */
  add_noise_to_image(img_buffer, noise_pattern, input_width, input_height, channels, noise_factor, noise_type);

  /* write the noisy image */
  stbi_write_jpg(outimg_filename, input_width, input_height, channels, img_buffer, jpeg_write_quality);

  /* deallocate resources */
  stbi_image_free(img_buffer);

#if TARGET_BUILD_OPSSAT /* this logic is specific to the OPS-SAT spacecraft */
  /* create classification result json object (because that's what the OPS-SAT SmartCam expects) */
  /* in this case we aren't classifying anything so just apply a constant "noisy" classifier label all the time */
  printf("{");

  /* there's a bug in the SmartCam: it doesn't ignore the keys that are prefixed with underscore and thus interprets them as labels */
  /* this is a problem when the metadata values are greater than the "noisy" label value (it will label the image as the name of the metadata) */
  /* workaround: just set a very high value for the noisy label */
  printf("\"noisy\": 10000, ");
  printf("\"_noise_factor\": %d, ", static_cast<int>(noise_factor)); /* prefixed by an underscore means it's metadata, not a label */
  printf("\"_noise_type\": %d", noise_type);                         /* prefixed by an underscore means it's metadata, not a label */
  printf("}");
#endif

  /* end of program */
  return 0;
}