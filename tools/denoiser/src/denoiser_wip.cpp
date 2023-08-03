#include <iostream>
#include <vector>
#include <dirent.h>
#include <cstring>
#include <string.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdlib.h>
#include <time.h>

#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* define the name of this program for convenience purposes when printing instructions */
#define PROGRAM_NAME                                                                         "denoiser"

/* define the program version */
#define PROGRAM_VERSION_MAJOR                                                                         2
#define PROGRAM_VERSION_MINOR                                                                         0

/* the output image label */
#define OUTPUT_IMAGE_LABEL                                                                   "denoised"

/* the input image label */
/* for the filename used to save the original image as backup */
#define INPUT_IMAGE_LABEL                                                              "original.noised"

/* indicate if we are building for the OPS-SAT spacecraft or not */
#define TARGET_BUILD_OPSSAT                                                                           1

/* default jpeg write quality */
#define DEFAULT_JPEG_WRITE_QUALITY                                                                  100

/* buffer size for the output noisy image filename */
#define BUFFER_MAX_SIZE_FILENAME                                                                    256

/* TODO: */
/*  - make this an argument that takes width and height, e.g. 56x56 or 100x20 */
/*  - the argument is optional, if not given then process the image without patching it */
#define PATCH_SIZE                                                                                   56

/* define convenience macros */
#define streq(s1,s2)    (!strcmp ((s1), (s2)))

/* tensorflow error codes */
typedef enum {
  TF_LOAD_MODEL                   = 11, /* error loading model */
  TF_BUILD_INTERPRETER            = 12, /* error building interpreter */
  TF_ALLOCATE_TENSOR              = 13, /* error allocating tensors */
  TF_INVOKE_INTERPRETER           = 14, /* error invoking interpreter */
} tf_error_code_t;

// --------------------------------------------------------------------------
// parse the program options

int parse_options(int argc, char **argv,
    int *argv_index_input, int *argv_index_resize, int *argv_index_model,
    int *argv_index_write_mode, int *argv_index_output, int *argv_index_write_quality)
{
  int argn;
  for (argn = 1; argn < argc; argn++)
  {
    if (streq (argv [argn], "--help")
    ||  streq (argv [argn], "-?"))
    {
      printf("%s v%d.%d [options] ...", PROGRAM_NAME, PROGRAM_VERSION_MAJOR, PROGRAM_VERSION_MINOR);
      printf("\n  --input    / -i       the file path of the input image");
      printf("\n  --resize   / -r       resize the input image (optional, e.g. 224x224)");
      printf("\n  --model    / -m       the file path of the model");
      printf("\n  --write    / -w       the write mode of the output image (optional)"
              "\n\t0 - do not write a new image (equivalent to not specifying --write)"
              "\n\t1 - write a new image as a new file"
              "\n\t2 - write a new image that overwrites the input image file"
              "\n\t3 - same as option 2 but backs up the original input image"
            );
      printf("\n  --output   / -o       the output image (optional, overwrites --write)");
      printf("\n  --quality  / -q       the jpeg output quality (optional, from 1 to 100)");
      printf("\n  --help     / -?       this information\n");

      /* program error exit code */
      /* 11 EAGAIN try again */
      return EAGAIN;
    }
    else
    if (streq (argv[argn], "--input")
    ||  streq (argv[argn], "-i"))
      *argv_index_input = ++argn;
    else
    if (streq (argv[argn], "--resize")
    ||  streq (argv[argn], "-r"))
      *argv_index_resize = ++argn;
    else
    if (streq (argv[argn], "--model")
    ||  streq (argv[argn], "-m"))
      *argv_index_model = ++argn;
    else
    if (streq (argv[argn], "--write")
    ||  streq (argv[argn], "-w"))
      *argv_index_write_mode = ++argn;
    else
    if (streq (argv[argn], "--output")
    ||  streq (argv[argn], "-o"))
      *argv_index_output = ++argn;
    else
    if (streq (argv [argn], "--quality")
    ||  streq (argv [argn], "-q"))
      *argv_index_write_quality = ++argn;
    else
    {
      /* print error message */
      printf("unknown option %s. get help: ./%s -?\n", argv[argn], PROGRAM_NAME);

      /* program error exit code */
      /* 22 EINVAL invalid argument */
      return EINVAL;
    }
  }

  // --------------------------------------------------------------------------
  // check that image input was given

  if(*argv_index_input == -1)
  {
    /* print error message */
    printf("no image input path specified. get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 EINVAL invalid argument */
    return EINVAL;
  }

  // --------------------------------------------------------------------------
  // check that the model was given

  if(*argv_index_model == -1)
  {
    /* print error message */
    printf("no model specified. get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 EINVAL invalid argument */
    return EINVAL;
  }

  /* success */
  return 0;
}


// --------------------------------------------------------------------------
// build file name output string (the file name of the output image that will be written)

int build_image_output_filename(int write_mode, char* inimg_filename, char *outimg_filename)
{
  /* the return code */
  int rc = 0;

  /* duplicate the inimg_filename before passing to basename() and dirname() */
  char *tmp1 = strdup(inimg_filename);
  char *tmp2 = strdup(inimg_filename);

  /* get base filename */
  char* base = basename(tmp1);

  /* get directory name */
  char* dir = dirname(tmp2);

  /* duplicate base filename to separate it from the extension */
  char *base_copy = strdup(base);

  /* get file extension */
  char* ext = strrchr(base_copy, '.');

  /* remove the file extension (including the dot) */
  if(ext != NULL)
  {
    /* null-terminate the base filename at the start of the extension */
    *ext = '\0';

    /* move past the period */
    ext++;
  }

  /* build output filename based on write mode */
  switch(write_mode)
  {
    case 1: /* write a new image as a new file */
      /* create new file name for the output image file */
      snprintf(outimg_filename, BUFFER_MAX_SIZE_FILENAME, "%s/%s.%s.%s", dir, base_copy, OUTPUT_IMAGE_LABEL, ext);
      break;

    case 2: /* write a new image that overwrites the input image file */
      /* use existing input image file name as the output image file name */
      snprintf(outimg_filename, BUFFER_MAX_SIZE_FILENAME, "%s/%s.%s", dir, base_copy, ext);
      break;

    case 3: /* write a new image that overwrites the input image file but back up the original input image */

      /* construct the new filename for the original file backup */
      char inimg_filename_new[BUFFER_MAX_SIZE_FILENAME] = {0};
      snprintf(inimg_filename_new, BUFFER_MAX_SIZE_FILENAME, "%s/%s.%s.%s", dir, base_copy, INPUT_IMAGE_LABEL, ext);

      /* rename the original file to the backup filename */
      rc = rename(inimg_filename, inimg_filename_new);

      /* error check */
      if(rc != 0)
      {
        /* print error message */
        printf("error renaming file from %s to %s", inimg_filename, inimg_filename_new);
      }
      else
      {
        /* use the new filename (without ".original") for the output file */
        snprintf(outimg_filename, BUFFER_MAX_SIZE_FILENAME, "%s/%s.%s", dir, base_copy, ext);
      }

      break;
  }

  /* free the duplicated strings */
  free(tmp1);
  free(tmp2);
  free(base_copy);

  /* success */
  return rc;
}


// --------------------------------------------------------------------------
// the main function

int main(int argc, char **argv)
{
  /* the return code */
  int rc = 0;

  /* TensorFlow Lite operation execution status for error handling */
  TfLiteStatus tflStatus;

  /**
   * STEP 1:
   *  - parse the program options
   *  - initalize variables
   *  - check for invalid values
   */

  /* get provider host and port from command arguments */
  int argv_index_input = -1;
  int argv_index_resize = -1;
  int argv_index_model = -1;
  int argv_index_write_mode = -1;
  int argv_index_output = -1;
  int argv_index_write_quality = -1;

  /* parse the program options */
  rc = parse_options(argc, argv,
    &argv_index_input, &argv_index_resize, &argv_index_model,
    &argv_index_write_mode, &argv_index_output, &argv_index_write_quality);

  /* error check */
  if(rc != 0)
  {
    /* there was an error */
    /* end of program */
    return rc;
  }

  /* the filename of the input image */
  char *inimg_filename = argv[argv_index_input];

  /* check if the given filename exists */
  rc = access(inimg_filename, F_OK);
  if(rc != 0)
  {
    /* print error message */
    printf("invalid input image file path value option (file does not exist). get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 EINVAL invalid argument */
    return EINVAL;
  }

  /* check if the image resize was given */
  int resize_width, resize_height;
  if(argv_index_resize != -1)
  {
    char *resize = argv[argv_index_resize];

    /* extract the resize values from the string and assign them to variables */
    if (sscanf(resize, "%dx%d", &resize_width, &resize_height) != 2)
    {
      /* print error message */
      printf("invalid resize value option. get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 EINVAL invalid argument */
      return EINVAL;
    }
  }

  /* the filename of the model */
  char *model_filename = argv[argv_index_model];

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
      printf("invalid write mode option. get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 EINVAL invalid argument */
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
      printf("invalid write quality option. get help: ./%s -?\n", PROGRAM_NAME);

      /* program error exit code */
      /* 22 EINVAL invalid argument */
      return EINVAL;
    }
  }


  /**
   * STEP 2:
   *  - read the input image into a data buffer
   *  - normalize the image's RGB data
   *  - prepare the image tensor input
   */

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


  /**
   * STEP 3:
   *  - load the denoising model
   *  - prepare the image tensor input
   */

  /* load the model */
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_filename);

  /* error check */
  if(model == nullptr)
  {
    return TF_LOAD_MODEL;
  }

  /* create the interpreter */
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  /* error check */
  if(interpreter == nullptr)
  {
    return TF_BUILD_INTERPRETER;
  }

  /* allocate tensors */
  tflStatus = interpreter->AllocateTensors();

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* there was an error */

    /* end of program */
    return TF_ALLOCATE_TENSOR;
  }


  /**
   * STEP 4:
   *  - prepare the image tensor input
   *  - normalize the input image bufer into an input tensor
   */

  /* prepare the image tensor input */
  float *input_tensor = interpreter->typed_input_tensor<float>(0);

  /* the size of the image buffer data */
  int img_buffer_size = input_width * input_height * channels;


  char *outimg_filename = argv[argv_index_output];

  /* TODO: */
  /*  - turn this into buffer for the entire output image (with all patches sewn together) */
  /*  - add RGB data to this buffer as we denormalize the output tensor data after each patch inference */
  int img_buffer_denoised_denormalized_size = PATCH_SIZE * PATCH_SIZE * channels;
  unsigned char* img_buffer_denoised_denormalized = (unsigned char*) malloc(img_buffer_denoised_denormalized_size * sizeof(unsigned char));

  /* figure out how many patches are in the image */
  int pwidth_max = input_width / PATCH_SIZE;
  int pheight_max = input_height / PATCH_SIZE;

  /* the input tensor will contain normalized RGB data for a single patch of image at a given time */
  for (int w = 0; w < pwidth_max; w++)
  {
    for (int h = 0; h < pheight_max; h++)
    {
      /* allocate the tensor input buffer */
      for (int j = h * PATCH_SIZE; j < (h + 1) * PATCH_SIZE; j++)
      {
        for (int i = w * PATCH_SIZE; i < (w + 1) * PATCH_SIZE; i++)
        {
          for (int k = 0; k < 3; k++)
          {
            int offset = (((channels) * ((input_width * j) + i)) + k);
            input_tensor[((channels) * ((PATCH_SIZE * (j - h * PATCH_SIZE)) + (i - w * PATCH_SIZE))) + k] = (float)img_buffer[offset] / 255.0;
          }
        }
      }

      /* invoke the interpreter */
      if(interpreter->Invoke() != kTfLiteOk)
      {
        printf("Failed to invoke tflite!\n");
        return 1;
      }

      /* get the pointer to the tensor data */
      float *output_tensor = interpreter->typed_output_tensor<float>(0);
      
      
      /* denormalize the denoised output image's RGB values */
      for(int i=0; i<img_buffer_denoised_denormalized_size); i++)
      {
        /* round to the nearest integer */
        img_buffer_denoised_denormalized[i] = (unsigned char)(output_tensor[i] * 255 + 0.5);
      }

      /* TODO: */
      /*  - don't write each patch of the image */
      /*  - construct the complete denoised image patch by patch and then write the constructed image */
      stbi_write_jpg(patch_filename, PATCH_SIZE, PATCH_SIZE, channels, (void *)img_buffer_denoised_denormalized, jpeg_write_quality);
    }
  }

  /* free the image data buffer */
  stbi_image_free(img_buffer);
  stbi_image_free(img_buffer_denoised_denormalized);


#if TARGET_BUILD_OPSSAT /* this logic is specific to the OPS-SAT spacecraft */
  /* create classification result json object (because that's what the OPS-SAT SmartCam expects) */
  /* in this case we aren't classifying anything so just apply a constant "denoised" classifier label all the time */
  printf("{\"%s\": 1}", OUTPUT_IMAGE_LABEL);
#endif

  /* end program */
  return 0;
}
