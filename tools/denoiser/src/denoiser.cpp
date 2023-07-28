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

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* define the name of this program for convenience purposes when printing instructions */
#define PROGRAM_NAME                                                                         "denoiser"

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

/* define convenience macros */
#define streq(s1,s2)    (!strcmp ((s1), (s2)))

/* tensorflow error codes */
typedef enum {
  TF_ALLOCATE_TENSOR              = 11, /* Error allocating tensors */
  TF_RESIZE_TENSOR                = 12, /* Error resizing tensor */
  TF_ALLOCATE_TENSOR_AFTER_RESIZE = 13, /* Error allocating tensors after resize */
  TF_COPY_BUFFER_TO_INPUT         = 14, /* Error copying input from buffer */
  TF_INVOKE_INTERPRETER           = 15, /* Error invoking interpreter */
  TF_COPY_OUTOUT_TO_BUFFER        = 16, /* Error copying output to buffer */
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
      printf("%s [options] ...", PROGRAM_NAME);
      printf("\n  --input    / -i       the file path of the input image");
      printf("\n  --resize   / -r       resize the input image (optional, e.g. 224x224)");
      printf("\n  --model    / -m       the file path of the model");
      printf("\n  --write    / -w       the write mode of the output image (optional)"
              "\n\t0 - do not write a new image (equivalent to not specifying --write)"
              "\n\t1 - write a new image as a new file"
              "\n\t2 - write a new image that overwrites the input image file"
              "\n\t3 - same as option 2 but backs up the original input image"
            );
      printf("\n  --output   / -o       the output image (optional, overwrites -w)");
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
      printf("unknown option %s. Get help: ./%s -?\n", argv[argn], PROGRAM_NAME);

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
    printf("no image input path specified. Get help: ./%s -?\n", PROGRAM_NAME);

    /* program error exit code */
    /* 22 EINVAL invalid argument */
    return EINVAL;
  }

  // --------------------------------------------------------------------------
  // check that the model was given

  if(*argv_index_model == -1)
  {
    /* print error message */
    printf("no model specified. Get help: ./%s -?\n", PROGRAM_NAME);

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
// dispose of the model and interpreter objects.

void dispose_tflite_objects(TfLiteModel* pModel, TfLiteInterpreter* pInterpreter)
{
  if(pModel != NULL)
  {
    TfLiteModelDelete(pModel);
  }

  if(pInterpreter)
  {
    TfLiteInterpreterDelete(pInterpreter);
  }
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

  /* the size of the image buffer data */
  int img_buffer_size = input_width * input_height * channels;

  /* the normalized image buffer */
  float img_buffer_normalized[img_buffer_size];

  /* normalize the image's RGB values */
  for(int i = 0; i < img_buffer_size; i++)
  {
    img_buffer_normalized[i] = (float)img_buffer[i] / 255.0;
  }


  /**
   * STEP 3:
   *  - load the denoising model
   *  - prepare the image tensor input
   *  - set the input dimension
   */

  /* load the model */
  TfLiteModel* model = TfLiteModelCreateFromFile(model_filename);

  /* create the interpreter */
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, NULL);

  /* allocate tensors */
  tflStatus = TfLiteInterpreterAllocateTensors(interpreter);

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* there was an error */

    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_ALLOCATE_TENSOR;
  }

  /* in TensorFlow, images are represented by tensors of shape [height, width, channels] */
  int input_dimensions[4] = {1, input_height, input_width, channels};
  tflStatus = TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dimensions, 4);

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* there was an error */

    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_RESIZE_TENSOR;
  }

  /* need to reallocate tensors after resizing */
  tflStatus = TfLiteInterpreterAllocateTensors(interpreter);

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_ALLOCATE_TENSOR_AFTER_RESIZE;
  }

  /**
   * STEP 4:
   *  - invoke the TensorFlow intepreter given the input and the model
   */

  /* The input tensor */
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

  /* copy the image data into the input tensor */
  tflStatus = TfLiteTensorCopyFromBuffer(input_tensor, img_buffer_normalized, img_buffer_size * sizeof(float));
  
  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* there was an error */

    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_COPY_BUFFER_TO_INPUT;
  }

  /* invoke the interpreter */
  tflStatus = TfLiteInterpreterInvoke(interpreter);

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* there was an error */

    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_INVOKE_INTERPRETER;
  }


  /**
   * STEP 5:
   *  - feed the input image into the denoiser model
   *  - get the denoised output image
   */

  /* extract the output tensor data */
  const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

  /* the model's output is a denoised image of the same size as the input image */
  float img_buffer_denoised[img_buffer_size];
  tflStatus = TfLiteTensorCopyToBuffer(output_tensor, img_buffer_denoised, img_buffer_size * sizeof(float));

  /* error check */
  if(tflStatus != kTfLiteOk)
  {
    /* deallocate the TensorFlow Lite objects */
    dispose_tflite_objects(model, interpreter);

    /* end of program */
    return TF_COPY_OUTOUT_TO_BUFFER;
  }

  /**
   * STEP 6:
   *  - denormalize the output image
   *  - write the output into an image file
   */

  /* normalize the denoised output image's RGB values */
  uint8_t img_buffer_denoised_denormalized[img_buffer_size];
  for(int i=0; i<img_buffer_size; i++)
  {
    /* round to the nearest integer */
    img_buffer_denoised_denormalized[i] = (uint8_t)(img_buffer_denoised[i] * 255 + 0.5);
  }
  
  /* write the denoised image */
  if(argv_index_output != -1)
  {
    /* the filename of the input image */
    char *outimg_filename = argv[argv_index_output];

    /* write the noisy image */
    stbi_write_jpg(outimg_filename, input_width, input_height, channels, img_buffer_denoised_denormalized, jpeg_write_quality);
  }
  else if(write_mode >= 1)
  {
    /* build file name output string (the file name of the output image that will be written) */
    char outimg_filename[BUFFER_MAX_SIZE_FILENAME] = {0};
    rc = build_image_output_filename(write_mode, inimg_filename, outimg_filename);

    /* error check */
    if(rc != 0)
    {
      /* there was an error */

      /* free the input image data buffer */
      stbi_image_free(img_buffer);

      /* end of program */
      return rc;
    }

    /* write the denoised image */
    stbi_write_jpg(outimg_filename, input_width, input_height, channels, img_buffer_denoised_denormalized, jpeg_write_quality);
  }
  
  /* deallocate resources */
  stbi_image_free(img_buffer);

#if TARGET_BUILD_OPSSAT /* this logic is specific to the OPS-SAT spacecraft */
  /* create classification result json object (because that's what the OPS-SAT SmartCam expects) */
  /* in this case we aren't classifying anything so just apply a constant "denoised" classifier label all the time */
  printf("{\"%s\": 1}", OUTPUT_IMAGE_LABEL);
#endif

  /* end program */
  return 0;
}