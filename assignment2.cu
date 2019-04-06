/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (image_filename) and generates
 * an output PGM image (image_filename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

enum kernel_type
{
  SHARPEN = 0,
  AVERAGE = 1,
  EDGE    = 2
};

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
// Define the files that are to be save ";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

////////////////////////////////////////////////////////////////////////////////
// Constants
const int edge_detect_kernel[3][3] = {
                                       {-1, 0, 1},
                                       {-2, 0, 2},
                                       {-1, 0, 1},
                                     };

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Generate the Kernel
////////////////////////////////////////////////////////////////////////////////
int generateKernel(float *kernel, int dim, kernel_type type)
{
  // The matrix has to be odd sized so if we take the integer division
  // we get the center of the matrix. We use this later for the SHARPEN
  // mask, ie set the center of the matrix to +9
  int center = dim / 2;
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      // For SHARPEN, all values are -1 except the center
      if (SHARPEN == type)
      {
        kernel[i*dim + j] = -1; 
      }
      // For AVERAGE, all values are 1/9
      else if (AVERAGE == type)
      {
        kernel[i*dim + j] = (float)1/9; 
      }
      // For EDGE, we have a 3x3 kernel which is read from above
      else if (EDGE == type)
      {
        // Dimensions has to be 3x3 otherwise return
        if (3 == dim)
        {
          kernel[i*dim + j] = edge_detect_kernel[i][j]; 
        }
        else
        {
          return 0;
        }
      }
    }
  }

  // Set the center of the SHARPEN kernel to +9
  if (SHARPEN == type)
    kernel[center*dim + center] = 9;

#ifdef DEBUG
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      printf("%f ",kernel[i*dim + j]);
    }
    printf("\n");
  }
#endif
  return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Pad the input Matrix
////////////////////////////////////////////////////////////////////////////////
int padInputData(float *hData, int width, int height, float *hPaddedData, int kernel_dim)
{
  int pad_size = kernel_dim / 2; 

  // Resize the new matrix to 2x the pad_size, once for top and bottom and once for left and right
  width += 2 * pad_size;
  height += 2 * pad_size;

  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      // Set all elements in the new padded matrix to 0.0 then update it with the input matrix if we are outside of
      // the padding area
      hPaddedData[i*width + j] = 0;

      // Only fill in matrix if we are in the middleware of the matrix, ie if (row >= pad_size & row >= pad_size) 
      // & (row < width-pad_size & row < height-pad_size)
      if ((i >= pad_size) && (j >= pad_size) && (i < (width - pad_size)) && (j < (height - pad_size)))
      {
        hPaddedData[i*width + j] = hData[i*(width - 2*pad_size) + (j - pad_size)];
      }
    }
  }

#ifdef DEBUG1
  /*for (int i = 0; i < 10; i++)*/
  for (int i = 497; i < 512; i++)
  {
    /*for (int j = 0; j < 10; j++)*/
    for (int j = 497; j < 512; j++)
    {
      printf("%f ",hData[i*512 + j]);
    }
    printf("\n");
  }
    printf("\n");
  /*for (int i = 0; i < 10; i++)*/
  for (int i = 499; i < width; i++)
  {
    /*for (int j = 0; j < 10; j++)*/
    for (int j = 499; j < height; j++)
    {
      printf("%f ",hPaddedData[i*width + j]);
    }
    printf("\n");
  }
#endif
  return 1;
}
////////////////////////////////////////////////////////////////////////////////
//! Serial convolution on CPU
////////////////////////////////////////////////////////////////////////////////
void serialConvolutionCPU(float *inputData,
                                int width,
                                int height,
                                float* kernel,
                                int dim,
                                float *outputData)
{
  // loop over the input image
  for (int i = 0; i < (width - 1); i++)
  {
    for (int j = 0; j < (height - 1); j++)
    {
      float sum =0.0;
      // loop over the kernel
      for (int x = 0; x < dim; x++)
      {
        for (int y = 0; y < dim; y++)
        {
          // Do the convolution
          sum += inputData[(x + i)*width + (y + j)] * edge_detect_kernel[x][y]; 
        }
      }
      outputData[(i + 1)*width + (j + 1)] = sum;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);
void serialTransformCPU(float *inputData,
                                int width,
                                int height,
                                int *outputData);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    // Process command-line arguments
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "input"))
        {
            getCmdLineArgumentString(argc,
                                     (const char **) argv,
                                     "input",
                                     (char **) &imageFilename);

            if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
            {
                getCmdLineArgumentString(argc,
                                         (const char **) argv,
                                         "reference",
                                         (char **) &refFilename);
            }
            else
            {
                printf("-input flag should be used with -reference flag");
                exit(EXIT_FAILURE);
            }
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
        {
            printf("-reference flag should be used with -input flag");
            exit(EXIT_FAILURE);
        }
    }

    runTest(argc, argv);

    cudaDeviceReset();
    printf("%s completed, returned %s\n",
           sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);

    printf("Loaded '%s', %d x %d pixels took %d bytes\n", imageFilename, width, height, size);

    // Generate Kernel
    int kernel_dim = 5;
    kernel_type type = SHARPEN; 
    float *kernel = (float *)malloc(kernel_dim*kernel_dim * sizeof(int));
    generateKernel(kernel, kernel_dim, type);

    // Pad matrix according to the dimension of the kernel. For example, if
    // the kernel dimension is 3, we need to pad the input data by 1 row above 
    // and below and 1 coloumn before and after. If the dimension of the kernel
    // is 5, then we need to pad by 2, 7 --> pad by 3 and so on.
    unsigned long paddedsize = (width + 2*(kernel_dim/2)) * (height + 2*(kernel_dim/2)) * sizeof(float);
    float *hPaddedData = (float *)malloc(paddedsize);
    padInputData(hData, width, height, hPaddedData, kernel_dim);
    
    // Run the serial convolution on the CPU
    float *hSerialDataOut = (float *) malloc(paddedsize);
    serialConvolutionCPU(hData, width, height, kernel, kernel_dim, hSerialDataOut);

    sdkSavePGM("./data/conv_output.pgm", hSerialDataOut, width, height);

    free(imagePath);
    free(hSerialDataOut);
    free(kernel);
    free(hPaddedData);
}

