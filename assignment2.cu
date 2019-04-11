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
#define TILE_SIZE 64 // 64 * 64 * 4 = 16384 < 49152bytes
#define MAX_SHARED_MEMORY_BYTES 16384
enum kernel_size
{
  SMALL  = 3,
  MEDIUM = 5,
  LARGE  = 7
};

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
        hPaddedData[i*width + j] = hData[(i - pad_size)*(width - 2*pad_size) + (j - pad_size)];
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

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using shared and constant memory for kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void parallelConvolutionSharedTextured(float *inputData,
                                                  int width,
                                                  int height,
                                                  int pad_size,
                                                  float *outputData)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  // Declare a square matrix of tile size in shared memory
  __shared__ float input[TILE_SIZE][TILE_SIZE];

  int kernel_dim = (2 * pad_size) + 1;
  
  for (int k = 0; k < (((width - 1)/ TILE_SIZE) + 1); k++)
  {
    // Make sure we are within the bound of the tile
    if ((row < height) && ((threadIdx.x + (k * TILE_SIZE)) < width))
    {
      // Copy the global memory to shared memory
      input[threadIdx.y][threadIdx.x] = inputData[(row*width) + threadIdx.x + (k*TILE_SIZE)];
    }
    else
    {
        input[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    float sum = 0.0; 
    for (int i = 0; i < kernel_dim; i++)
    {
      for (int j = 0; j < kernel_dim; j++)
      {
        // Do the convolution
        sum += input[row + i][col + j] * tex2D(tex,i * kernel_dim, j); 
      }
    }
    outputData[(row * width) + col] = sum;
  }
}

__constant__ float kernel_gpu[MAX_SHARED_MEMORY_BYTES];
////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using shared and constant memory for kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void parallelConvolutionSharedConstant(float *inputData,
                                                  int width,
                                                  int height,
                                                  int pad_size,
                                                  float *outputData)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  // Declare a square matrix of tile size in shared memory
  __shared__ float input[TILE_SIZE][TILE_SIZE];

  int kernel_dim = (2 * pad_size) + 1;
  
  for (int k = 0; k < (((width - 1)/ TILE_SIZE) + 1); k++)
  {
    // Make sure we are within the bound of the tile
    if ((row < height) && ((threadIdx.x + (k * TILE_SIZE)) < width))
    {
      // Copy the global memory to shared memory
      input[threadIdx.y][threadIdx.x] = inputData[(row*width) + threadIdx.x + (k*TILE_SIZE)];
    }
    else
    {
        input[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    float sum = 0.0; 
    for (int i = 0; i < kernel_dim; i++)
    {
      for (int j = 0; j < kernel_dim; j++)
      {
        // Do the convolution
        sum += input[row + i][col + j] * kernel_gpu[(i * kernel_dim) + j]; 
      }
    }
    outputData[(row * width) + col] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void parallelConvolutionShared(float *inputData,
                                                int width,
                                                int height,
                                                float* kernel,
                                                int pad_size,
                                                float *outputData)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  // Declare a square matrix of tile size in shared memory
  __shared__ float input[TILE_SIZE][TILE_SIZE];

  int kernel_dim = (2 * pad_size) + 1;
  
  for (int k = 0; k < (((width - 1)/ TILE_SIZE) + 1); k++)
  {
    // Make sure we are within the bound of the tile
    if ((row < height) && ((threadIdx.x + (k * TILE_SIZE)) < width))
    {
      // Copy the global memory to shared memory
      input[threadIdx.y][threadIdx.x] = inputData[(row*width) + threadIdx.x + (k*TILE_SIZE)];
    }
    else
    {
        input[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    float sum = 0.0; 
    for (int i = 0; i < kernel_dim; i++)
    {
      for (int j = 0; j < kernel_dim; j++)
      {
        // Do the convolution
        sum += input[row + i][col + j] * kernel[(i * kernel_dim) + j]; 
      }
    }
    outputData[(row * width) + col] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void parallelConvolutionGlobal(float *inputData,
                                          int width,
                                          int height,
                                          float* kernel,
                                          int kernel_dim,
                                          float *outputData)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  
  if ((x > width) || (y > height))
    return;

  float sum =0.0;
  // loop over the kernel
  for (int i = 0; i < kernel_dim; i++)
  {
    for (int j = 0; j < kernel_dim; j++)
    {
      // Do the convolution
      sum += inputData[(x + i)*width + (y + j)] * kernel[(i * kernel_dim) + j]; 
    }
  }
  outputData[(x * width) + y] = sum;
}

////////////////////////////////////////////////////////////////////////////////
//! Serial convolution on CPU
////////////////////////////////////////////////////////////////////////////////
void serialConvolutionCPU(float *inputData,
                                int width,
                                int height,
                                float* kernel,
                                int kernel_dim,
                                float *outputData)
{
  // loop over the input image
  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      float sum =0.0;
      // loop over the kernel
      for (int x = 0; x < kernel_dim; x++)
      {
        for (int y = 0; y < kernel_dim; y++)
        {
          // Do the convolution
          sum += inputData[(x + i)*width + (y + j)] * kernel[(x * kernel_dim) + y]; 
        }
      }
      outputData[(i * width) + j] = sum;
    }
  }
#ifdef DEBUG2
  printf("Data after serialConvolutionCPU\n");
  for (int i = 512; i < 530; i++)
  {
    printf("%f\n",*(outputData +i));
  }
#endif
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
//! Compare results
////////////////////////////////////////////////////////////////////////////////
int compareResults(float * serialData, float * parallelData, unsigned long size)
{
  for (unsigned long i = 0; i < size; i++)
  {
    if (*(serialData + i) != *(parallelData + i))
    {
      /*printf("%d %f -- %f\n",i,*(serialData + i), *(parallelData + i));*/
      return 0;
    }
  }
  return 1;
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

////////////////////////////// Generate Kernel ////////////////////////////////////////////////////
    int pad_size = 20;
    int kernel_dim = (2 * pad_size) + 1;
    kernel_type type = AVERAGE; 
    float *hKernel = (float *)malloc(kernel_dim*kernel_dim * sizeof(int));
    generateKernel(hKernel, kernel_dim, type);
////////////////////////////// Generate Kernel Complete //////////////////////////////////////////

////////////////////////////// Pad Input Data ////////////////////////////////////////////////////
    // Pad matrix according to the dimension of the kernel. For example, if
    // the kernel dimension is 3, we need to pad the input data by 1 row above 
    // and below and 1 coloumn before and after. If the dimension of the kernel
    // is 5, then we need to pad by 2, 7 --> pad by 3 and so on.
    unsigned int paddedWidth = width + 2*(kernel_dim/2);
    unsigned int paddedHeight = height + 2*(kernel_dim/2);
    unsigned long paddedSize = paddedWidth * paddedHeight * sizeof(float);
    float *hPaddedData = (float *)malloc(paddedSize);
    padInputData(hData, width, height, hPaddedData, kernel_dim);
////////////////////////////// Pad Input Data Complete ////////////////////////////////////////////
    
////////////////////////////// Serial Convolution /////////////////////////////////////////////////
    // Run the serial convolution on the CPU
    float *hSerialDataOut = (float *) malloc(paddedSize);
    StopWatchInterface *timer0 = NULL;
    sdkCreateTimer(&timer0);
    sdkStartTimer(&timer0);

    serialConvolutionCPU(hPaddedData, paddedWidth, paddedHeight, hKernel, kernel_dim, hSerialDataOut);

    sdkStopTimer(&timer0);
    printf("Processing time Global Memory SerialCPU: %f (ms)\n", sdkGetTimerValue(&timer0));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer0) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer0);
////////////////////////////// Serial Convolution Complete ////////////////////////////////////////

////////////////////////////// Naive Parallel Convolution /////////////////////////////////////////
    // Allocate device memory for input data
    float *dPaddedInputData = NULL;
    float *dParallelOutputData = NULL;
		// Allocate input device memory
    checkCudaErrors(cudaMalloc((void **) &dPaddedInputData, paddedSize));
    // Allocate device memory for result
    checkCudaErrors(cudaMalloc((void **) &dParallelOutputData, paddedSize));
		checkCudaErrors(cudaMemcpy(dPaddedInputData,
                               hPaddedData,
                               paddedSize,
                               cudaMemcpyHostToDevice));


    // Allocate device memory for kernel
    float *dKernel = NULL;
    checkCudaErrors(cudaMalloc((void **) &dKernel, kernel_dim*kernel_dim * sizeof(float)));
		checkCudaErrors(cudaMemcpy(dKernel,
                               hKernel,
                               kernel_dim*kernel_dim * sizeof(float),
                               cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid((paddedWidth / dimBlock.x)+1, (paddedHeight / dimBlock.y)+1, 1);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer1 = NULL;
    sdkCreateTimer(&timer1);
    sdkStartTimer(&timer1);
    // Execute the kernel
    parallelConvolutionGlobal<<<dimGrid, dimBlock, 0>>>(dPaddedInputData, paddedWidth, paddedHeight, dKernel, kernel_dim, dParallelOutputData);

    // Allocate host memory for the result;
    float *hParallelOutputData = (float *) malloc(paddedSize);
		checkCudaErrors(cudaMemcpy(hParallelOutputData,
                               dParallelOutputData,
                               paddedSize,
                               cudaMemcpyDeviceToHost));

     testResult = compareData(hParallelOutputData,
                              hSerialDataOut,
                              width*height,
                              MAX_EPSILON_ERROR,
                              0.15f);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed 1");
    checkCudaErrors(cudaFree(dPaddedInputData));
    checkCudaErrors(cudaFree(dKernel));
    checkCudaErrors(cudaFree(dParallelOutputData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer1);
    printf("Processing time Naive Solution: %f (ms)\n", sdkGetTimerValue(&timer1));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer1) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer1);
////////////////////////////// Naive Parallel Convolution Complete ////////////////////////////////////////

////////////////////////////// Shared Memory Parallel Convolution /////////////////////////////////////////
// The approach taken here is as follows
//    A1) Uses tiles so that we can fit the data into shared memory

		// Allocate input device memory
    checkCudaErrors(cudaMalloc((void **) &dPaddedInputData, paddedSize));
    // Allocate device memory for result
    checkCudaErrors(cudaMalloc((void **) &dParallelOutputData, paddedSize));
		checkCudaErrors(cudaMemcpy(dPaddedInputData,
                               hPaddedData,
                               paddedSize,
                               cudaMemcpyHostToDevice));

    // Allocate device memory for kernel
    checkCudaErrors(cudaMalloc((void **) &dKernel, kernel_dim*kernel_dim * sizeof(float)));
		checkCudaErrors(cudaMemcpy(dKernel,
                               hKernel,
                               kernel_dim*kernel_dim * sizeof(float),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    dim3 dimBlockShared(TILE_SIZE, 1, 1);
    dim3 dimGridShared((width / TILE_SIZE)+1, (height / TILE_SIZE)+1, 1);
    parallelConvolutionShared<<<dimGridShared, dimBlockShared, 0>>>(dPaddedInputData, paddedWidth, paddedHeight, dKernel, pad_size, dParallelOutputData);

    memset(hParallelOutputData,0,paddedSize);
    // Copy the reult back into host memory
    checkCudaErrors(cudaMemcpy(hParallelOutputData,
                               dParallelOutputData,
                               paddedSize,
                               cudaMemcpyDeviceToHost));
#ifdef DEBUG
  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      printf("%f ",hParallelOutputData[i*width + j]);
    }
    printf("\n");
  }
#endif

    testResult = compareData(hParallelOutputData,
                              hSerialDataOut,
                              paddedWidth*paddedHeight,
                              MAX_EPSILON_ERROR,
                              0.15f);

    // Check if kernel execution generated an error
    getLastCudaError("parallelConvolutionShared Kernel execution failed");
    checkCudaErrors(cudaFree(dPaddedInputData));
    checkCudaErrors(cudaFree(dKernel));
    checkCudaErrors(cudaFree(dParallelOutputData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time Shared Memory: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);
////////////////////////////// Shared Memory Parallel Convolution Complete ////////////////////////////////////////

////////////////////////////// Shared Memory Constant Kernel Parallel Convolution /////////////////////////////////
// The approach taken here is as follows
//    A1) Uses tiles so that we can fit the data into shared memory
//    A1) Uses constant memory for the kernel

		// Allocate input device memory
    checkCudaErrors(cudaMalloc((void **) &dPaddedInputData, paddedSize));
    // Allocate device memory for result
    checkCudaErrors(cudaMalloc((void **) &dParallelOutputData, paddedSize));
		checkCudaErrors(cudaMemcpy(dPaddedInputData,
                               hPaddedData,
                               paddedSize,
                               cudaMemcpyHostToDevice));

    // Allocate device memory for kernel
#if 0
    checkCudaErrors(cudaMalloc((void **) &dKernel, kernel_dim*kernel_dim * sizeof(int)));
		checkCudaErrors(cudaMemcpy(dKernel,
                               hKernel,
                               kernel_dim*kernel_dim * sizeof(int),
                               cudaMemcpyHostToDevice));
#endif

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer2 = NULL;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);

    // Execute the kernel
    /*dim3 dimBlockShared(TILE_SIZE, 1, 1);*/
    /*dim3 dimGridShared((width / TILE_SIZE)+1, (height / TILE_SIZE)+1, 1);*/
    cudaMemcpyToSymbol(kernel_gpu, hKernel, kernel_dim * kernel_dim * sizeof(float));

    parallelConvolutionSharedConstant<<<dimGridShared, dimBlockShared, 0>>>(dPaddedInputData, paddedWidth, paddedHeight, pad_size, dParallelOutputData);

    memset(hParallelOutputData,0,paddedSize);
    // Copy the reult back into host memory
    checkCudaErrors(cudaMemcpy(hParallelOutputData,
                               dParallelOutputData,
                               paddedSize,
                               cudaMemcpyDeviceToHost));

    testResult = compareData(hParallelOutputData,
                              hSerialDataOut,
                              paddedWidth*paddedHeight,
                              MAX_EPSILON_ERROR,
                              0.15f);

    // Check if kernel execution generated an error
    getLastCudaError("parallelConvolutionShared Kernel execution failed");
    checkCudaErrors(cudaFree(dPaddedInputData));
    checkCudaErrors(cudaFree(dParallelOutputData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer2);
    printf("Processing time Constant and Shared Memory: %f (ms)\n", sdkGetTimerValue(&timer2));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer2) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer2);
////////////////////////////// Shared Memory Constant Memory Parallel Convolution Complete /////////////////////////

////////////////////////////// Shared Memory Textured Kernel Parallel Convolution /////////////////////////////////
// The approach taken here is as follows
//    A1) Uses tiles so that we can fit the data into shared memory
//    A1) Uses textured memory for the kernel

		// Allocate input device memory
    checkCudaErrors(cudaMalloc((void **) &dPaddedInputData, paddedSize));
    // Allocate device memory for result
    checkCudaErrors(cudaMalloc((void **) &dParallelOutputData, paddedSize));
		checkCudaErrors(cudaMemcpy(dPaddedInputData,
                               hPaddedData,
                               paddedSize,
                               cudaMemcpyHostToDevice));

    // Allocate device memory for kernel
#if 0
    checkCudaErrors(cudaMalloc((void **) &dKernel, kernel_dim*kernel_dim * sizeof(int)));
		checkCudaErrors(cudaMemcpy(dKernel,
                               hKernel,
                               kernel_dim*kernel_dim * sizeof(int),
                               cudaMemcpyHostToDevice));
#endif

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer3 = NULL;
    sdkCreateTimer(&timer3);
    sdkStartTimer(&timer3);

    // Execute the kernel
#if 0
    cudaMemcpyToSymbol(kernel_gpu, hKernel, kernel_dim * kernel_dim * sizeof(int));
#else
// Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuKernel;
    checkCudaErrors(cudaMallocArray(&cuKernel,
                                    &channelDesc,
                                    kernel_dim,
                                    kernel_dim));
    checkCudaErrors(cudaMemcpyToArray(cuKernel,
                                      0,
                                      0,
                                      hKernel,
                                      kernel_dim * kernel_dim * sizeof(float),
                                      cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuKernel, channelDesc));

#endif
    parallelConvolutionSharedTextured<<<dimGridShared, dimBlockShared, 0>>>(dPaddedInputData, paddedWidth, paddedHeight, pad_size, dParallelOutputData);

    memset(hParallelOutputData,0,paddedSize);
    // Copy the reult back into host memory
    checkCudaErrors(cudaMemcpy(hParallelOutputData,
                               dParallelOutputData,
                               paddedSize,
                               cudaMemcpyDeviceToHost));

    testResult = compareData(hParallelOutputData,
                              hSerialDataOut,
                              paddedWidth*paddedHeight,
                              MAX_EPSILON_ERROR,
                              0.15f);

    // Check if kernel execution generated an error
    getLastCudaError("parallelConvolutionShared Textured Kernel execution failed");
    checkCudaErrors(cudaFree(dPaddedInputData));
    checkCudaErrors(cudaFree(dParallelOutputData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer3);
    printf("Processing time Constant and Textured Memory: %f (ms)\n", sdkGetTimerValue(&timer3));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer3) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer3);
////////////////////////////// Shared Memory Textured Kernel Parallel Convolution Complete ////////////////////////////////////////

    free(imagePath);
    free(hSerialDataOut);
    free(hKernel);
    free(hPaddedData);
    free(hParallelOutputData);
}

