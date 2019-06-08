// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
using namespace std;

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f
#define TILE_SIZE 64 // 64 * 64 * 4 = 16384 < 49152bytes
#define MAX_SHARED_MEMORY_BYTES 16384

#define G 6.67428e-11

// Assumed scale: 100 pixels = 1AU.
#define AU  (149.6e6 * 1000)     // 149.6 million km, in meters.
#define SCALE  (250 / AU)
#define NUM_BODIES 3

typedef struct {
    char name[20];
    double mass;
    double px;
    double py;
    double vx;
    double vy;
}Body;

typedef struct {
    char name[20];
    double fx;
    double fy;
}Force;

////////////////////////////////////////////////////////////////////////////////
// Constants

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using shared memory
////////////////////////////////////////////////////////////////////////////////
__device__ float totalFx = 0;
__device__ float totalFy = 0;

__global__ void nBodyAcceleration(int bodyIindex, Body bodies[])
{
  /*printf("Body I [%e %e %e %e %e]\n", BodyI->mass, BodyI->px, BodyI->py, BodyI->vx, BodyI->vy);
  printf("Body J [%e %e %e %e %e]\n", BodyJ->mass, BodyJ->px, BodyJ->py, BodyJ->vx, BodyJ->vy);*/
  Force myForce;

  int myid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  if (myid != bodyIindex)
  {
  float dx = (bodies[bodyIindex].px-bodies[myid].px);
  float dy = (bodies[bodyIindex].py-bodies[myid].py);
  float d = sqrt(dx*dx + dy*dy);
  float f = G * bodies[bodyIindex].mass * bodies[myid].mass / (d*d);

  float theta = atan2(dy, dx);
  myForce.fx = cos(theta) * f;
  myForce.fy = sin(theta) * f;
  printf("[%s : %s] partial fx, partial fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, myForce.fx, myForce.fy);
  atomicAdd(&totalFx, myForce.fx);
  atomicAdd(&totalFy, myForce.fy);
  /*totalFx += myForce.fx;*/
  /*totalFy += myForce.fy;*/
  __syncthreads();
  printf("[%s] total fx, total fy = [%e, %e]\n",bodies[bodyIindex].name, totalFx, totalFy);
  }
}
__global__ void parallelConvolutionShared(float *inputData,
                                                int width,
                                                int height,
                                                float* kernel,
                                                int pad_size,
                                                float *outputData)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if ((row > width) || (col > height))
    return;
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
    /*__syncthreads();*/

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
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);
void serialTransformCPU(float *inputData,
                                int width,
                                int height,
                                int *outputData);



////////////////////////////////////////////////////////////////////////////////
//! Compare results
////////////////////////////////////////////////////////////////////////////////
int compareResults(float * serialData, float * parallelData, unsigned long size)
{
  for (unsigned long i = 0; i < size; i++)
  {
    if (*(serialData + i) != *(parallelData + i))
    {
      return 0;
    }
  }
  return 1;
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("Starting...\n");

    // Declare an object of class geeks

    Body earth, sun, venus;
    Body *bodies;

    Body *d_bodies;
    int d_body = 1;
    long timestep = 24*3600;

    bodies = (Body *)malloc(NUM_BODIES * sizeof(Body));

    sprintf(sun.name, "%s", "sun");
    /*sun.mass = 1.98892 * pow(10,30);*/
    sun.mass = 1.98892 * pow(10,24);
    sun.px = 0;
    sun.py = 0;
    sun.vx = 0;
    sun.vy = 0;

    sprintf(earth.name, "%s", "earth");
    earth.mass = 5.9742 * pow(10,24);
    earth.px = -1*AU;
    earth.py = 0;
    earth.vx = 0;
    earth.vy = 29.783*1000;            // 29.783 km/sec

    sprintf(venus.name, "%s", "venus");
    venus.mass = 4.8685 * pow(10,24);
    venus.px = 0.723 * AU;
    venus.py = 0;
    venus.vx = 0;
    venus.vy = -35.02 * 1000;

    memcpy(&bodies[0], &sun, sizeof(Body));
    memcpy(&bodies[1], &earth, sizeof(Body));
    memcpy(&bodies[2], &venus, sizeof(Body));

    int devID = findCudaDevice(argc, (const char **) argv);

		// Allocate input device memory
    checkCudaErrors(cudaMalloc((void **) &d_bodies, NUM_BODIES * sizeof(Body)));
		checkCudaErrors(cudaMemcpy(d_bodies,
                               &bodies[0],
                               NUM_BODIES * sizeof(Body),
                               cudaMemcpyHostToDevice));

    // Allocate output device memory
    dim3 dimBlock(NUM_BODIES, 1, 1);
    dim3 dimGrid(1, 1, 1);
    nBodyAcceleration<<<dimGrid, dimBlock, 0>>>(d_body, d_bodies);

    //checkCudaErrors(cudaFree(d_earth));
    //checkCudaErrors(cudaFree(d_sun));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    printf("completed, returned %s\n",
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

