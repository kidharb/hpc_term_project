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
#define NUM_STEPS 360

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

__global__ void nBodyAcceleration( Body bodies[], 
                                  int timestep)
{
  Force myForce;

  int myid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  for (int bodyIindex = 0; bodyIindex < NUM_BODIES; bodyIindex++)
  {
    if (myid != bodyIindex)
    {
      double dx = (bodies[bodyIindex].px-bodies[myid].px);
      double dy = (bodies[bodyIindex].py-bodies[myid].py);
      double d = sqrt(dx*dx + dy*dy);
      double f = G * bodies[bodyIindex].mass * bodies[myid].mass / (d*d);
  
      double theta = atan2(dy, dx);
      myForce.fx = cos(theta) * f;
      myForce.fy = sin(theta) * f;
      /*printf("[%s : %s] partial fx, partial fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, myForce.fx, myForce.fy);*/
  
      atomicAdd(&totalFx, myForce.fx);
      atomicAdd(&totalFy, myForce.fy);
    }
    __syncthreads();
  
    if (tid == bodyIindex)
    {
      bodies[bodyIindex].vx += totalFx / bodies[bodyIindex].mass * timestep;
      bodies[bodyIindex].vy += totalFy / bodies[bodyIindex].mass * timestep;
      bodies[bodyIindex].px += bodies[bodyIindex].vx * timestep;
      bodies[bodyIindex].py += bodies[bodyIindex].vy * timestep;
      /*printf("Cuda [%s] Total fx, Total fy = [%e, %e]\n",bodies[bodyIindex].name, totalFx, totalFy);*/
      /*printf("Cuda [%s] Updated vx, Updated vy = [%e, %e]\n",bodies[bodyIindex].name, bodies[bodyIindex].vx, bodies[bodyIindex].vy);*/
      //printf("Cuda [%s] [%e, %e, %e, %e]\n",bodies[bodyIindex].name, bodies[bodyIindex].px/AU, bodies[bodyIindex].py/AU, bodies[bodyIindex].vx, bodies[bodyIindex].vy);
    }
  }
  /*printf("\n");*/
}

////////////////////////////////////////////////////////////////////////////////
//! Serial nBody  on CPU
////////////////////////////////////////////////////////////////////////////////
void serialNbody(Body bodies[],
                 int timestep)
{
  Force myForce;
  double Fx[NUM_BODIES], Fy[NUM_BODIES], dx, dy, d, f, theta;

  for (int bodyIindex = 0; bodyIindex < NUM_BODIES; bodyIindex++)
  {
    printf("Serial %s \t%f, \t%f, \t%f, \t%f\n",bodies[bodyIindex].name, bodies[bodyIindex].px/AU, bodies[bodyIindex].py/AU, bodies[bodyIindex].vx, bodies[bodyIindex].vy);
    Fx[bodyIindex] = 0;
    Fy[bodyIindex] = 0;
    for (int myid = 0; myid < NUM_BODIES; myid++)
    {
      /* Do not calculate attraction to myself */
      if (myid == bodyIindex)
        continue;

      dx = (bodies[myid].px-bodies[bodyIindex].px);
      dy = (bodies[myid].py-bodies[bodyIindex].py);
      d = sqrt(dx*dx + dy*dy);
      f = G * bodies[bodyIindex].mass * bodies[myid].mass / (d*d);
      
      theta = atan2(dy, dx);
      myForce.fx = cos(theta) * f;
      myForce.fy = sin(theta) * f;
      Fx[bodyIindex] += myForce.fx;
      Fy[bodyIindex] += myForce.fy;
      /*printf("[%s %s] partial fx, partial fy = [%e, %e]\n",bodies[myid].name, bodies[bodyIindex].name, Fx[bodyIindex], Fy[bodyIindex]);*/
    }
  }
  for (int bodyIindex = 0; bodyIindex < NUM_BODIES; bodyIindex++)
  {
    /*printf("[%s] Total fx, Total fy = [%e, %e]\n",bodies[bodyIindex].name, Fx[bodyIindex], Fy[bodyIindex]);*/
    bodies[bodyIindex].vx += Fx[bodyIindex] / bodies[bodyIindex].mass * timestep;
    bodies[bodyIindex].vy += Fy[bodyIindex] / bodies[bodyIindex].mass * timestep;
    bodies[bodyIindex].px += bodies[bodyIindex].vx * timestep;
    bodies[bodyIindex].py += bodies[bodyIindex].vy * timestep;
    /*printf("Serial [%s] Total Fx, Total Fy = [%e, %e]\n",bodies[bodyIindex].name, Fx, Fy);*/
    /*printf("Serial [%s] Updated vx, Updated vy = [%e, %e]\n",bodies[bodyIindex].name, bodies[bodyIindex].vx, bodies[bodyIindex].vy);*/
  }
  printf("\n");
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
    long timestep = 24*3600;

    bodies = (Body *)malloc(NUM_BODIES * sizeof(Body));

    sprintf(sun.name, "%s", "sun");
    sun.mass = 1.98892 * pow(10,30);
    /*sun.mass = 1.98892 * pow(10,24);*/
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
    for (int step = 1; step < NUM_STEPS; step++)
    {
      printf("Step #%d\n",step);
      nBodyAcceleration<<<dimGrid, dimBlock, 0>>>(d_bodies, timestep);
      serialNbody(bodies, timestep);
    }

    checkCudaErrors(cudaFree(d_bodies));
    free(bodies);
    //checkCudaErrors(cudaFree(d_sun));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    printf("completed, returned %s\n",
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

