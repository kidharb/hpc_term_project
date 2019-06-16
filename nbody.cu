// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nbody.h"

// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

////////////////////////////////////////////////////////////////////////////////
// Constants

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Parallel convolution on GPU using shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void nBodyAcceleration(Body bodies[], 
				  int num_planets,
				  Body rockets[],
				  int num_rockets,
                                  int step)
{
  Force myForce;
  double dx, dy, d, f, theta;

  int myid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  /* Private to each thread */
  double totalFx = 0;
  double totalFy = 0;

  if (tid == 0)
  {
    printf("\nStep #%d\n",step);
  }
  for (int bodyIindex = 0; bodyIindex < num_planets; bodyIindex++)
  {
    if (tid == 0)
    {
      printf("Cuda %s \t%f, \t%f, \t%f, \t%f\n",bodies[bodyIindex].name, bodies[bodyIindex].px/AU, bodies[bodyIindex].py/AU, bodies[bodyIindex].vx, bodies[bodyIindex].vy);
    }

    /* Do not calculate attraction to myself */
    if (myid == bodyIindex)
      continue;

    dx = (bodies[bodyIindex].px-rockets[myid].px);
    dy = (bodies[bodyIindex].py-rockets[myid].py);
    d = sqrt(dx*dx + dy*dy);
    f = G * bodies[bodyIindex].mass * rockets[myid].mass / (d*d);
  
    theta = atan2(dy, dx);
    myForce.fx = cos(theta) * f;
    myForce.fy = sin(theta) * f;
    /*printf("[%s : %s] partial fx, partial fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, myForce.fx, myForce.fy);*/
  
    /*atomicAdd(&totalFx, myForce.fx);*/
    /*atomicAdd(&totalFy, myForce.fy);*/
    totalFx += myForce.fx;
    totalFy += myForce.fy;
    /*printf("[%s %s] Total fx, Total fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, totalFx, totalFy);*/
  }
  /*__syncthreads();*/
  
  /* Use one thread to do the updates */
  bodies[tid].vx += totalFx / bodies[tid].mass * TIMESTEP;
  bodies[tid].vy += totalFy / bodies[tid].mass * TIMESTEP;
  bodies[tid].px += bodies[tid].vx * TIMESTEP;
  bodies[tid].py += bodies[tid].vy * TIMESTEP;
}

////////////////////////////////////////////////////////////////////////////////
//! Serial nBody  on CPU
////////////////////////////////////////////////////////////////////////////////
#ifdef MAC
void serialNbody(Body bodies[],
                 int step)
{
  Force myForce;
  double Fx[NUM_BODIES], Fy[NUM_BODIES], dx, dy, d, f, theta;

  printf("Step #%d\n",step);
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
    bodies[bodyIindex].vx += Fx[bodyIindex] / bodies[bodyIindex].mass * TIMESTEP;
    bodies[bodyIindex].vy += Fy[bodyIindex] / bodies[bodyIindex].mass * TIMESTEP;
    bodies[bodyIindex].px += bodies[bodyIindex].vx * TIMESTEP;
    bodies[bodyIindex].py += bodies[bodyIindex].vy * TIMESTEP;
    /*printf("Serial [%s] Total Fx, Total Fy = [%e, %e]\n",bodies[bodyIindex].name, Fx, Fy);*/
    /*printf("Serial [%s] Updated vx, Updated vy = [%e, %e]\n",bodies[bodyIindex].name, bodies[bodyIindex].vx, bodies[bodyIindex].vy);*/
  }
  printf("\n");
}
#endif


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
void nbody_cuda(Body *planets, int step)
{
    Body *d_bodies;

    checkCudaErrors(cudaMalloc((void **) &d_bodies, NUM_BODIES * sizeof(Body)));
    checkCudaErrors(cudaMemcpy(d_bodies,
                               planets,
                               NUM_BODIES * sizeof(Body),
                               cudaMemcpyHostToDevice));

    // Allocate output device memory
    dim3 dimBlock(NUM_BODIES, 1, 1);
    dim3 dimGrid(1, 1, 1);

    nBodyAcceleration<<<dimGrid, dimBlock, 0>>>(d_bodies, NUM_BODIES, d_bodies, NUM_ROCKETS, step);
      /*serialNbody(bodies, step);*/
    checkCudaErrors(cudaMemcpy(planets,
                               d_bodies,
                               NUM_BODIES * sizeof(Body),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_bodies));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();
}

