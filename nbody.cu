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
__global__ void nBodyAcceleration(Body n_bodies[], 
				  int num_n_bodies,
				  Body m_bodies[],
				  int num_m_bodies,
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
    //printf("\nStep #%d\n",step);
  }
  for (int bodyIindex = 0; bodyIindex < num_m_bodies; bodyIindex++)
  {
    if (tid == 0)
    {
      //printf("Cuda %s \t%f, \t%f, \t%f, \t%f\n",n_bodies[bodyIindex].name, n_bodies[bodyIindex].px/AU, n_bodies[bodyIindex].py/AU, n_bodies[bodyIindex].vx, n_bodies[bodyIindex].vy);
    }

    /* Do not calculate attraction to myself */
    if ((myid == bodyIindex) && (num_n_bodies == num_m_bodies))
      continue;

    dx = (n_bodies[bodyIindex].px-m_bodies[myid].px);
    dy = (n_bodies[bodyIindex].py-m_bodies[myid].py);
    d = sqrt(dx*dx + dy*dy);
    f = G * n_bodies[bodyIindex].mass * m_bodies[myid].mass / (d*d);
  
    theta = atan2(dy, dx);
    myForce.fx = cos(theta) * f;
    myForce.fy = sin(theta) * f;
    /*printf("[%s : %s] partial fx, partial fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, myForce.fx, myForce.fy);*/
  
    totalFx += myForce.fx;
    totalFy += myForce.fy;
    /*printf("[%s %s] Total fx, Total fy = [%e, %e]\n",bodies[bodyIindex].name, bodies[myid].name, totalFx, totalFy);*/
  }
  
  /* Use one thread to do the updates */
  n_bodies[tid].vx += totalFx / n_bodies[tid].mass * TIMESTEP;
  n_bodies[tid].vy += totalFy / n_bodies[tid].mass * TIMESTEP;
  n_bodies[tid].px += n_bodies[tid].vx * TIMESTEP;
  n_bodies[tid].py += n_bodies[tid].vy * TIMESTEP;
}

////////////////////////////////////////////////////////////////////////////////
//! Serial nBody  on CPU
////////////////////////////////////////////////////////////////////////////////
#if (1)//def MAC
void serialNbody(Body n_bodies[],
	         int num_n_bodies,
		 Body m_bodies[],
		 int num_m_bodies,
                 int step)
{
  Force myForce;
  double Fx[NUM_ROCKETS], Fy[NUM_ROCKETS], dx, dy, d, f, theta;

  printf("Step #%d\n",step);
  for (int bodyIindex = 0; bodyIindex < num_n_bodies; bodyIindex++)
  {
    printf("Serial %s \t%f, \t%f, \t%f, \t%f\n",n_bodies[bodyIindex].name, n_bodies[bodyIindex].px/AU, n_bodies[bodyIindex].py/AU, n_bodies[bodyIindex].vx, n_bodies[bodyIindex].vy);
    Fx[bodyIindex] = 0;
    Fy[bodyIindex] = 0;
    for (int myid = 0; myid < num_m_bodies; myid++)
    {
      /* Do not calculate attraction to myself */
      if ((myid == bodyIindex) && (num_n_bodies == num_m_bodies))
        continue;

      dx = (n_bodies[myid].px-m_bodies[bodyIindex].px);
      dy = (n_bodies[myid].py-m_bodies[bodyIindex].py);
      d = sqrt(dx*dx + dy*dy);
      f = G * n_bodies[bodyIindex].mass * m_bodies[myid].mass / (d*d);
      
      theta = atan2(dy, dx);
      myForce.fx = cos(theta) * f;
      myForce.fy = sin(theta) * f;
      Fx[bodyIindex] += myForce.fx;
      Fy[bodyIindex] += myForce.fy;
      //printf("[%s %s] partial fx, partial fy = [%e, %e]\n",n_bodies[myid].name, n_bodies[bodyIindex].name, Fx[bodyIindex], Fy[bodyIindex]);
    }
  }
  for (int bodyIindex = 0; bodyIindex < num_n_bodies; bodyIindex++)
  {
    /*printf("[%s] Total fx, Total fy = [%e, %e]\n",bodies[bodyIindex].name, Fx[bodyIindex], Fy[bodyIindex]);*/
    n_bodies[bodyIindex].vx += Fx[bodyIindex] / n_bodies[bodyIindex].mass * TIMESTEP;
    n_bodies[bodyIindex].vy += Fy[bodyIindex] / n_bodies[bodyIindex].mass * TIMESTEP;
    n_bodies[bodyIindex].px += n_bodies[bodyIindex].vx * TIMESTEP;
    n_bodies[bodyIindex].py += n_bodies[bodyIindex].vy * TIMESTEP;
    /*printf("Serial [%s] Total Fx, Total Fy = [%e, %e]\n",bodies[bodyIindex].name, Fx, Fy);*/
    //printf("Serial [%s] Updated vx, Updated vy = [%e, %e]\n",bodies[bodyIindex].name, bodies[bodyIindex].vx, bodies[bodyIindex].vy);
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
void nbody_cuda(Body *n_bodies, int num_n_bodies, Body *m_bodies, int num_m_bodies, int step)
{

#if (1)//def SERIAL
    serialNbody(n_bodies, num_n_bodies, m_bodies, num_m_bodies, step);
#else
    Body *d_n_bodies;
    Body *d_m_bodies;

    checkCudaErrors(cudaMalloc((void **) &d_n_bodies, num_n_bodies * sizeof(Body)));
    checkCudaErrors(cudaMalloc((void **) &d_m_bodies, num_m_bodies * sizeof(Body)));
    checkCudaErrors(cudaMemcpy(d_n_bodies,
                               n_bodies,
                               num_n_bodies * sizeof(Body),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_m_bodies,
                               m_bodies,
                               num_m_bodies * sizeof(Body),
                               cudaMemcpyHostToDevice));

    dim3 dimBlock(num_n_bodies, 1, 1);
    dim3 dimGrid(1, 1, 1);

    nBodyAcceleration<<<dimGrid, dimBlock, 0>>>(d_n_bodies, num_n_bodies, d_m_bodies, num_m_bodies, step);
    checkCudaErrors(cudaMemcpy(n_bodies,
                               d_n_bodies,
                               num_n_bodies * sizeof(Body),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_bodies,
                               d_m_bodies,
                               num_m_bodies * sizeof(Body),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_n_bodies));
    checkCudaErrors(cudaFree(d_m_bodies));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();
#endif
}
