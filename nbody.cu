// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
                                  int mpi_threadId)
{
  Force myForce;
  double dx, dy, d, f, theta;

  int myid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  /* Private to each thread */
  double totalFx = 0;
  double totalFy = 0;

  
  for (long i = 0; i < NUM_STEPS; i++)
  {
    /* Update planet position */
    for (int bodyIindex = 0; bodyIindex < num_m_bodies; bodyIindex++)
    {
      for (int planetid = 0; planetid < num_m_bodies; planetid++)
      {
        /* Do not calculate attraction to myself */
        if (planetid == bodyIindex)
          continue;
  
        dx = (m_bodies[bodyIindex].px-m_bodies[planetid].px);
        dy = (m_bodies[bodyIindex].py-m_bodies[planetid].py);
        d = sqrt(dx*dx + dy*dy);
        f = G * m_bodies[bodyIindex].mass * m_bodies[planetid].mass / (d*d);
    
        theta = atan2(dy, dx);
        myForce.fx = cos(theta) * f;
        myForce.fy = sin(theta) * f;
    
        totalFx += myForce.fx;
        totalFy += myForce.fy;
      }
    
      /* Use one thread to do the updates */
      m_bodies[bodyIindex].vx += totalFx / m_bodies[bodyIindex].mass * TIMESTEP;
      m_bodies[bodyIindex].vy += totalFy / m_bodies[bodyIindex].mass * TIMESTEP;
      m_bodies[bodyIindex].px += m_bodies[bodyIindex].vx * TIMESTEP;
      m_bodies[bodyIindex].py += m_bodies[bodyIindex].vy * TIMESTEP;
    } /* end planet updates */

    /* this handles launching rockets based on mpi_threadId */ 
    if (i > mpi_threadId)
    { 
      /* start body updates based on updated planet positions */
      for (int bodyIindex = 0; bodyIindex < num_m_bodies; bodyIindex++)
      {
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
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Serial nBody  on CPU
////////////////////////////////////////////////////////////////////////////////
void serialNbody(Body n_bodies[],
	         int num_n_bodies,
		 Body m_bodies[],
		 int num_m_bodies,
                 long start_step,
                 long stop_step)
{
  Force myForce;
  double Fx[NUM_ROCKETS], Fy[NUM_ROCKETS], dx, dy, d, f, theta;

  for (long i = start_step; i < stop_step; i++)
  {
    for (int bodyIindex = 0; bodyIindex < num_n_bodies; bodyIindex++)
    {
      //printf("Serial %s \t%f, \t%f, \t%f, \t%f\n",n_bodies[bodyIindex].name, n_bodies[bodyIindex].px/AU, n_bodies[bodyIindex].py/AU, n_bodies[bodyIindex].vx, n_bodies[bodyIindex].vy);
      Fx[bodyIindex] = 0;
      Fy[bodyIindex] = 0;
      for (int myid = 0; myid < num_m_bodies; myid++)
      {
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
  }
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
void nbody_cuda(Body *n_bodies, int num_n_bodies, Body *m_bodies, int num_m_bodies, int threadId)
{

#if (0)//def SERIAL
    long start_step = 0;
    long stop_step  = 8 * STEPS_PER_THREAD;
    clock_t start2 = clock();
    serialNbody(n_bodies, num_n_bodies, m_bodies, num_m_bodies, start_step, stop_step);
    clock_t end2 = clock() ;
    double elapsed_time = (end2-start2)/(double)CLOCKS_PER_SEC ;
    printf("Serial (%d Bodies) time = %f\n", NUM_ROCKETS, elapsed_time);
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

    int nBlocks = (num_n_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    clock_t start2 = clock();
    nBodyAcceleration<<<nBlocks, BLOCK_SIZE>>>(d_n_bodies, num_n_bodies, d_m_bodies, num_m_bodies, threadId);
    clock_t end2 = clock() ;
    double elapsed_time = (end2-start2)/(double)CLOCKS_PER_SEC ;
    printf("Cuda + MPI (%d Bodies) Thread %d time = %f\n", NUM_ROCKETS, threadId, elapsed_time);

/*    checkCudaErrors(cudaMemcpy(n_bodies,
                               d_n_bodies,
                               num_n_bodies * sizeof(Body),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_bodies,
                               d_m_bodies,
                               num_m_bodies * sizeof(Body),
                               cudaMemcpyDeviceToHost));

			       */
    checkCudaErrors(cudaFree(d_n_bodies));
    checkCudaErrors(cudaFree(d_m_bodies));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();
#endif
}
