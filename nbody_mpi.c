#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nbody.h"

void nbody_cuda(Body *, int, Body *, int, int, char*);

void nbody_init_rockets(Body *bodies)
{
    double vx=0,vr,vy=0;
    double vx_arr[NUM_ROCKETS];
    double vy_arr[NUM_ROCKETS];

    vr = 46 * 1000; //46 km/s

    for (int i = 0; i < NUM_ROCKETS; i+=4)
    {
      vx += vr / NUM_ROCKETS * 4;
      vy = sqrt((pow(vr,2)+0.0001) - pow(vx,2));
      vx_arr[i] = vx;
      vy_arr[i] = vy;

      vx_arr[1+i] = vx;
      vy_arr[1+i] = -vy;

      vx_arr[2+i] = -vx;
      vy_arr[2+i] = vy;

      vx_arr[3+i] = -vx;
      vy_arr[3+i] = -vy;
    }

    for (int k = 0; k < NUM_ROCKETS; k++)
    {
      //sprintf(bodies[k].name, "Body %d", k);
      bodies[k].mass = 1 * pow(10,3);
      bodies[k].px = -1 * AU + k;
      bodies[k].py = k;
      bodies[k].vx = vx_arr[k];
      bodies[k].vy = vy_arr[k];
      //printf("Body %d \t%f \t%f \t%f \t%f \t%f\n",k, bodies[k].name, bodies[k].px/AU, bodies[k].py/AU, bodies[k].vx, bodies[k].vy);
    }
}
void nbody_init_planets(Body *bodies)
{
    Body earth, sun, venus;

    //sprintf(sun.name, "%s", "sun");
    sun.mass = 1.98892 * pow(10,30);
    /*sun.mass = 1.98892 * pow(10,24);*/
    sun.px = 0;
    sun.py = 0;
    sun.vx = 0;
    sun.vy = 0;

    //sprintf(earth.name, "%s", "earth");
    earth.mass = 5.9742 * pow(10,24);
    earth.px = -1*AU;
    earth.py = 0;
    earth.vx = 0;
    earth.vy = 29.783*1000;            // 29.783 km/sec

    //sprintf(venus.name, "%s", "venus");
    venus.mass = 4.8685 * pow(10,24);
    venus.px = 0.723 * AU;
    venus.py = 0;
    venus.vx = 0;
    venus.vy = -35.02 * 1000;

    memcpy(&bodies[0], &sun, sizeof(Body));
    memcpy(&bodies[1], &earth, sizeof(Body));
    memcpy(&bodies[2], &venus, sizeof(Body));
}

int main(int argc, char** argv) {
    MPI_Status status;
    Body *planets;
    Body *rockets;
    Body planets_disp;
    char config[7]; // Serial or Parallel

    sprintf(config, "%s", argv[1]);
    /* Init MPI */
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Thread %d reporting out of %d\n", world_rank, world_size);
    {
      planets = (Body *)malloc(NUM_PLANETS * sizeof(Body));
      rockets = (Body *)malloc(NUM_ROCKETS * sizeof(Body));
      nbody_init_planets(planets);
      nbody_init_rockets(rockets);

      nbody_cuda(rockets, NUM_ROCKETS, planets, NUM_PLANETS, world_rank, config);

      free(planets);
      free(rockets);
    }
    // Finalize the MPI environment.
    MPI_Finalize();
}
