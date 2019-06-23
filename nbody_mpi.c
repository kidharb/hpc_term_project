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
    Body earth, sun, mercury, venus, mars, jupiter, saturn, neptune, uranus;

    sun.mass = 1.98892 * pow(10,30);
    sun.px = 0;
    sun.py = 0;
    sun.vx = 0;
    sun.vy = 0;

    mercury.mass = 3.3 * pow(10,23);
    mercury.px = 0.39 * AU;
    mercury.py = 0;
    mercury.vx = 0;
    mercury.vy = 47.36 * 1000;

    earth.mass = 5.9742 * pow(10,24);
    earth.px = -1*AU;
    earth.py = 0;
    earth.vx = 0;
    earth.vy = 29.783*1000;            // 29.783 km/sec

    venus.mass = 4.8685 * pow(10,24);
    venus.px = 0.723 * AU;
    venus.py = 0;
    venus.vx = 0;
    venus.vy = -35.02 * 1000;

    mars.mass = 6.39 * pow(10,23);
    mars.px = 1.524 * AU;
    mars.py = 0;
    mars.vx = 0;
    mars.vy = 24.07 * 1000;

    jupiter.mass = 1.9 * pow(10,27);
    jupiter.px = 5.23 * AU;
    jupiter.py = 0;
    jupiter.vx = 0;
    jupiter.vy = 13.06 * 1000;

    saturn.mass = 5.69 * pow(10,26);
    saturn.px = 9.539 * AU;
    saturn.py = 0;
    saturn.vx = 0;
    saturn.vy = 9.68 * 1000;

    uranus.mass = 8,68 * pow(10,25);
    uranus.px = 19.18 * AU;
    uranus.py = 0;
    uranus.vx = 0;
    uranus.vy = -6.80 * 1000;

    neptune.mass = 1.02 * pow(10,26);
    neptune.px = 30.06 * AU;
    neptune.py = 0;
    neptune.vx = 0;
    neptune.vy = 5.43 * 1000;

    memcpy(&bodies[0], &sun, sizeof(Body));
    memcpy(&bodies[1], &mercury, sizeof(Body));
    memcpy(&bodies[2], &venus, sizeof(Body));
    memcpy(&bodies[3], &earth, sizeof(Body));
    memcpy(&bodies[4], &mars, sizeof(Body));
    memcpy(&bodies[5], &jupiter, sizeof(Body));
    memcpy(&bodies[6], &saturn, sizeof(Body));
    memcpy(&bodies[7], &uranus, sizeof(Body));
    memcpy(&bodies[8], &neptune, sizeof(Body));
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
