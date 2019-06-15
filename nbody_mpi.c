#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nbody.h"

void nbody_cuda(Body *, int);

void nbody_init_rockets(Body *bodies)
{
    int rockets;
    double vx=0,vr,vy=0;
    double vx_arr[NUM_ROCKETS];
    double vy_arr[NUM_ROCKETS];

    vr = 46 * 1000; //46 km/s

    for (int i = 0; i < NUM_ROCKETS; i+=4)
    {
      vx += vr / NUM_ROCKETS * 4;
      vy = sqrt(pow(vr,2) - pow(vx,2) + 0.000000001);
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
      sprintf(bodies[k].name, "Body %d", k);
      bodies[k].mass = 1 * pow(10,3);
      bodies[k].px = -1 * AU + k;
      bodies[k].py = k;
      bodies[k].vx = vx_arr[k];
      bodies[k].vy = vy_arr[k];
      printf("Body %d \t%f \t%f \t%f \t%f \t%f\n",k, bodies[k].name, bodies[k].px/AU, bodies[k].py/AU, bodies[k].vx, bodies[k].vy);
    }
}
void nbody_init_planets(Body *bodies)
{
    Body earth, sun, venus;

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
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    int step = 0;

    MPI_Status status;
    Body *bodies;
    Body planets;
    MPI_Datatype planettype;
    MPI_Datatype type[NUM_TYPES] = { MPI_CHAR, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    int blocklen[NUM_TYPES] = { 20, 1, 1, 1, 1, 1 };
    MPI_Aint array_of_displacements[NUM_TYPES];
    MPI_Aint name_addr, mass_addr, px_addr, py_addr, vx_addr, vy_addr;


    /* Init MPI */
    MPI_Init(NULL, NULL);

    /* Create our user defined data structure for the planets to make communication easier, later */
    MPI_Get_address(&planets.name, &name_addr);
    MPI_Get_address(&planets.mass, &mass_addr);
    MPI_Get_address(&planets.px, &px_addr);
    MPI_Get_address(&planets.py, &py_addr);
    MPI_Get_address(&planets.vx, &vx_addr);
    MPI_Get_address(&planets.vy, &vy_addr);

    array_of_displacements[0] = 0;
    array_of_displacements[1] = mass_addr - name_addr;
    array_of_displacements[2] = px_addr - name_addr;
    array_of_displacements[3] = py_addr - name_addr;
    array_of_displacements[4] = vx_addr - name_addr;
    array_of_displacements[5] = vy_addr - name_addr;

    MPI_Type_create_struct(NUM_TYPES, blocklen, array_of_displacements, type, &planettype);
    MPI_Type_commit(&planettype);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Thread %d reporting\n",world_rank);
    if (world_rank == 0)
    {
      /* setup planets */
      /* Allocate memory for the planets */
      bodies = (Body *)malloc(NUM_BODIES * sizeof(Body));
      nbody_init_planets(bodies);
      while (step++ < NUM_STEPS)
      {
	/* Update planets positions */
    	nbody_cuda(bodies, step);
        MPI_Bcast(bodies, 3, planettype, 0, MPI_COMM_WORLD);
        /*message--;*/
      }
    }
    else
    {
      bodies = (Body *)malloc(NUM_BODIES * sizeof(Body));
      nbody_init_rockets(bodies);
      while (step++ < NUM_STEPS)
      {
        MPI_Bcast(bodies, 3, planettype, 0, MPI_COMM_WORLD);
        printf("MPI[%d] %s \t%f, \t%f, \t%f, \t%f\n",world_rank, bodies[1].name, bodies[1].px/AU, bodies[1].py/AU, bodies[1].vx, bodies[1].vy);
      }
    }
    // Finalize the MPI environment.
    free(bodies);
    MPI_Type_free(&planettype);
    MPI_Finalize();
}
