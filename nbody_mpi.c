#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nbody.h"

void nbody_cuda(Body *);

void nbody_init_data(Body *bodies)
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
    Body planets[8];
    MPI_Datatype planettype;
    MPI_Datatype type[NUM_TYPES] = { MPI_CHAR, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    int blocklen[NUM_TYPES] = { 20, 1, 1, 1, 1, 1 };
    MPI_Aint array_of_displacements[NUM_TYPES];
    MPI_Aint name_addr, mass_addr, px_addr, py_addr, vx_addr, vy_addr;

    /* setup planets */
    Body *bodies;
    /* Allocate memory for the planets */
    bodies = (Body *)malloc(NUM_BODIES * sizeof(Body));
    nbody_init_data(bodies);

    /* Init MPI */
    MPI_Init(NULL, NULL);

    /* Create our user defined data structure for the planets to make communication easier, later */
    MPI_Get_address(&planets[0].name, &name_addr);
    MPI_Get_address(&planets[0].mass, &mass_addr);
    MPI_Get_address(&planets[0].px, &px_addr);
    MPI_Get_address(&planets[0].py, &py_addr);
    MPI_Get_address(&planets[0].vx, &vx_addr);
    MPI_Get_address(&planets[0].vy, &vy_addr);

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
      while (step++ < NUM_STEPS)
      {
        sprintf(planets[0].name, "%s", "earth");
        planets[0].mass = 5.9742 * pow(10,24);
        planets[0].px = -1*AU;
        planets[0].py = 0;
        planets[0].vx = 0;
        planets[0].vy = 29.783*1000;            // 29.783 km/sec
        MPI_Bcast(&planets[0], 1, planettype, 0, MPI_COMM_WORLD);
        printf("Step #%d Sent planet info for '%s' to everyone\n", step, planets[0].name);
        /*message--;*/
      }
    }
    else
    {
      while (step++ < NUM_STEPS)
      {
        MPI_Bcast(&planets[0], 1, planettype, 0, MPI_COMM_WORLD);
        printf("Step #%d received planet name %s from Process 0\n",step, planets[0].name);
        printf("Step #%d received planet mass %e from Process 0\n",step, planets[0].mass);
        printf("Step #%d received planet px %e from Process 0\n",step, planets[0].px);
        printf("Step #%d received planet py %e from Process 0\n",step, planets[0].py);
        printf("Step #%d received planet vx %e from Process 0\n",step, planets[0].vx);
        printf("Step #%d received planet vy %e from Process 0\n",step, planets[0].vy);
      }
    }
    // Finalize the MPI environment.
    MPI_Type_free(&planettype);
    MPI_Finalize();
    printf("Nbody Cuda call\n");
    nbody_cuda(bodies);
}
