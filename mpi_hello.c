#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define G 6.67428e-11

// Assumed scale: 100 pixels = 1AU.
#define AU  (149.6e6 * 1000)     // 149.6 million km, in meters.
#define SCALE  (250 / AU)
#define NUM_ROCKETS 1600
#define NUM_TYPES 6

typedef struct {
  char name[20];
  double mass;
  double px;
  double py;
  double vx;
  double vy;
}Body;

int main(int argc, char** argv) {
    // Initialize the MPI environment
    int message = 10;
#define MPI
#ifdef MPI
    MPI_Status status;
    Body planets[8];
    MPI_Datatype planettype;
    MPI_Datatype type[NUM_TYPES] = { MPI_CHAR, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    int blocklen[NUM_TYPES] = { 20, 1, 1, 1, 1, 1 };
    MPI_Aint array_of_displacements[NUM_TYPES];
    MPI_Aint name_addr, mass_addr, px_addr, py_addr, vx_addr, vy_addr;

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

    if (world_rank == 0)
    {
      /*while (message)*/
      {
        sprintf(planets[0].name, "%s", "earth");
        planets[0].mass = 5.9742 * pow(10,24);
        planets[0].px = -1*AU;
        planets[0].py = 0;
        planets[0].vx = 0;
        planets[0].vy = 29.783*1000;            // 29.783 km/sec
        MPI_Bcast(&planets[0], 1, planettype, 0, MPI_COMM_WORLD);
        printf("Sent planet info for '%s' to everyone\n", planets[0].name);
        /*message--;*/
      }
    }
    else
    {
      /*while (message != 1) */
      {
        MPI_Bcast(&planets[0], 1, planettype, 0, MPI_COMM_WORLD);
        printf("received planet name %s from Process 0\n",planets[0].name);
        printf("received planet mass %e from Process 0\n",planets[0].mass);
        printf("received planet px %e from Process 0\n",planets[0].px);
        printf("received planet py %e from Process 0\n",planets[0].py);
        printf("received planet vx %e from Process 0\n",planets[0].vx);
        printf("received planet vy %e from Process 0\n",planets[0].vy);
      }
    }
    printf("Process %d terminated\n",world_rank);
    // Finalize the MPI environment.
    MPI_Type_free(&planettype);
    MPI_Finalize();
#else

    Body rocket, sun, venus;
    Body *bodies;
    Body *d_bodies;


    bodies = (Body *)malloc(NUM_ROCKETS * sizeof(Body));

    int rockets;
    double vx=0,vr,vy=0;
    double vx_arr[NUM_ROCKETS];
    double vy_arr[NUM_ROCKETS];

    vr = 46 * 1000; // 46 km/sec

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
      printf("Body %d \t%f, \t%f, \t%f, \t%f\n",k, bodies[k].name, bodies[k].px/AU, bodies[k].py/AU, bodies[k].vx, bodies[k].vy);
    }
#endif
}
