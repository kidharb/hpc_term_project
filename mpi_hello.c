/*#include <mpi.h>*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define G 6.67428e-11

// Assumed scale: 100 pixels = 1AU.
#define AU  (149.6e6 * 1000)     // 149.6 million km, in meters.
#define SCALE  (250 / AU)

int main(int argc, char** argv) {
    // Initialize the MPI environment
    int message = 10;
#ifdef MPI
    MPI_Status status;

    MPI_Init(NULL, NULL);

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
      while (message)
      {
        MPI_Bcast(&message, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("Sent message %d to everyone\n", message);
        message--;
      }
    }
    else
    {
      while (message != 1) 
      {
        MPI_Bcast(&message, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("received message %d from Process 0\n",message);
      }
    }
    printf("Process %d terminated\n",world_rank);
    // Finalize the MPI environment.
    MPI_Finalize();
#else
    #define NUM_ROCKETS 1600
    typedef struct {
      char name[20];
      double mass;
      double px;
      double py;
      double vx;
      double vy;
    }Body;

    Body rocket, sun, venus;
    Body *bodies;
    Body *d_bodies;

    bodies = (Body *)malloc(NUM_ROCKETS * sizeof(Body));

    int rockets;
    double vx=0,vr,vy=0;
    double vx_arr[NUM_ROCKETS];
    double vy_arr[NUM_ROCKETS];

    vr = 46;

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
