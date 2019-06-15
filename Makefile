INC="./common/inc"
NVCCFLAGS=-I$(INC) -m64
OMPFLAG=-fopenmp
NVCC=nvcc
MPICC=mpic++
CCFLAGS=-g -c
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcuda

#cluster: nbody_mpi.o

all: nbody

nbody: nbody.o nbody_mpi.o
	$(MPICC) nbody.o nbody_mpi.o -o nbody $(LDFLAGS)

nbody.o: nbody.cu
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) nbody.cu -o nbody.o

nbody_mpi.o: nbody_mpi.c
	$(MPICC) $(CCFLAGS) nbody_mpi.c -o nbody_mpi.o

clean: nbody.o nbody nbody_mpi.o
	rm nbody nbody_mpi.o nbody.o
