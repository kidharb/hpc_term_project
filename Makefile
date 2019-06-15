INC="./common/inc"
NVCCFLAGS=-I$(INC) -m64
OMPFLAG=-fopenmp
NVCC=nvcc
MPICC=mpicc
CCFLAGS=-g -c
LDFLAGS=-L/usr/local/cuda/lib -lcudart


all: nbody

nbody: nbody.o nbody_mpi.o
	$(MPICC) $(LDFLAGS) nbody.o nbody_mpi.o -o nbody

nbody.o: nbody.cu
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) nbody.cu -o nbody.o

nbody_mpi.o: nbody_mpi.c
	$(MPICC) $(CCFLAGS) nbody_mpi.c -o nbody_mpi.o

clean:
	rm nbody nbody_mpi.o nbody.o
