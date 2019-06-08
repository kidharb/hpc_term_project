INC="./common/inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: nbody

nbody: nbody.cu
	$(NVCC) $(NVCCFLAGS) nbody.cu -o nbody

clean:
	rm nbody
