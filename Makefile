INC="./common/inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: assignment2

assignment2: assignment2.cu
	$(NVCC) $(NVCCFLAGS) assignment2.cu -o assignment2

clean:
	rm assignment2
