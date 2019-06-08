#!/bin/bash
#./assignment2 |& tee -a assignment2_out.txt
nvprof ./assignment2 5 0 lena_bw.pgm
nvprof ./assignment2 5 0 U2plane.pgm
nvprof ./assignment2 10 0 lena_bw.pgm
nvprof ./assignment2 10 0 U2plane.pgm
nvprof ./assignment2 15 0 lena_bw.pgm
nvprof ./assignment2 15 0 U2plane.pgm
