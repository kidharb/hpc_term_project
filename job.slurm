#!/bin/bash
# specify a partition
#SBATCH -p ha
# specify number of nodes
#SBATCH -N 4
# specify number of cores
#SBATCH -n 8
# specify memory pool for all cores
##SBATCH --mem 1
# specify the wall clock time limit for the job
#SBATCH -t 00:5:00
# specify the job name
#SBATCH -J kids-hpc-job
# specify the filename to be used for writing output
# NOTE: You must replace the username `hwang' with your own account name!!
#SBATCH -o /home-mscluster/kbachan/slurm_%N.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/kbachan/slurm_%N.err

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR
#make clean
#make
#mpicc -o mpi_mat_vect_mult mpi_mat_vect_mult.c
mpiexec -n 2 ./nbody
