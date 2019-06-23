# hpc_term_project

Build Steps
1. make clean
2. make all

Run steps
1. ./run_ll.sh (to run the parallel job on the MSL cluster)
2. ./run_ser.sh (to run the serial on 1 node of the MSL cluster)
3. ./run_all.sh to run both the serial and parallel implementation on the cluster 

Debug output
1. The debug output file will be placed in the current directory

Simulation
1. The simulation is setup for 100'000 time steps, 30'000 rockets per timestep, 9 bodies (8 planets + sun)
2. The parallel implementation will run on 8 nodes, the serial implementation will be run on 1 node
3. With the config above, the parallel job should complete in 510 seconds, the serial job, should take 12824 seconds
4. To change the paramets edit nbody.h parameters NUM_STEPS and NUM_ROCKETS, make clean and make all
