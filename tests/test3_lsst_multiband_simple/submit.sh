#!/usr/bin/env bash

echo $1
for ii in {0..9}; do
    xsubSmall mpirun -np 208 ./impt_summary.py --config ./config_n1_process1.ini --mpi --runid $ii --magcut $1 && sleep 0.5;
done

