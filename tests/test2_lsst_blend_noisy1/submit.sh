for ii in {0..9}; do
    xsubSmall mpirun -np 208 ./impt_summary.py --config ./config_n2_process1.ini --mpi --runid $ii && sleep 0.5;
done
