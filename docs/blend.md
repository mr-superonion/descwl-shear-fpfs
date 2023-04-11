# Blend Tests
This page shows the tests with blended galaxies. Please move to [this
directory](tests/test2_lsst_blend).

## Image Simulations

First, run the following command to make image simulations:

```shell
xsubSmall mpirun -np 208 ./desc_simulate.py --mpi
```

## From Images to Catalogs

Then measure galaxy shape, size and flux from the images:

```shell
xsubSmall mpirun -np 208 ./fpfs_run_descsim.py --config ./config_process1.ini --mpi
```

## Summary

Estimate shear and derive muultiplicative and additive biases:
```shell
xsubMini ./fpfs_summary_descsim.py --config ./config_process1.ini --ncores 52
```
The notebook showing the estimated biases can be found [here](./tests/test2_lsst_blend/1_2_process_basic.ipynb)
