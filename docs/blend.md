# Basic Blending Tests
This page shows the tests with blended galaxies. Please visit
[this directory](https://github.com/mr-superonion/descwl-shear-fpfs/tree/main/tests/test2_lsst_blend).

## Image Simulations

First, run the following command to make image simulations:

```shell
xsubSmall mpirun -np 208 ./desc_simulate.py --mpi
```
Here xsubSmall is the shell script to submit job to PBS/Slurm. See
[here](https://github.com/mr-superonion/udots/blob/main/.xbin/xsub) as an
example. You can find desc_simulate.py [here](https://github.com/mr-superonion/descwl-shear-fpfs/blob/main/tests/test2_lsst_blend/desc_simulate.py)

## From Images to Catalogs

The next step is to measure galaxy shape, size and flux from the images:

```shell
xsubSmall mpirun -np 208 ./fpfs_run_descsim.py --config ./config_process1.ini --mpi
```
You can find fpfs_run_descsim.py
[here](https://github.com/mr-superonion/descwl-shear-fpfs/blob/main/tests/test2_lsst_blend/fpfs_run_descsim.py),
and config_process1.ini is the configure file under the same directory.


## Summary

Estimate shear and derive muultiplicative and additive biases:
```shell
xsubMini ./fpfs_summary_descsim.py --config ./config_process1.ini --ncores 52
```
The notebook showing the estimated biases can be found [here](./tests/test2_lsst_blend/1_2_process_basic.ipynb)
fpfs_summary_descsim.py can be found under the same directory.
