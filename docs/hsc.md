# HSC Tests

The tests shown here run on the simulation in [Li & Mandelbaum
(2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp..851L/abstract) to make
sure the updated algorithm can get consistent results on the preivous
simulation. Morever, we found a -0.5 \% multiplicative bias on the preivous
simulations, and the tests shown here is to find the reason for that bias.


## From Images to Catalogs

First we measure galaxy shape, size and flux from the noisy blended galaxy
images simulated in the [Li & Mandelbaum
(2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp..851L/abstract)

```shell
xsubSmall mpirun -np 208 ./fpfs_run_hsc.py --config ./config_process1.ini --minId 0 --maxId 4000 --mpi
```
You can find fpfs_run_descsim.py [here
](https://github.com/mr-superonion/descwl-shear-fpfs/blob/main/tests/test1_hsc_pre/fpfs_run_hsc.py),
and config_process1.ini is the configure file under the same directory.

## Summary

Estimate shear and derive multiplicative and additive biases:
```shell
xsubSmall mpirun -np 208 ./fpfs_summary_hsc.py --config ./config_process1.ini --minId 0 --maxId 4000 --mpi
```
Here fpfs_summary_hsc.py can be found under the same directory.

## Notebooks

The notebook showing the estimated biases for noiseless galaxies can be found
[here](./tests/test1_hsc_pre/make_plot.ipynb).
