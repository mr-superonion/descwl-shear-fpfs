# Blend Tests
This page shows the tests with blended galaxies. Please move to [this
directory](../tests/test2_lsst_blend).

## Image Simulations

First, run the following command to make image simulations.

```shell
xsubSmall mpirun -np 208 ./desc_simulate.py --mpi
```

## From Images to Catalogs

Then measure galaxy shape, size and flux from the images.


## Summary

Estimate shear and derive muultiplicative and additive biases.
