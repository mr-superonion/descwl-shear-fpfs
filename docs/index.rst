.. FPFS documentation master file, created by
   sphinx-quickstart on Wed Apr 27 02:50:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

desc-wl-shear-fpfs
======================================

This project applies the FPFS shear estimator to the LSST weak-lensing shear
image simulation. We first test for the blended galaxy case. Then we add stars
masks and coadding process to the test.

Please refer the following papers when using the algorithms

`Li et. al (2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract>`_
`Li, Li & Massey (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.4850L/abstract>`_
`Li & Mandelbaum (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp..851L/abstract>`_

.. toctree::
    :maxdepth: 2

    Blend Test <blend.md>

    PSF Variation Test <psf.md>

    Star Test <star.md>

    Mask Test <mask.md>

    Coadd Test <coadd.md>
