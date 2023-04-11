#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20221013 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
import os
import fpfs
import glob
import pickle
import schwimmbad
import numpy as np
import lsst.afw.image as afwimage
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser
import numpy.lib.recfunctions as rfn
from descwl_shear_sims.psfs import make_dm_psf
import lsst.geom as lsstgeom


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.imgdir = cparser.get("procsim", "img_dir")
        self.catdir = cparser.get("procsim", "cat_dir")
        self.psf_fname = cparser.get("procsim", "psf_fname")
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.rcut = cparser.getint("FPFS", "rcut")

        if not os.path.isdir(self.imgdir):
            print(self.imgdir)
            raise FileNotFoundError("Cannot find input images directory!")
        print("The input directory for galaxy images is %s. " % self.imgdir)

        if not os.path.isdir(self.catdir):
            os.makedirs(self.catdir, exist_ok=True)
        print("The output directory for shear catalogs is %s. " % self.catdir)

        # setup survey parameters
        self.noi_var = cparser.getfloat("survey", "noi_var")
        if self.noi_var > 1e-20:
            self.noidir = os.path.join(self.imgdir, "noise")
            if not os.path.exists(self.noidir):
                raise FileNotFoundError("Cannot find input noise directory!")
            self.noiPfname = cparser.get("survey", "noiP_filename")
            print("The input directory for noise images is %s. " % self.noidir)
        else:
            self.noidir = None
            self.noiPfname = None

        return

    def prepare_psf(self, exposure, rcut, ngrid2):
        ngrid = 64
        beg = ngrid // 2 - rcut
        end = beg + 2 * rcut
        bbox = exposure.getBBox()
        bcent = bbox.getCenter()
        psf_model = exposure.getPsf()
        psf_array = psf_model.computeImage(lsstgeom.Point2I(bcent)).getArray()
        npad = (ngrid - psf_array.shape[0]) // 2
        psf_array2 = np.pad(
            psf_array,
            (npad + 1, npad),
            mode="constant"
        )[beg:end, beg:end]
        del npad
        npad = (ngrid2 - psf_array.shape[0]) // 2
        psf_array3 = np.pad(psf_array, (npad + 1, npad), mode="constant")
        return psf_array2, psf_array3

    def run(self, fname):
        print("running for galaxy image: %s" % (fname))
        if not os.path.isfile(fname):
            print("Cannot find input galaxy file: %s" % fname)
            return

        exposure = afwimage.ExposureF.readFits(fname)
        with open(self.psf_fname, "rb") as f:
            psf_dict = pickle.load(f)
        psf_dm = make_dm_psf(**psf_dict)

        exposure.setPsf(psf_dm)

        if not exposure.hasPsf():
            print("exposure doesnot have PSF")
        mi = exposure.getMaskedImage()
        im = mi.getImage()
        gal_array = im.getArray()
        # mskDat  =   mi.getMask().getArray()
        # varDat  =   mi.getVariance().getArray()
        wcs = exposure.getInfo().getWcs()
        scale = wcs.getPixelScale().asArcseconds()
        zero_flux = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
        magz = np.log10(zero_flux) * 2.5

        image_ny, image_nx = gal_array.shape
        psf_array2, psf_array3 = self.prepare_psf(
            exposure,
            self.rcut,
            image_nx
        )

        # FPFS Task
        if self.noi_var > 1e-20:
            # noise
            print("Using noisy setup with variance: %.3f" % self.noi_var)
            meas_task = fpfs.image.measure_source(
                psf_array2,
                sigma_arcsec=self.sigma_as,
                pix_scale=scale,
            )
            noise_array = 0.0
        else:
            print("Using noiseless setup")
            # by default noiFit=None
            meas_task = fpfs.image.measure_source(
                psf_array2,
                sigma_arcsec=self.sigma_as,
                pix_scale=scale,
            )
            noise_array = 0.0
        print(
            "The upper limit of Fourier wave number is %s pixels" % (
                meas_task.klim_pix,
            )
        )
        gal_array = gal_array + noise_array
        out_fname = os.path.join(self.catdir, fname.split("/")[-1])
        if os.path.exists(out_fname):
            print("Already has measurement for this simulation. ")
            return

        cutmag = 25.5
        thres = 10 ** ((magz - cutmag) / 2.5) * scale**2.0
        thres2 = -0.05
        print(thres, thres2)
        coords = fpfs.image.detect_sources(
            gal_array,
            psf_array3,
            gsigma=meas_task.sigmaF,
            thres=thres,
            thres2=thres2,
            klim=meas_task.klim,
        )
        print("pre-selected number of sources: %d" % len(coords))

        img_list = [
            gal_array[
                cc["fpfs_y"] - self.rcut: cc["fpfs_y"] + self.rcut,
                cc["fpfs_x"] - self.rcut: cc["fpfs_x"] + self.rcut,
            ]
            for cc in coords
        ]
        out = meas_task.measure(img_list)
        out = rfn.merge_arrays([coords, out], flatten=True, usemask=False)
        pyfits.writeto(out_fname, out)
        print("finish %s" % (fname))
        return

    def __call__(self, fname):
        print("start image file: %s" % (fname))
        self.run(fname)
        # try:
        #     self.run(fname)
        # except:
        #     print("Error on file: ", fname)
        return


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    worker = Worker(args.config)
    fname_list = glob.glob(os.path.join(worker.imgdir, "*"))[8000:]
    for r in pool.map(worker, fname_list):
        pass
    pool.close()
