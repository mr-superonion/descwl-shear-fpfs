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


def get_seed_from_fname(fname):
    fid = int(fname.split('image-')[-1].split('_')[0])+212
    rid = int(fname.split('rot')[1][0])
    return fid*2+rid


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.imgdir = cparser.get("procsim", "img_dir")
        self.catdir = cparser.get("procsim", "cat_dir")
        self.psf_fname = cparser.get("procsim", "psf_fname")
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.sigma_det = cparser.getfloat("FPFS", "sigma_det")
        self.rcut = cparser.getint("FPFS", "rcut")

        if not os.path.isdir(self.imgdir):
            print(self.imgdir)
            raise FileNotFoundError("Cannot find input images directory!")
        print("The input directory for galaxy images is %s. " % self.imgdir)

        if not os.path.isdir(self.catdir):
            os.makedirs(self.catdir, exist_ok=True)
        print("The output directory for shear catalogs is %s. " % self.catdir)
        # setup survey parameters
        self.noi_std = cparser.getfloat("survey", "noise")
        self.ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        if not os.path.isfile(self.ncov_fname) and self.noi_std > 1e-20:
            ngrid = 2 * self.rcut
            self.noise_pow = np.ones((ngrid, ngrid))*self.noi_std**2.*ngrid**2.
        else:
            self.noise_pow = None
        return

    def prepare_psf(self, exposure, rcut, ngrid2):
        # pad to (64, 64) and then cut off
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
        # pad to exposure size
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

        # FPFS Tasks
        # noise cov task
        if self.noise_pow is not None:
            noise_task = fpfs.image.measure_noise_cov(
                psf_array2,
                sigma_arcsec=self.sigma_as,
                pix_scale=scale,
                sigma_detect=self.sigma_det,
            )
            cov_elem = noise_task.measure(self.noise_pow)
            pyfits.writeto(self.ncov_fname, cov_elem, overwrite=True)

        # measurement task
        meas_task = fpfs.image.measure_source(
            psf_array2,
            sigma_arcsec=self.sigma_as,
            pix_scale=scale,
            sigma_detect=self.sigma_det,
        )
        print(
            "The upper limit of Fourier wave number is %s pixels" % (
                meas_task.klim_pix,
            )
        )
        out_fname = os.path.join(self.catdir, fname.split("/")[-1])
        out_fname = out_fname.replace('image-', 'src-')
        if os.path.exists(out_fname):
            print("Already has measurement for this simulation. ")
            return
        if self.noi_std > 1e-20:
            # noise
            seed = get_seed_from_fname(fname)
            rng = np.random.RandomState(seed)
            print("Using noisy setup with std: %.2f" % self.noi_std)
            print("The random seed is %d" % seed)
            gal_array = gal_array + rng.normal(
                scale=self.noi_std,
                size=gal_array.shape,
            )
        else:
            print("Using noiseless setup")

        cutmag = 26.5
        thres = 10 ** ((magz - cutmag) / 2.5) * scale**2.0
        thres2 = -0.05
        coords = fpfs.image.detect_sources(
            gal_array,
            psf_array3,
            gsigma=meas_task.sigmaF_det,
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
    band = "i"
    fname_list = glob.glob(
        os.path.join(worker.imgdir, "image-*_%s.fits" % band)
    )
    for r in pool.map(worker, fname_list):
        pass
    pool.close()
