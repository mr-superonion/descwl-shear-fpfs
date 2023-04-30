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

band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
}

nstd_map = {
    "g": 0.315,
    "r": 0.371,
    "i": 0.595,
    "z": 1.155,
    "*": 0.2186,
}


def get_seed_from_fname(fname, band):
    fid = int(fname.split("image-")[-1].split("_")[0]) + 212
    rid = int(fname.split("rot")[1][0])
    bid = band_map[band]
    return (fid * 2 + rid) * 4 + bid


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
        self.noi_ratio = cparser.getfloat("survey", "noise_ratio")
        self.ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        band = cparser.get("survey", "band")
        self.band = band
        self.nstd_f = nstd_map[band] * self.noi_ratio
        if not os.path.isfile(self.ncov_fname) and self.noi_ratio > 1e-10:
            ngrid = 2 * self.rcut
            self.noise_pow = np.ones((ngrid, ngrid)) * self.nstd_f**2.0 * ngrid**2.0
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
        psf_array2 = np.pad(psf_array, (npad + 1, npad), mode="constant")[
            beg:end, beg:end
        ]
        del npad
        # pad to exposure size
        npad = (ngrid2 - psf_array.shape[0]) // 2
        psf_array3 = np.pad(psf_array, (npad + 1, npad), mode="constant")
        return psf_array2, psf_array3

    def make_noise_psf(self, fname):
        exposure = afwimage.ExposureF.readFits(fname)
        wcs = exposure.getInfo().getWcs()
        self.scale = wcs.getPixelScale().asArcseconds()
        with open(self.psf_fname, "rb") as f:
            psf_dict = pickle.load(f)
        psf_dm = make_dm_psf(**psf_dict)
        exposure.setPsf(psf_dm)
        # zero_flux = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
        # magz = np.log10(zero_flux) * 2.5
        self.image_nx = exposure.getWidth()
        self.psf_array2, self.psf_array3 = self.prepare_psf(
            exposure, self.rcut, self.image_nx
        )
        # FPFS Tasks
        # noise cov task
        if self.noise_pow is not None:
            noise_task = fpfs.image.measure_noise_cov(
                self.psf_array2,
                sigma_arcsec=self.sigma_as,
                pix_scale=self.scale,
                sigma_detect=self.sigma_det,
            )
            cov_elem = noise_task.measure(self.noise_pow)
            pyfits.writeto(self.ncov_fname, cov_elem, overwrite=True)
        return

    def run(self, fname):
        out_fname = os.path.join(self.catdir, fname.split("/")[-1])
        out_fname = out_fname.replace("image-", "src-").replace("_g.fits", ".fits")
        if os.path.exists(out_fname):
            print("Already has measurement for this simulation. ")
            return

        self.make_noise_psf(fname)

        if not os.path.isfile(fname):
            print("Cannot find input galaxy file: %s" % fname)
            return
        if self.band != "*":
            blist = [self.band]
        else:
            blist = ["g", "r", "i", "z"]

        gal_array = np.zeros((self.image_nx, self.image_nx))
        weight_all = 0.0
        for band in blist:
            print("processing %s band" % band)
            noi_std = nstd_map[band] * self.noi_ratio
            weight = 1.0 / nstd_map[band] ** 2.0
            weight_all = weight_all + weight
            fname2 = fname.replace("_g.fits", "_%s.fits" % band)
            exposure = afwimage.ExposureF.readFits(fname2)
            mi = exposure.getMaskedImage()
            # mskDat  =   mi.getMask().getArray()
            # varDat  =   mi.getVariance().getArray()
            im = mi.getImage()
            gal_array = gal_array + im.getArray() * weight

            if noi_std > 1e-15:
                # noise
                seed = get_seed_from_fname(fname, band)
                rng = np.random.RandomState(seed)
                print("Using noisy setup with std: %.2f" % noi_std)
                print("The random seed is %d" % seed)
                gal_array = (
                    gal_array
                    + rng.normal(
                        scale=noi_std,
                        size=gal_array.shape,
                    )
                    * weight
                )
            else:
                print("Using noiseless setup")
        gal_array = gal_array / weight_all

        # measurement task
        meas_task = fpfs.image.measure_source(
            self.psf_array2,
            sigma_arcsec=self.sigma_as,
            pix_scale=self.scale,
            sigma_detect=self.sigma_det,
        )
        print(
            "The upper limit of Fourier wave number is %s pixels"
            % (meas_task.klim_pix,)
        )

        if self.nstd_f > 1e-10:
            thres = 35.0 * self.nstd_f * self.scale**2.0  # approx 10 sigma
            thres2 = -1.5 * self.nstd_f * self.scale**2.0  # approx 0.5 sigma
        else:
            magz = 30.0
            cutmag = 26.5
            thres = 10 ** ((magz - cutmag) / 2.5) * self.scale**2.0
            thres2 = -0.05
        coords = fpfs.image.detect_sources(
            gal_array,
            self.psf_array3,
            gsigma=meas_task.sigmaF_det,
            thres=thres,
            thres2=thres2,
            klim=meas_task.klim,
        )
        print("pre-selected number of sources: %d" % len(coords))

        img_list = [
            gal_array[
                cc["fpfs_y"] - self.rcut : cc["fpfs_y"] + self.rcut,
                cc["fpfs_x"] - self.rcut : cc["fpfs_x"] + self.rcut,
            ]
            for cc in coords
        ]
        out = meas_task.measure(img_list)
        out = rfn.merge_arrays([coords, out], flatten=True, usemask=False)
        pyfits.writeto(out_fname, out)
        return

    def __call__(self, fname):
        print("start image file: %s" % (fname))
        self.run(fname)
        print("finish %s" % (fname))
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
    fname_list = glob.glob(os.path.join(worker.imgdir, "image-*_g.fits"))
    for r in pool.map(worker, fname_list):
        pass
    pool.close()
