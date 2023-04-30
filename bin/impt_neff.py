#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20220312 Xiangchong Li.
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
import gc
import jax
import time
import impt
import fitsio
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits

from argparse import ArgumentParser
from configparser import ConfigParser

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ[
    "XLA_FLAGS"
] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"


class Worker(object):
    def __init__(self, config_name, gver="g1"):
        cparser = ConfigParser()
        cparser.read(config_name)
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")

        # setup processor
        self.catdir = cparser.get("procsim", "cat_dir")
        self.sum_dir = cparser.get("procsim", "sum_dir")
        self.proc_name = cparser.get("procsim", "proc_name")
        self.do_noirev = True  # cparser.getboolean("FPFS", "do_noirev")
        if self.do_noirev:
            ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
            self.cov_mat = fitsio.read(ncov_fname)
        else:
            self.cov_mat = np.zeros((31, 31))
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        # This task change the cut on one observable and see how the biases
        # changes.
        # setup WL distortion parameter
        self.gver = gver
        return

    def prepare_functions(self, cut_mag):
        lower_m00 = 10 ** ((self.magz - cut_mag) / 2.5)
        params = impt.fpfs.FpfsParams(
            Const=20.0,
            lower_m00=lower_m00,
            lower_r2=0.1,
            upper_r2=10,
            lower_v=0.3,
            sigma_m00=4.0,
            sigma_r2=4.0,
            sigma_v=1.5,
        )
        funcnm = "ss2"
        # funcnm = "ts2"
        # ellipticity
        # e1_impt = impt.fpfs.FpfsE1(params, func_name=funcnm)
        # w_det = impt.fpfs.FpfsWeightDetect(params, func_name=funcnm)
        # w_sel = impt.fpfs.FpfsWeightSelect(params, func_name=funcnm)
        # e1 = e1_impt * w_sel * w_det

        e1 = impt.fpfs.FpfsWeightE1(params, func_name=funcnm)
        enoise = impt.BiasNoise(e1, self.cov_mat)
        res1 = impt.RespG1(e1)
        rnoise = impt.BiasNoise(res1, self.cov_mat)
        return e1, enoise, res1, rnoise

    def process(self, field):
        ncuts = 6
        start_time = time.time()
        out = np.zeros((3, ncuts))
        in_nm = os.path.join(
            self.catdir,
            "src-%05d_%s-1_rot0_i.fits" % (field, self.gver),
        )
        mm = impt.fpfs.read_catalog(in_nm)

        for im in range(ncuts):
            cut_mag = 26.5 - 0.5 * im
            e1, enoise, res1, rnoise = self.prepare_functions(cut_mag)
            # noise bias
            e1_sum = jax.lax.reduce(
                e1.evaluate(mm),
                0.0,
                jax.lax.add,
                dimensions=[0],
            )
            r1_sum = jax.lax.reduce(
                res1.evaluate(mm),
                0.0,
                jax.lax.add,
                dimensions=[0],
            )
            if self.do_noirev:
                e1_corr = jax.lax.reduce(
                    enoise.evaluate(mm),
                    0.0,
                    jax.lax.add,
                    dimensions=[0],
                )
                r1_corr = jax.lax.reduce(
                    rnoise.evaluate(mm),
                    0.0,
                    jax.lax.add,
                    dimensions=[0],
                )
                e1_sum = e1_sum - e1_corr
                r1_sum = r1_sum - r1_corr

            out[0, im] = cut_mag
            out[1, im] = e1_sum
            out[2, im] = r1_sum
            del e1, enoise, res1, rnoise
        dtime = time.time() - start_time
        print("--- computational time: %.2f seconds ---" % dtime)
        gc.collect()
        return out

    def run(self, field):
        try:
            return self.process(field)
        except Exception:
            print("failed on field: %d" % field)
            return None


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--runid",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
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

    cparser = ConfigParser()
    cparser.read(args.config)
    gver = "g1"
    shear_value = cparser.getfloat("distortion", "shear_value")

    print("Testing for %s . " % gver)

    worker = Worker(args.config, gver=gver)
    summary_dirname = worker.sum_dir
    os.makedirs(summary_dirname, exist_ok=True)

    id_list = np.arange(args.runid * 500, (args.runid + 1) * 500)
    # id_list = id_list[300:301]
    outs = []

    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        for res in pool.map(worker.run, id_list):
            if res is not None:
                outs.append(res)
    outs = np.stack(outs)
    if worker.do_noirev:
        ofname = os.path.join(
            summary_dirname,
            "%s_bin_neff_run%d.fits" % (worker.proc_name, args.runid),
        )
    else:
        ofname = os.path.join(
            summary_dirname,
            "%s_bin_neff_run%d_nnrev.fits" % (worker.proc_name, args.runid),
        )
    pyfits.writeto(
        ofname,
        outs,
        overwrite=True,
    )
    del worker
