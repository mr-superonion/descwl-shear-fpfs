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
import impt
import time
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
        self.do_noirev = cparser.getboolean("FPFS", "do_noirev")
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
        # Here is  the observable used for test
        self.upper_mag = 26.0
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)
        # setup WL distortion parameter
        self.gver = gver
        return

    def prepare_functions(self):
        params = impt.fpfs.FpfsParams(
            Const=20.0,
            lower_m00=self.lower_m00,
            lower_r2=0.03,
            upper_r2=100.0,
            lower_v=0.3,
            sigma_m00=4.0,
            sigma_r2=4.0,
            sigma_v=1.5,
        )
        funcnm = "ss2"
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

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        assert os.path.isfile(
            in_nm
        ), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = impt.fpfs.read_catalog(in_nm)
        nobjs = len(mm)
        print("number of galaxies: %d" % nobjs)
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
        gc.collect()
        del mm
        return e1_sum, r1_sum

    def process(self, field):
        print("start ID: %d" % field)
        start_time = time.time()
        out = np.zeros((4, 1))
        # names= [('cut','<f8'), ('de','<f8'), ('eA','<f8') ('res','<f8')]
        out[0, 0] = self.upper_mag

        e1, enoise, res1, rnoise = self.prepare_functions()
        for irot in range(2):
            in_nm1 = os.path.join(
                self.catdir,
                "src-%05d_%s-1_rot%d_i.fits" % (field, self.gver, irot),
            )
            sum_e1_1, sum_r1_1 = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
            in_nm2 = os.path.join(
                self.catdir,
                "src-%05d_%s-0_rot%d_i.fits" % (field, self.gver, irot),
            )
            sum_e1_2, sum_r1_2 = self.get_sum_e_r(in_nm2, e1, enoise, res1, rnoise)
            dtime = time.time() - start_time
            print("--- computational time: %.2f seconds ---" % dtime)
            gc.collect()

            out[1, 0] = out[1, 0] + sum_e1_2 - sum_e1_1
            out[2, 0] = out[2, 0] + (sum_e1_1 + sum_e1_2) / 2.0
            out[3, 0] = out[3, 0] + (sum_r1_1 + sum_r1_2) / 2.0
        del e1, enoise, res1, rnoise
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
    ofname = os.path.join(
        summary_dirname,
        "%s_bin_%s_run%d.fits" % (worker.proc_name, worker.upper_mag, args.runid),
    )
    pyfits.writeto(
        ofname,
        outs,
        overwrite=True,
    )
    del worker
