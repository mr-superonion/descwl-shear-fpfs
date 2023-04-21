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
import impt
import time
import glob
import fitsio
import schwimmbad
import numpy as np
import pandas as pd
import jax.numpy as jnp
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
        do_noirev = cparser.getboolean("FPFS", "do_noirev")
        if do_noirev:
            ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
            self.cov_mat = fitsio.read(ncov_fname)
        else:
            self.cov_mat = np.ones((31, 31))
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        # This task change the cut on one observable and see how the biases
        # changes.
        # Here is  the observable used for test
        self.upper_mag = 26.
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)
        print(self.lower_m00)
        self.lower_r2 = 0.05
        # setup WL distortion parameter
        self.gver = gver
        return

    def prepare_functions(self):
        params = impt.fpfs.FpfsParams(
            Const=20,
            lower_m00=self.lower_m00,
            sigma_m00=2.0,
            lower_r2=self.lower_r2,
            upper_r2=200,
            lower_v=0.3,
            sigma_r2=4,
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
        gc.collect()
        return e1, enoise, res1, rnoise

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        assert os.path.isfile(
            in_nm
        ), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = impt.fpfs.read_catalog(in_nm)
        print("number of galaxies: %d" % len(mm))
        e1_sum = jnp.sum(e1.evaluate(mm))
        e1_sum = e1_sum - jnp.sum(enoise.evaluate(mm))

        # shear response
        r1_sum = jnp.sum(res1.evaluate(mm))
        r1_sum = r1_sum - jnp.sum(rnoise.evaluate(mm))
        del mm
        return e1_sum, r1_sum

    def run(self, field):
        start_time = time.time()
        out = np.zeros((4, 1))
        # names= [('cut','<f8'), ('de','<f8'), ('eA','<f8') ('res','<f8')]
        out[0, 0] = self.upper_mag

        for irot in range(2):
            e1, enoise, res1, rnoise = self.prepare_functions()
            in_nm1 = os.path.join(
                self.catdir,
                "src-%05d_%s-1_rot%d_i.fits" % (field, self.gver, irot),
            )
            sum_e1_1, sum_r1_1 = \
                self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
            gc.collect()
            print(sum_e1_1)

            in_nm2 = os.path.join(
                self.catdir,
                "src-%05d_%s-0_rot%d_i.fits" % (field, self.gver, irot),
            )
            sum_e1_2, sum_r1_2 = \
                self.get_sum_e_r(in_nm2, e1, enoise, res1, rnoise)
            gc.collect()
            dtime = time.time() - start_time
            print(
                "--- computational time: %.2f seconds ---" % dtime
            )

            out[1, 0] = out[1, 0] + sum_e1_2 - sum_e1_1
            out[2, 0] = out[2, 0] + (sum_e1_1 + sum_e1_2) / 2.0
            out[3, 0] = out[3, 0] + (sum_r1_1 + sum_r1_2) / 2.0
        return out


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

    cparser = ConfigParser()
    cparser.read(args.config)
    glist = []
    if cparser.getboolean("distortion", "test_g1"):
        glist.append("g1")
    if cparser.getboolean("distortion", "test_g2"):
        glist.append("g2")
    if len(glist) < 1:
        raise ValueError("Cannot test nothing!! Must test g1 or test g2. ")
    shear_value = cparser.getfloat("distortion", "shear_value")

    for gver in glist:
        print("Testing for %s . " % gver)
        worker = Worker(args.config, gver=gver)
        fname_list = glob.glob(os.path.join(worker.catdir, "*%s*" % gver))
        outs = []
        id_list = np.unique(
            [int(ff.split("src-")[1].split("_")[0]) for ff in fname_list]
        )
        for r in pool.map(worker.run, id_list):
            outs.append(r)
        outs = np.stack(outs)
        nsims = outs.shape[0]
        summary_dirname = worker.sum_dir
        os.makedirs(summary_dirname, exist_ok=True)
        pyfits.writeto(
            os.path.join(
                summary_dirname,
                "bin_%s.fits" % (worker.upper_mag),
            ),
            outs,
            overwrite=True,
        )

        res = np.average(outs, axis=0)
        err = np.std(outs, axis=0)
        mbias = (res[1] / res[3] / 2.0 - shear_value) / shear_value
        merr = (err[1] / res[3] / 2.0) / shear_value / np.sqrt(nsims)
        cbias = res[2] / res[3]
        cerr = err[2] / res[3] / np.sqrt(nsims)
        df = pd.DataFrame(
            {
                "binave": res[0],
                "mbias": mbias,
                "merr": merr,
                "cbias": cbias,
                "cerr": cerr,
            }
        )
        df.to_csv(
            os.path.join(
                summary_dirname,
                "%s_bin_%s.csv"
                % (worker.proc_name, worker.upper_mag),
            ),
            index=False,
        )

        print("Separate galaxies into %d bins: %s" % (len(res[0]), res[0]))
        print("Multiplicative biases for those bins are: ", mbias)
        print("Errors are: ", merr)
        print("Additive biases for those bins are: ", cbias)
        print("Errors are: ", cerr)
        del worker
    pool.close()
