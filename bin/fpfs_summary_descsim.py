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
import glob
import fpfs
import schwimmbad
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser

msig = 2.0
rsig = 4.0
psig = 1.5
mcut = 25.0
rcut = 0.03
# pcut = 0.005
pcut = 0.3
rcut_upp = 2.0


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
            cov_mat = pyfits.getdata(ncov_fname)
            self.nn = fpfs.catalog.imptcov_to_fpfscov(cov_mat)
        else:
            self.nn = None

        self.selnm = []
        self.cutsig = []
        self.cut = []
        # detection cut
        self.do_detcut = cparser.getboolean("FPFS", "do_detcut")
        if self.do_detcut:
            self.selnm.append("detect")
            self.cutsig.append(psig)
            self.cut.append(pcut)
        # magnitude cut
        self.do_magcut = cparser.getboolean("FPFS", "do_magcut")
        if self.do_magcut:
            self.selnm.append("M00")
            self.cutsig.append(msig)
            self.cut.append(10 ** ((self.magz - mcut) / 2.5))
        # resolution cut
        self.do_magcut = cparser.getboolean("FPFS", "do_rcut")
        if self.do_magcut:
            self.selnm.append("R2")
            self.cutsig.append(rsig)
            self.cut.append(rcut)
        assert len(self.selnm) >= 1, "Must do at least one selection."
        self.selnm = np.array(self.selnm)
        self.cutsig = np.array(self.cutsig)
        self.cut = np.array(self.cut)

        # We change the cut on one observable and see difference in bias
        # Here is  the observable used for test
        self.test_name = cparser.get("FPFS", "test_name")
        assert self.test_name in self.selnm
        self.test_ind = np.where(self.selnm == self.test_name)[0]
        self.cutB = cparser.getfloat("FPFS", "cutB")
        self.dcut = cparser.getfloat("FPFS", "dcut")
        self.ncut = cparser.getint("FPFS", "ncut")

        if not os.path.exists(self.catdir):
            raise FileNotFoundError("Cannot find input directory: %s!" % self.catdir)
        print("The input directory for galaxy catalogs is %s. " % self.catdir)
        # setup WL distortion parameter
        self.gver = gver
        self.Const = cparser.getfloat("FPFS", "weighting_c")
        return

    def run(self, field):
        # names= [('cut','<f8'), ('de','<f8'), ('eA1','<f8'), ('eA2','<f8'),
        # ('res1','<f8'), ('res2','<f8')]
        out = np.zeros((6, self.ncut))
        for irot in range(2):
            in_nm1 = os.path.join(
                self.catdir,
                "src-%05d_%s-1_rot%d.fits" % (field, self.gver, irot),
            )
            in_nm2 = os.path.join(
                self.catdir,
                "src-%05d_%s-0_rot%d.fits" % (field, self.gver, irot),
            )
            assert os.path.isfile(in_nm1) & os.path.isfile(in_nm2), (
                "Cannot find input galaxy shear catalog distorted by"
                "positive and negative shear: %s , %s" % (in_nm1, in_nm2)
            )
            mm1 = pyfits.getdata(in_nm1)
            mm2 = pyfits.getdata(in_nm2)
            ells1 = fpfs.catalog.fpfs_m2e(
                mm1,
                const=self.Const,
                nn=self.nn,
            )
            ells2 = fpfs.catalog.fpfs_m2e(
                mm2,
                const=self.Const,
                nn=self.nn,
            )

            fs1 = fpfs.catalog.summary_stats(
                mm1,
                ells1,
                use_sig=False,
            )
            fs2 = fpfs.catalog.summary_stats(
                mm2,
                ells2,
                use_sig=False,
            )

            for i in range(self.ncut):
                fs1.clear_outcomes()
                fs2.clear_outcomes()
                icut = self.cutB + self.dcut * i
                if self.test_name == "M00":
                    self.cut[self.test_ind] = 10 ** ((self.magz - icut) / 2.5)
                else:
                    self.cut[self.test_ind] = icut
                fs1.update_selection_weight(self.selnm, self.cut, self.cutsig)
                fs2.update_selection_weight(self.selnm, self.cut, self.cutsig)
                fs1.update_selection_bias(self.selnm, self.cut, self.cutsig)
                fs2.update_selection_bias(self.selnm, self.cut, self.cutsig)
                fs1.update_ellsum()
                fs2.update_ellsum()
                out[0, i] = icut
                out[1, i] = (
                    out[1, i] + (fs2.sumE1 + fs2.corE1) - (fs1.sumE1 + fs1.corE1)
                )
                out[2, i] = out[2, i] + (fs1.sumE1 + fs2.sumE1) / 2.0
                out[3, i] = (
                    out[3, i] + (fs1.sumE1 + fs2.sumE1 + fs1.corE1 + fs2.corE1) / 2.0
                )
                out[4, i] = out[4, i] + (fs1.sumR1 + fs2.sumR1) / 2.0
                out[5, i] = (
                    out[5, i] + (fs1.sumR1 + fs2.sumR1 + fs1.corR1 + fs2.corR1) / 2.0
                )
        return out

    def __call__(self, field):
        print("start ID: %d" % (field))
        return self.run(field)


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
        # id_list = [0, 2, 3,0, 2, 3,  4, 5]
        for r in pool.map(worker, id_list):
            outs.append(r)
        outs = np.stack(outs)
        nsims = outs.shape[0]
        summary_dirname = worker.sum_dir
        os.makedirs(summary_dirname, exist_ok=True)
        pyfits.writeto(
            os.path.join(
                summary_dirname,
                "bin_%s.fits" % (worker.test_name),
            ),
            outs,
            overwrite=True,
        )

        res = np.average(outs, axis=0)
        err = np.std(outs, axis=0)
        mbias = (res[1] / res[5] / 2.0 - shear_value) / shear_value
        merr = (err[1] / res[5] / 2.0) / shear_value / np.sqrt(nsims)
        cbias = res[3] / res[5]
        cerr = err[3] / res[5] / np.sqrt(nsims)
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
                "%s_bin_%s.csv" % (worker.proc_name, worker.test_name),
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
