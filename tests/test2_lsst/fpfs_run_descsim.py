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
import fpfs
import glob
import pickle
import schwimmbad
import numpy as np
import lsst.afw.image as afwImg
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser
import numpy.lib.recfunctions as rfn
from descwl_shear_sims.psfs import make_dm_psf
import lsst.geom as lsstGeom

_DefaultImgSize=6400



class Worker(object):
    def __init__(self,config_name):
        cparser     =   ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.imgdir =   cparser.get('procsim', 'img_dir')
        self.catdir =   cparser.get('procsim', 'cat_dir')
        self.simname=   cparser.get('procsim', 'sim_name')
        proc_name   =   cparser.get('procsim', 'proc_name')
        self.sigma_as=  cparser.getfloat('FPFS', 'sigma_as')
        self.rcut   =   cparser.getint('FPFS', 'rcut')
        self.indir  =   os.path.join(self.imgdir,self.simname)

        if not os.path.exists(self.indir):
            raise FileNotFoundError('Cannot find input images directory!')
        print('The input directory for galaxy images is %s. ' %self.indir)

        self.outdir=os.path.join(self.catdir,self.simname,proc_name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir,exist_ok=True)
        print('The output directory for shear catalogs is %s. ' %self.outdir)

        # setup survey parameters
        self.noi_var=cparser.getfloat('survey','noi_var')
        if self.noi_var>1e-20:
            self.noidir=os.path.join(self.imgdir,'noise')
            if not os.path.exists(self.noidir):
                raise FileNotFoundError('Cannot find input noise directory!')
            self.noiPfname=cparser.get('survey', 'noiP_filename')
            print('The input directory for noise images is %s. ' %self.noidir)
        else:
            self.noidir= None
            self.noiPfname=None

        return

    def prepare_PSF(self,exposure,rcut,ngrid2):
        ngrid       =   64
        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut
        bbox        =   exposure.getBBox()
        bcent       =   bbox.getCenter()
        psfExp      =   exposure.getPsf()
        psfData     =   psfExp.computeImage(lsstGeom.Point2I(bcent)).getArray()
        npad        =   (ngrid-psfData.shape[0])//2
        psfData2    =   np.pad(psfData,(npad+1,npad),mode='constant')[beg:end,beg:end]
        del npad
        npad        =   (ngrid2-psfData.shape[0])//2
        psfData3    =   np.pad(psfData,(npad+1,npad),mode='constant')
        return psfData2,psfData3

    def run(self,fname):
        print('running for galaxy image: %s' %(fname))
        if not os.path.isfile(fname):
            print('Cannot find input galaxy file: %s' %fname)
            return

        exposure    =   afwImg.ExposureF.readFits(fname)
        with open('outputs/PSF_test0.pkl', 'rb') as f:
             psf_dict= pickle.load(f)
        psf_dm=     make_dm_psf(**psf_dict)

        exposure.setPsf(psf_dm)

        if not exposure.hasPsf():
            print("exposure doesnot have PSF")
        mi      =   exposure.getMaskedImage()
        im      =   mi.getImage()
        galData =   im.getArray()
        # mskDat  =   mi.getMask().getArray()
        # varDat  =   mi.getVariance().getArray()
        wcs     =   exposure.getInfo().getWcs()
        scale   =   wcs.getPixelScale().asArcseconds()
        zero_flux=  exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()

        magz    =   np.log10(zero_flux)*2.5

        image_ny,image_nx= galData.shape

        nbegin  =   (_DefaultImgSize-image_nx)//2
        nend    =   nbegin+image_nx

        nbegin  =   (_DefaultImgSize-image_nx)//2
        nend    =   nbegin+image_nx
        psfData2,psfData3=self.prepare_PSF(exposure,self.rcut,image_nx)

        # FPFS Task
        if self.noi_var>1e-20:
            # noise
            print('Using noisy setup with variance: %.3f' %self.noi_var)
            assert self.noidir is not None
            noiFname    =   os.path.join(self.noidir,'noi%04d.fits' %Id)
            if not os.path.isfile(noiFname):
                print('Cannot find input noise file: %s' %noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData     =   pyfits.getdata(noiFname)[nbegin:nend,nbegin:nend]*10.*np.sqrt(self.noi_var)
            # Also times 100 for the noivar model
            powIn       =   np.load(self.noiPfname,allow_pickle=True).item()['%s'%self.rcut]*self.noi_var*100
            powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
            powModel[0] =   powIn
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=self.sigma_as,noiFit=powModel[0])
        else:
            print('Using noiseless setup')
            # by default noiFit=None
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=self.sigma_as)
            noiData     =   0.
        print('The upper limit of Fourier wave number is %s pixels' %(measTask.klim_pix))

        galData     =   galData+noiData
        outFname    =   os.path.join(self.outdir,fname.split('/')[-1])
        if  os.path.exists(outFname):
            print('Already has measurement for this simulation. ')
            return
        if  self.sigma_as<0.5:
            cutmag  =   25.5
        else:
            cutmag  =   26.0
        thres       =   10**((magz-cutmag)/2.5)*scale**2.
        thres2      =   -thres/20.
        coords      =   fpfs.image.detect_sources(galData,psfData3,gsigma=measTask.sigmaF,\
                        thres=thres,thres2=thres2,klim=measTask.klim)

        print('pre-selected number of sources: %d' %len(coords))
        imgList =   [galData[cc['fpfs_y']-self.rcut:cc['fpfs_y']+self.rcut,\
                    cc['fpfs_x']-self.rcut:cc['fpfs_x']+self.rcut] for cc in coords]
        out     =   measTask.measure(imgList)
        out     =   rfn.merge_arrays([coords,out],flatten=True,usemask=False)
        pyfits.writeto(outFname,out)
        del imgList,out,coords,galData,outFname
        gc.collect()
        print('finish %s' %(fname))
        return

    def __call__(self,fname):
        print('start image file: %s' %(fname))
        return self.run(fname)

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument('--config', required=True,type=str,
                        help='configure file name')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    worker  =   Worker(args.config)
    fname_list= glob.glob(os.path.join(worker.indir,'*'))
    print(fname_list)
    for r in pool.map(worker,fname_list[0:1]):
        pass
    pool.close()
