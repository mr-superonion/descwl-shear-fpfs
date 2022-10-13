#!/usr/bin/env python
"""
simple example with ring test (rotating intrinsic galaxies)
"""
import os
import pickle
import schwimmbad
import numpy as np
from argparse import ArgumentParser
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog   # one of the galaxy catalog classes
from descwl_shear_sims.stars import StarCatalog                 # star catalog class
from descwl_shear_sims.psfs import make_ps_psf,make_fixed_psf   # for making a power spectrum PSF
from descwl_shear_sims.sim import get_se_dim                    # convert coadd dims to SE dims

rotate  =   False
dither  =   False
itest   =   0


def work(ifield=0):
    rng = np.random.RandomState(ifield)
    coadd_dim = 2000
    buff   = 50

    nrot= 4
    g1_list=[0.02,-0.02]
    # band_list=['r', 'i', 'z']
    band_list=['i']
    rot_list=[np.pi/nrot*i for i in range(nrot)]
    nshear=len(g1_list)

    if itest==0:
        # basic test
        args={
                'cosmic_rays':False,
                'bad_columns':False,
                'star_bleeds':False,
        }
        star_catalog=None
        psf = make_fixed_psf(psf_type='moffat')
    elif itest==1:
        # spatial varying PSF
        args={
                'cosmic_rays':False,
                'bad_columns':False,
                'star_bleeds':False,
        }
        star_catalog=None
        # this is the single epoch image sized used by the sim, we need
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    elif itest==2:
        # with star
        args={
                'cosmic_rays':False,
                'bad_columns':False,
                'star_bleeds':False,
        }
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=(ifield%1000)/10+1,
            layout='random_circle',
        )
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    elif itest==3:
        # with mask plane
        args={
                'cosmic_rays':True,
                'bad_columns':True,
                'star_bleeds':True,
        }
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=(ifield%1000)/10+1,
            layout='random_circle',
        )
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    else:
        raise ValueError('itest must be 0, 1 or 2 !!!')

    os.makedirs('outputs/test%d' %itest,exist_ok=True)

    # galaxy catalog; you can make your own
    galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout='random_circle',
    )

    for irot in range(nrot):
        for ishear in range(nshear):
            sim_data = make_sim(
                rng=rng,
                galaxy_catalog=galaxy_catalog,
                star_catalog=star_catalog,
                coadd_dim=coadd_dim,
                g1=g1_list[ishear],
                g2=0.00,
                psf=psf,
                dither=dither,
                rotate=rotate,
                bands=band_list,
                noise_factor=0.,
                theta0=rot_list[irot],
                **args
            )
            # this is only for fixed PSF..
            if irot==0 and ishear==0 and not os.path.isfile('outputs/PSF_test%d.pkl' %itest):
                psf_dim=sim_data['psf_dims'][0]
                se_wcs=sim_data['se_wcs'][0]
                with open('outputs/PSF_test%d.pkl' %itest, 'wb') as f:
                    pickle.dump({'psf':psf, 'psf_dim': psf_dim, 'wcs': se_wcs },f)

            for bb in band_list:
                sim_data['band_data'][bb][0].\
                    writeFits('outputs/test%d/field%04d_shear1-%d_rot%d_%s.fits' \
                    %(itest,ifield,ishear,irot,bb))
    return

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    for r in pool.map(work,list(range(100))):
        pass
    pool.close()
