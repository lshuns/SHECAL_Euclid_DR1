# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-01-13 16:03:13

### Extract useful VIS info from DpdVisAnalysisResults

import os 
import glob
import json

import numpy as np
import pandas as pd
from astropy.time import Time

## >>>>>>>>>>>>>>>>> I/O

## Where to find the downloaded DpdVisAnalysisResults 
indir = '/somewhere/DpdVisAnalysisResults_DR1_R1_F006_D0E1'
outpath = './outputs/VIS_info_DR1_R1_F006_D0E1.csv'

## >>>>>>>>>>>>>>>>> Workhorse

## Find all files for one long exposure
inpath_list = glob.glob(os.path.join(indir, 
                                     'EUC_VIS_QC-SCI-*-00-1-0000000__*.json'))
Nfiles = len(inpath_list)
print(">>> total number of long exposures", Nfiles)

## Loop over and get useful info
saa_expo = np.zeros(Nfiles)
ra_expo = np.zeros(Nfiles)
dec_expo = np.zeros(Nfiles)
cr_expo = np.zeros(Nfiles)
bkg_expo = np.zeros(Nfiles)
zp_expo = np.zeros(Nfiles)
stars_expo = np.zeros(Nfiles)
ghosts_expo = np.zeros(Nfiles)

exptime_expo = np.zeros(Nfiles)
obstime_expo = np.zeros(Nfiles)

dithobs_expo = np.zeros(Nfiles)
expnum_expo = np.zeros(Nfiles)
obsid_expo = np.zeros(Nfiles)
patchid_expo = np.zeros(Nfiles)

CALBLKID = np.zeros(Nfiles).astype(object)
OBSMODE = np.zeros(Nfiles).astype(object)

for i_inpath, inpath in enumerate(inpath_list):
    with open(inpath) as f:
        data = json.load(f)
    if data['RAW.EXPTIME'] < 500:
        raise Exception(f'{inpath} is not a long exp!!!')
    saa_expo[i_inpath] = data['RAW.SAA']
    ra_expo[i_inpath] = data['RAW.RA']
    dec_expo[i_inpath] = data['RAW.DEC']
    cr_expo[i_inpath] = data['FPA.cosmic_pixel_percent']
    bkg_expo[i_inpath] = data['FPA.avg_bkg']
    zp_expo[i_inpath] = data['FPA.zero_point']
    stars_expo[i_inpath] = data['FPA.flagged_stars']
    ghosts_expo[i_inpath] = data['FPA.flagged_ghosts']

    exptime_expo[i_inpath] = data['RAW.EXPTIME']
    obstime_expo[i_inpath] = Time(data['RAW.DATE-OBS'], 
                                    format='isot', scale='utc').mjd

    dithobs_expo[i_inpath] = data['_dithobs']
    expnum_expo[i_inpath] = data['_expnum']
    obsid_expo[i_inpath] = data['_obs_id']
    patchid_expo[i_inpath] = data['RAW.PATCH_ID']

    CALBLKID[i_inpath] = data['RAW.CALBLKID']
    OBSMODE[i_inpath] = data['RAW.OBSMODE']

    del data

## Save data
df = pd.DataFrame({'CALBLKID': CALBLKID.astype(str),
                   'OBSMODE': OBSMODE.astype(str),
                    'patch_id': patchid_expo.astype(int),
                    'obs_id': obsid_expo.astype(int),
                    'dithobs': dithobs_expo.astype(int),
                    'expnum': expnum_expo.astype(int),
                    'SAA': saa_expo,
                    'RA': ra_expo, 
                    'DEC': dec_expo,
                    'cosmic_pixel_percent': cr_expo,
                    'avg_bkg': bkg_expo,
                    'zero_point': zp_expo,
                    'flagged_stars': stars_expo.astype(int),
                    'flagged_ghosts': ghosts_expo.astype(int),
                    'EXPTIME': exptime_expo,
                    'ObservationDateTime.MJD': obstime_expo})
df.to_csv(outpath, index=False)
del df
print(">>> info saved to", outpath)