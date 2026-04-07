# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-04-01 02:18:58

### compare PHZ estimation with the true redshift in the SIM TU catalogue

import os
import glob

import numpy as np
import pandas as pd
from astropy.io import fits

import plotting

import sys
sys.path.append('../../misc') 
import CrossMatch

## >>>>>>>>>>>>>>>> I/O

## Where to find the catalogues
TITLE = 'DR1_STEP2_COSMOS_MER_R4'
indir_sim_in = '/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdTrueUniverseOutput/DR1_STEP2_COSMOS_VIS_TU_R3'
indir_sim_MER = '/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdMerFinalCatalog/DR1_STEP2_COSMOS_MER_R4'
indir_sim_PHZ = '/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdPhzPfOutputCatalog/Mar25'

## Columns to load
FINAL_CAT_COLS = ['OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION',
                  'VIS_DET', 'SPURIOUS_FLAG', 'DET_QUALITY_FLAG', 'POINT_LIKE_PROB',
                  'FLUX_DETECTION_TOTAL', 'FLUXERR_DETECTION_TOTAL']
PHZ_CAT_COLS = ['OBJECT_ID', 'PHZ_FLAGS',
                 'PHZ_MEDIAN', 'PHZ_MODE_1']

## DET_QUALITY_FLAG: allow bits 1 (bad pixels/close neighbour), 2 (blended),
## and 512 (within extended object area); reject any other set bits.
ALLOWED_DET_FLAGS = np.int32(1 | 2 | 512)

## >>>>>>>>>>>>>>>> Load input
## Find all the catalogue
inpath_list = glob.glob(os.path.join(indir_sim_in, 'EUC_SIM_TUGALCAT-*.fits'))
print(">>> number of files in sim TU", len(inpath_list))
cata_in = []
for inpath in inpath_list:
    ## Load cata
    with fits.open(inpath) as hdul:
        cata0_tmp = hdul[1].data
    ## used values
    cata0_tmp = pd.DataFrame({'SOURCE_ID': np.int64(cata0_tmp['SOURCE_ID']),
                            'RA_MAG_ORIG': np.float64(cata0_tmp['RA_MAG_ORIG']),
                            'DEC_MAG_ORIG': np.float64(cata0_tmp['DEC_MAG_ORIG']),
                            'Z_OBS': np.float64(cata0_tmp['Z_OBS']),
                            'TU_FNU_VIS_MAG': np.float64(cata0_tmp['TU_FNU_VIS_MAG'])})
    cata_in.append(cata0_tmp)
    del cata0_tmp
cata_in = pd.concat(cata_in, ignore_index=True)
print(">>>> number sim TU", len(cata_in))
cata_in.drop_duplicates(inplace=True, ignore_index=True)
print(">>>> number sim TU drop_duplicates ", len(cata_in))

## >>>>>>>>>>>>>>>> Load output

## Loop over all PHZ files and load desired columns
inpath_list = glob.glob(os.path.join(indir_sim_PHZ, 'EUC_PHZ_PHZCAT*.fits.gz'))
print(">>> Number of PHZ found", len(inpath_list))
cata_phz_list = []
for inpath in inpath_list:
    with fits.open(inpath) as hdul:
        data = hdul[1].data
        cata_tmp = pd.DataFrame({col: np.array(data[col]).byteswap().newbyteorder()
                                    for col in PHZ_CAT_COLS})
    cata_phz_list.append(cata_tmp)
    del cata_tmp
cata_phz = pd.concat(cata_phz_list, ignore_index=True)
del cata_phz_list
print(">>> Number of sources in PHZ cata", len(cata_phz))

## Loop over all MER files and load desired columns
inpath_list = glob.glob(os.path.join(indir_sim_MER, 'EUC_MER_FINAL-CAT_TILE*.fits.gz'))
print(">>> Number of MER found", len(inpath_list))
cata_mer_list = []
for inpath in inpath_list:
    with fits.open(inpath) as hdul:
        data = hdul[1].data
        cata_tmp = pd.DataFrame({col: np.array(data[col]).byteswap().newbyteorder()
                                    for col in FINAL_CAT_COLS})
    cata_mer_list.append(cata_tmp)
    del cata_tmp
cata_mer = pd.concat(cata_mer_list, ignore_index=True)
del cata_mer_list
print(">>> Number of sources in MER cata", len(cata_mer))

## Merge on OBJECT_ID
cata_tmp = cata_mer.merge(cata_phz, on='OBJECT_ID')
del cata_mer, cata_phz
print(">>> Number of sources after merging", len(cata_tmp))

## Quality cuts: keep only sources with good detection and measurement quality
cata_out = cata_tmp[
    (cata_tmp['VIS_DET'] == 1)
    & (cata_tmp['SPURIOUS_FLAG'] == 0)
    & ((cata_tmp['DET_QUALITY_FLAG'].astype(np.int32) & ~ALLOWED_DET_FLAGS) == 0)
    & (cata_tmp['FLUX_DETECTION_TOTAL'] / cata_tmp['FLUXERR_DETECTION_TOTAL'] > 5)
    & (cata_tmp['POINT_LIKE_PROB'] <= 0.8)
    & (cata_tmp['PHZ_FLAGS'] == 0)
].reset_index(drop=True)
del cata_tmp
print(">>> Number of sources after selection", len(cata_out))
cata_out['MAG_DETECTION_TOTAL'] = -2.5 * np.log10(cata_out['FLUX_DETECTION_TOTAL']) + 23.9

## >>>>>>>>>>>>>>>> Cross match based on sky position
id_list = ['SOURCE_ID', 'OBJECT_ID']
position_list = [['RA_MAG_ORIG', 'DEC_MAG_ORIG'], ['RIGHT_ASCENSION', 'DECLINATION']]
mag_list = ['TU_FNU_VIS_MAG', 'MAG_DETECTION_TOTAL']
matched_cata, _, _ = CrossMatch.run_position2id(cata_in, cata_out, id_list, position_list, mag_list,
                    outDir=None, basename=None, save_matched=False, save_false=False, save_missed=False,
                    r_max=0.5/3600., k=4, mag_closest=False, running_info=True,
                    useTan=False, pixel_scale=0.1, r_max_pixel=4)
cata_in = cata_in.merge(matched_cata, left_on='SOURCE_ID', right_on='id_input')
del matched_cata
print(">>> input after matching", len(cata_in))
cata_final = cata_in.merge(cata_out, left_on='id_detec', right_on='OBJECT_ID')
print(">>> output after matching", len(cata_final))
del cata_in, cata_out

## >>>>>>>>>>>>>>>> Check the cross-match

outpath = 'show'
dec_rad = np.deg2rad(cata_final['DECLINATION'].values)
paras = [(cata_final['RIGHT_ASCENSION'].values - cata_final['RA_MAG_ORIG'].values) * np.cos(dec_rad) * 3600,
         (cata_final['DECLINATION'].values - cata_final['DEC_MAG_ORIG'].values) * 3600]
wgs = None
COLORs = ['r', 'b']
LABELs = ['dRA', 'dDEC']
nbins = 60

XRANGE = [-0.5, 0.5]
XLABEL = 'MER - TU (arcsec)'
YLABEL = 'counts'

plotting.HistPlotFunc(outpath,
                paras, wgs, COLORs, LABELs,
                nbins, XRANGE, YRANGE=None,
                XLABEL=XLABEL, YLABEL=YLABEL,
                DENSITY=False, HISTTYPE='step', STACKED=False,
                TITLE=TITLE, xtick_min_label=True, ytick_min_label=True,
                xtick_spe=None, ytick_spe=None,
                vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                xlog=False, ylog=False,
                loc_legend='best', 
                LABEL_position='inSub', LABEL_cols=1,
                font_size=12, usetex=False,
                cumulative=False, 
                FIGSIZE=[6.4, 4.8],
                LINEs=None, LINEWs=None,
                TIGHT=False,
                alpha=None)

## >>>>>>>>>>>>>>>> Compare the z
outpath = 'show'
x_val = cata_final['Z_OBS'].values
y_val = cata_final['PHZ_MODE_1'].values 
wg = None
nbins = 60

XLABEL = 'Z_OBS (TrueUniverse)'
YLABEL = 'PHZ_MODE_1'

XRANGE = [0, 2.5]
YRANGE = [0, 2.5]

CBAR_LABEL = 'counts'
DENSITY = False
count_log = True

plotting.Hist2DPlotFunc(outpath,
                x_val, y_val, wg,
                nbins, XRANGE=XRANGE, YRANGE=YRANGE,
                XLABEL=XLABEL, YLABEL=YLABEL, CBAR_LABEL=CBAR_LABEL,
                COLOR_MAP='Reds',
                DENSITY=DENSITY, count_scale=[None, None], count_log=count_log,
                TITLE=TITLE, xtick_min_label=True, ytick_min_label=True,
                xtick_spe=None, ytick_spe=None,
                vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                font_size=12, usetex=False, 
                FIGSIZE=[6.4, 4.8],
                TIGHT=False)