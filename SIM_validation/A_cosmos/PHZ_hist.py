# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-04-07 16:59:10

### Compare PHZ products
###### Distributions of key parameters

import os
import glob

import pandas as pd
import numpy as np

from astropy.io import fits

import plotting

## >>>>>>>>>>>>>>>> I/O

## Where to find catalogues
indir_phz_list = ['/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdPhzPfOutputCatalog/Mar25',
                  '/disks/shear10/ssli/euclid/COSMOS/DATA/COSMOS-WIDELIKE-DR1/DpdPhzPfOutputCatalog']
indir_mer_list = ['/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdMerFinalCatalog/DR1_STEP2_COSMOS_MER_R4',
              '/disks/shear10/ssli/euclid/COSMOS/DATA/COSMOS-WIDELIKE-DR1/DpdMerFinalCatalog']
LABELs = ['SIM', 'DATA']
TITLE = 'DR1_STEP2_COSMOS_MER_R4 vs COSMOS-WIDELIKE-DR1'
COLORs = ['blue', 'orange']

## Columns to load
FINAL_CAT_COLS = ['OBJECT_ID',
                  'VIS_DET', 'SPURIOUS_FLAG', 'DET_QUALITY_FLAG', 'POINT_LIKE_PROB',
                  'FLUX_DETECTION_TOTAL', 'FLUXERR_DETECTION_TOTAL']
PHZ_CAT_COLS = ['OBJECT_ID', 'PHZ_FLAGS',
                 'PHZ_MEDIAN', 'PHZ_MODE_1',
                  'FLUX_VIS_UNIF', 'FLUX_Y_UNIF']

## DET_QUALITY_FLAG: allow bits 1 (bad pixels/close neighbour), 2 (blended),
## and 512 (within extended object area); reject any other set bits.
ALLOWED_DET_FLAGS = np.int32(1 | 2 | 512)

## >>>>>>>>>>>>>>>> Load and merge catalogues
cata_list = []
for indir_phz, indir_mer in zip(indir_phz_list, indir_mer_list):

    ## Loop over all PHZ files and load desired columns
    inpath_list = glob.glob(os.path.join(indir_phz, 'EUC_PHZ_PHZCAT*.fits.gz'))
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
    inpath_list = glob.glob(os.path.join(indir_mer, 'EUC_MER_FINAL-CAT_TILE*.fits.gz'))
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
    cata_tmp = cata_tmp[
        (cata_tmp['VIS_DET'] == 1)
        & (cata_tmp['SPURIOUS_FLAG'] == 0)
        & ((cata_tmp['DET_QUALITY_FLAG'].astype(np.int32) & ~ALLOWED_DET_FLAGS) == 0)
        & (cata_tmp['FLUX_DETECTION_TOTAL'] / cata_tmp['FLUXERR_DETECTION_TOTAL'] > 5)
        & (cata_tmp['POINT_LIKE_PROB'] <= 0.8)
        & (cata_tmp['PHZ_FLAGS'] == 0)
    ]
    print(">>> Number of sources after selection", len(cata_tmp))

    ## Some columns for plotting convenience
    cata_tmp['MAG_DETECTION_TOTAL'] = -2.5 * np.log10(cata_tmp['FLUX_DETECTION_TOTAL']) + 23.9
    cata_tmp['MAG_VIS_UNIF-MAG_Y_UNIF'] = -2.5 * (np.log10(cata_tmp['FLUX_VIS_UNIF']
                                                                /cata_tmp['FLUX_Y_UNIF']))
    
    cata_list.append(cata_tmp.reset_index(drop=True))
    del cata_tmp

## >>>>>>>>>>>>>>>> Plot: compare the overall distribution

## What to plot
col_list = ['MAG_DETECTION_TOTAL', 'MAG_VIS_UNIF-MAG_Y_UNIF', 'PHZ_MODE_1']
XRANGE_list = [[20, 26], [-1, 2], [0, 5.0]]

outpath = 'show'
DENSITY = True
YLABEL = 'Density'
ylog = False
nbins = 60
wgs = None
for col, XRANGE in zip(col_list, XRANGE_list):
    paras = []
    for cata in cata_list:
        values = cata[col].values
        values = values[np.isfinite(values)]   # drop NaN/inf from failed fits
        paras.append(values)

    plotting.HistPlotFunc(outpath,
                    paras, wgs, COLORs, LABELs,
                    nbins, XRANGE, YRANGE=None,
                    XLABEL=col, YLABEL=YLABEL,
                    DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
                    TITLE=TITLE, xtick_min_label=True, ytick_min_label=True,
                    xtick_spe=None, ytick_spe=None,
                    vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                    hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                    xlog=False, ylog=ylog,
                    loc_legend='best',
                    font_size=10, usetex=False,
                    cumulative=False,
                    FIGSIZE=[6.4, 4.8],
                    LINEs=None, LINEWs=None,
                    TIGHT=False,
                    alpha=None)

## >>>>>>>>>>>>>>>> Plot: compare distributions in magnitude bins

## What to plot
col_list = ['MAG_VIS_UNIF-MAG_Y_UNIF', 'PHZ_MODE_1']
XRANGE_list = [[-1, 2], [0, 2]]

## Magnitude bins
magbins_low  = [20, 21, 22, 23, 24, 24.5]
magbins_high = [20.5, 21.5, 22.5, 23.5, 24.5, 25]

outpath = 'show'
N_plots = len(magbins_low)
nbins_list = [60] * N_plots
for col_name, XRANGE in zip(col_list, XRANGE_list):
    paras_list = []
    wgs_list = None
    COLORs_list = []
    LABELs_list = []
    subLABEL_list = []
    for bin_low, bin_high in zip(magbins_low, magbins_high):
        subLABEL_list.append(f'VIS [{bin_low}, {bin_high})')
        paras = []
        colors_bin = []
        labels_bin = []
        for cata, color, label in zip(cata_list, COLORs, LABELs):
            mask_tmp = (cata['MAG_DETECTION_TOTAL'] >= bin_low) & (cata['MAG_DETECTION_TOTAL'] < bin_high)
            if mask_tmp.sum() == 0:
                print(f'  >> no sources in VIS [{bin_low}, {bin_high}) for {label}, skipping')
                continue
            paras.append(cata.loc[mask_tmp, col_name].values)
            colors_bin.append(color)
            labels_bin.append(label)
        paras_list.append(paras)
        COLORs_list.append(colors_bin)
        LABELs_list.append(labels_bin)
        del paras

    plotting.HistPlotFunc_subplots(outpath, N_plots,
                                paras_list, wgs_list, COLORs_list, LABELs_list,
                                nbins_list, XRANGE, YRANGE=None,
                                subLABEL_list=subLABEL_list, subLABEL_locX=0.5, subLABEL_locY=0.8,
                                XLABEL=col_name, YLABEL=YLABEL,
                                DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
                                TITLE=TITLE, xtick_min_label=True, ytick_min_label=True,
                                xtick_spe=None, ytick_spe=None,
                                vlines_list=None,
                                vline_styles_list=None, vline_colors_list=None, vline_labels_list=None, vline_widths_list=None,
                                hlines_list=None,
                                hline_styles_list=None, hline_colors_list=None, hline_labels_list=None, hline_widths_list=None,
                                xlog=False, ylog=ylog,
                                loc_legend='best',
                                font_size=12, usetex=False,
                                LABEL_position=None, LABEL_position_SUBid=0,
                                LABEL_cols=None,
                                FIGSIZE=[6.4, 4.8],
                                TIGHT=False,
                                HISTTYPEs_list=None)


## >>>>>>>>>>>>>>>> Plot: compare distributions in redshift bins

## What to plot
col_list = ['MAG_DETECTION_TOTAL', 'MAG_VIS_UNIF-MAG_Y_UNIF']
XRANGE_list = [[20, 26], [-1, 2]]

## Redshift bins
redshiftbins_low  = [0.2, 0.48, 0.67, 0.86, 1.08, 1.44]
redshiftbins_high = [0.48, 0.67, 0.86, 1.08, 1.44, 2.5]

outpath = 'show'
N_plots = len(redshiftbins_low)
nbins_list = [60] * N_plots
for col_name, XRANGE in zip(col_list, XRANGE_list):
    paras_list = []
    wgs_list = None
    COLORs_list = []
    LABELs_list = []
    subLABEL_list = []
    for bin_low, bin_high in zip(redshiftbins_low, redshiftbins_high):
        subLABEL_list.append(f'PHZ [{bin_low}, {bin_high})')
        paras = []
        colors_bin = []
        labels_bin = []
        for cata, color, label in zip(cata_list, COLORs, LABELs):
            mask_tmp = (cata['PHZ_MODE_1'] >= bin_low) & (cata['PHZ_MODE_1'] < bin_high)
            if mask_tmp.sum() == 0:
                print(f'  >> no sources in Redshift [{bin_low}, {bin_high}) for {label}, skipping')
                continue
            paras.append(cata.loc[mask_tmp, col_name].values)
            colors_bin.append(color)
            labels_bin.append(label)
        paras_list.append(paras)
        COLORs_list.append(colors_bin)
        LABELs_list.append(labels_bin)
        del paras

    plotting.HistPlotFunc_subplots(outpath, N_plots,
                                paras_list, wgs_list, COLORs_list, LABELs_list,
                                nbins_list, XRANGE, YRANGE=None,
                                subLABEL_list=subLABEL_list, subLABEL_locX=0.4, subLABEL_locY=0.8,
                                XLABEL=col_name, YLABEL=YLABEL,
                                DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
                                TITLE=TITLE, xtick_min_label=True, ytick_min_label=True,
                                xtick_spe=None, ytick_spe=None,
                                vlines_list=None,
                                vline_styles_list=None, vline_colors_list=None, vline_labels_list=None, vline_widths_list=None,
                                hlines_list=None,
                                hline_styles_list=None, hline_colors_list=None, hline_labels_list=None, hline_widths_list=None,
                                xlog=False, ylog=ylog,
                                loc_legend='best',
                                font_size=12, usetex=False,
                                LABEL_position=None, LABEL_position_SUBid=0,
                                LABEL_cols=None,
                                FIGSIZE=[6.4, 4.8],
                                TIGHT=False,
                                HISTTYPEs_list=None)
