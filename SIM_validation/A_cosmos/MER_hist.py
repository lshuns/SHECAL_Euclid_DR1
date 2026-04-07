# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-03-27 23:58:35
# @Last Modified by:   lshuns
# @Last Modified time: 2026-04-01 00:23:23

### Compare MER products
###### Overall distributions of key parameters (e.g. magnitude, size, ellipticity) in SIM vs DATA, for galaxies and stars separately

import os
import re
import glob

import numpy as np
import pandas as pd
from astropy.io import fits

import plotting

## >>>>>>>>>>>>>>>> I/O

## Where to find catalogues
indir_list = ['/disks/shear10/ssli/euclid/DR1/SIM/STEP2_COSMOS/DpdMerFinalCatalog/DR1_STEP2_COSMOS_MER_R4',
              '/disks/shear10/ssli/euclid/COSMOS/DATA/COSMOS-WIDELIKE-DR1/DpdMerFinalCatalog']
LABELs = ['SIM', 'DATA']
# TITLE_main = 'DR1_STEP2_COSMOS_MER_R4 vs COSMOS-WIDELIKE-DR1'
TITLE_main = ''
COLORs = ['blue', 'orange']

## Columns to be compared
plotted_cols = [
    ## FLUX_* columns are converted to magnitude
    'FLUX_DETECTION_TOTAL',
    'FLUX_Y_1FWHM_APER',
    'FLUX_J_1FWHM_APER',
    'FLUX_H_1FWHM_APER',
    ## all others are plotted directly
    'MU_MAX',
    'MUMAX_MINUS_MAG',
    'ELLIPTICITY',
    'FLUX_RADIUS',
    'SERSIC_SERSIC_VIS_RADIUS',
    'SERSIC_SERSIC_VIS_AXIS_RATIO',
    'SERSIC_SERSIC_VIS_INDEX'
]

## X-axis ranges per column (None = auto)
XRANGE_map = {
    ## overridden based on object type (galaxies vs stars)
    'FLUX_DETECTION_TOTAL':     None,   
    'FLUX_Y_1FWHM_APER':       None,
    'FLUX_J_1FWHM_APER':       None,
    'FLUX_H_1FWHM_APER':       None,

    'MU_MAX':                   None,
    'MUMAX_MINUS_MAG':          None,
    'ELLIPTICITY':              [0, 1],
    'FLUX_RADIUS':              None,
    'SERSIC_SERSIC_VIS_RADIUS': None,
    'SERSIC_SERSIC_VIS_AXIS_RATIO': [0, 1],
    'SERSIC_SERSIC_VIS_INDEX': None,
}

XLABEL_map = {
    'FLUX_DETECTION_TOTAL':     'Magnitude (VIS)',
    'FLUX_Y_1FWHM_APER':       'Magnitude (NISP Y)',
    'FLUX_J_1FWHM_APER':       'Magnitude (NISP J)',
    'FLUX_H_1FWHM_APER':       'Magnitude (NISP H)',

    'MU_MAX':                   'MU_MAX (mag/arcsec2)',
    'MUMAX_MINUS_MAG':          'MU_MAX - MAG (mag/arcsec2)',
    'ELLIPTICITY':              'Ellipticity',
    'FLUX_RADIUS':              'FLUX_RADIUS (pixels)',
    'SERSIC_SERSIC_VIS_RADIUS': 'Sersic VIS radius (arcsec)',
    'SERSIC_SERSIC_VIS_AXIS_RATIO': 'Sersic VIS axis ratio',
    'SERSIC_SERSIC_VIS_INDEX': 'Sersic VIS Sersic index'
}

## Columns whose values are fluxes and need to be converted to magnitude
FLUX_COLS = {'FLUX_DETECTION_TOTAL', 'FLUX_Y_1FWHM_APER', 'FLUX_J_1FWHM_APER', 'FLUX_H_1FWHM_APER'}

## Columns to load
FINAL_CAT_COLS = ['OBJECT_ID',
                  'VIS_DET', 'SPURIOUS_FLAG', 'DET_QUALITY_FLAG', 'POINT_LIKE_PROB',
                  'FLUX_DETECTION_TOTAL', 'FLUXERR_DETECTION_TOTAL',
                  'FLUX_Y_1FWHM_APER', 'FLUX_J_1FWHM_APER', 'FLUX_H_1FWHM_APER',
                  'MU_MAX', 'MUMAX_MINUS_MAG', 'ELLIPTICITY']
MORPH_CAT_COLS = ['OBJECT_ID', 'FLUX_RADIUS', 'SERSIC_SERSIC_VIS_RADIUS',
                  'SERSIC_SERSIC_VIS_AXIS_RATIO', 'SERSIC_SERSIC_VIS_INDEX']

def latest_per_tile(inpath_list, pattern=r'TILE(\d+)'):
    """Return one path per tile — the most recent by datetime in the filename
    (e.g. 20250709T225637). Prints a warning when duplicates are found."""

    def _datetime(path):
        m = re.search(r'(\d{8}T\d{6})', os.path.basename(path))
        return m.group(1) if m else ''

    tile_to_path = {}
    for inpath in inpath_list:
        m = re.search(pattern, os.path.basename(inpath))
        if not m:
            continue
        tile_idx = m.group(1)
        if tile_idx not in tile_to_path:
            tile_to_path[tile_idx] = inpath
        elif _datetime(inpath) > _datetime(tile_to_path[tile_idx]):
            print(f"+++ Duplicate tile {tile_idx}: keeping {os.path.basename(inpath)}"
                  f" over {os.path.basename(tile_to_path[tile_idx])}")
            tile_to_path[tile_idx] = inpath
        else:
            print(f"+++ Duplicate tile {tile_idx}: keeping {os.path.basename(tile_to_path[tile_idx])}"
                  f" over {os.path.basename(inpath)}")
    return list(tile_to_path.values())

## DET_QUALITY_FLAG: allow bits 1 (bad pixels/close neighbour), 2 (blended),
## and 512 (within extended object area); reject any other set bits.
ALLOWED_DET_FLAGS = np.int32(1 | 2 | 512)

## >>>>>>>>>>>>>>>> Load and merge catalogues
cata_list = []
for indir in indir_list:
    inpath_list = latest_per_tile(glob.glob(os.path.join(indir, 'EUC_MER_FINAL-CAT_TILE*.fits.gz')))
    print(">>> Number of FINAL-CAT files found:", len(inpath_list))
    cata = []
    for inpath in inpath_list:
        ## Load FINAL-CAT
        with fits.open(inpath) as hdul:
            data = hdul[1].data
            cata_tmp = pd.DataFrame({col: np.array(data[col]).byteswap().newbyteorder()
                                     for col in FINAL_CAT_COLS})
        print(">>> Number of sources in final cata", len(cata_tmp))

        ## Find and load the matching MORPH-CAT
        tile_label = re.search(r'EUC_MER_FINAL-CAT_TILE(\d+)', os.path.basename(inpath)).group(1)
        inpath_morph_list = latest_per_tile(
            glob.glob(os.path.join(indir, f'EUC_MER_FINAL-MORPH-CAT_TILE{tile_label}*.fits.gz')))
        assert len(inpath_morph_list) == 1, \
            f"Expected 1 morph file for tile {tile_label}, found {len(inpath_morph_list)}"

        with fits.open(inpath_morph_list[0]) as hdul:
            data_morph = hdul[1].data
            cata_morph_tmp = pd.DataFrame({col: np.array(data_morph[col]).byteswap().newbyteorder()
                                           for col in MORPH_CAT_COLS})
        print(">>> Number of sources in morph cata", len(cata_morph_tmp))

        ## Merge on OBJECT_ID
        cata_tmp = cata_tmp.merge(cata_morph_tmp, on='OBJECT_ID')
        print(">>> Number of sources after merging", len(cata_tmp))

        ## Quality cuts: keep only sources with good detection and measurement quality
        cata_tmp = cata_tmp[
            (cata_tmp['VIS_DET'] == 1)
            & (cata_tmp['SPURIOUS_FLAG'] == 0)
            & ((cata_tmp['DET_QUALITY_FLAG'].astype(np.int32) & ~ALLOWED_DET_FLAGS) == 0)
            & (cata_tmp['FLUX_DETECTION_TOTAL'] / cata_tmp['FLUXERR_DETECTION_TOTAL'] > 5)
        ]
        print(">>> Number of sources after quality cuts", len(cata_tmp))
        # print("++++++ Unique DET_QUALITY_FLAG values and counts:\n", cata_tmp['DET_QUALITY_FLAG'].value_counts())

        cata.append(cata_tmp)
        del cata_tmp, cata_morph_tmp
    cata = pd.concat(cata, ignore_index=True)
    cata_list.append(cata)

## >>>>>>>>>>>>>>>> Plot: one figure per (type, column)
type_list = ['galaxies', 'stars']
YLABEL = 'Density'
DENSITY = True

for s_type in type_list:
    TITLE = f"{TITLE_main} ({s_type})".strip()

    ## Selection based on POINT_LIKE_PROB
    if s_type == 'galaxies':
        masks = [(cata['POINT_LIKE_PROB'] <= 0.8)
                 for cata in cata_list]
    else:
        masks = [(cata['POINT_LIKE_PROB'] > 0.8)
                 for cata in cata_list]

    for col in plotted_cols:

        paras = []
        for cata, mask in zip(cata_list, masks):
            if col in FLUX_COLS:
                flux = cata.loc[mask, col].values
                valid = flux > 0
                values = -2.5 * np.log10(flux[valid]) + 23.9
            else:
                values = cata.loc[mask, col].values
                values = values[np.isfinite(values)]   # drop NaN/inf from failed fits
            paras.append(values)

        ## XRANGE: magnitude range depends on object type; others auto from data
        if col in FLUX_COLS:
            XRANGE = [20, 27] if s_type == 'galaxies' else [18, 24]
        elif XRANGE_map[col] is not None:
            XRANGE = XRANGE_map[col]
        else:
            non_empty = [p for p in paras if len(p) > 0]
            if not non_empty:
                print(f'  >> no finite values for {col}, skipping')
                continue
            all_vals = np.concatenate(non_empty)
            XRANGE = [float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))]

        outpath = 'show'
        wgs = None
        nbins = 60

        plotting.HistPlotFunc(outpath,
                paras, wgs, COLORs, LABELs,
                nbins, XRANGE, YRANGE=None,
                XLABEL=XLABEL_map[col], YLABEL=YLABEL,
                DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
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
                alpha=None,
                HISTTYPE_list=None)