# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-01-13 16:03:57

### Plot VIS footprint coloured by key properties 

import os

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## >>>>>>>>>>>>>>>>> I/O

## Where to find the catalogue
inpath = './outputs/VIS_info_DR1_R1_F006_D0E1.csv'

## Properties to be plotted
plotted_cols = ['cosmic_pixel_percent',
                'avg_bkg',
                'zero_point',
                'flagged_stars', 
                'flagged_ghosts',
                'ObservationDateTime.MJD'
                ]
plotted_cval_ranges = [[1.6, 3.0],
                      [13, 20],
                      [24.35, 24.55],
                      [1500, 5000],
                      [50, 250],
                      [60360, 60800]]

## >>>>>>>>>>>>>>>>> Workhorse

## Load the data
df = pd.read_csv(inpath)

## Convert bkg to microjansky as unit
df['avg_bkg'] = df['avg_bkg'] * 10 ** (-0.4*(df['zero_point'] - 23.90))
bkg_label = 'avg_bkg (microjansky)'

## Plot the footprint
cvals = [df[plotted_col] 
         for plotted_col in plotted_cols]
LABELs = [plotted_col 
          if plotted_col!='avg_bkg' 
          else bkg_label 
          for plotted_col in plotted_cols]
for i_col, cval in enumerate(cvals):
    label = LABELs[i_col]
    crange = plotted_cval_ranges[i_col]

    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(1, 2, 
                           width_ratios=[16, 4], 
                           wspace=0.05)
    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])

    if crange is not None:
        norm = mpl.colors.Normalize(vmin=crange[0], 
                                    vmax=crange[1])
    else:
        norm = None

    ## Scatter plot
    sc = ax.scatter(df['RA'].values, 
                    df['DEC'].values, 
                    c=cval, 
                    cmap='viridis', 
                    norm=norm, 
                    s=1, 
                    alpha=0.7)

    ## Custom colorbar with histogram overlay
    cb = mpl.colorbar.ColorbarBase(cax, 
                                   cmap='viridis', 
                                   norm=norm, 
                                   orientation='vertical')
    cb.set_label(label)

    ## Add histogram to colorbar
    hist_vals, bins = np.histogram(cval, 
                                   bins=60, 
                                   range=CRANGE, 
                                   density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    heights = 1 - hist_vals / hist_vals.max()
    cax.plot(heights, bin_centers, color='k', lw=1)

    ## Ecliptic and galactic lines
    ra_array = np.linspace(0, 360, 100)
    epsilon = 23.44 / 180 * np.pi
    dec_array = np.arctan(np.sin(ra_array/180*np.pi) * np.tan(epsilon)) / np.pi * 180
    ax.plot(ra_array, dec_array, color='r', label='ecliptic')

    ## Two clouds
    deltaG = 27.12825 / 180 * np.pi
    alphaG = 192.85948 / 180 * np.pi
    dec_array = np.arctan2(-1*np.cos(ra_array/180*np.pi - alphaG), np.tan(deltaG)) / np.pi * 180
    ax.plot(ra_array, dec_array, color='orange', label='galactic')
    ra_G = (17+45.6/60)/24*360
    dec_G = -28.94
    ax.plot(ra_G, dec_G, color='orange', marker='D', markersize=5)
    ax.plot(85.02, -69.76, color='purple', marker='*', markersize=10, label='LMC')
    ax.plot(13.19, -72.83, color='m', marker='*', markersize=8, label='SMC')

    ## Axis setup
    ax.set_xlim((0, 360))
    ax.xaxis.set_inverted(True)
    ax.set_ylim((-90, 90))
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title(os.path.basename(inpath))
    ax.legend()

    plt.show()
    plt.close()

