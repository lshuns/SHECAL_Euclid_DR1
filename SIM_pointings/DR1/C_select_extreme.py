# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-01-13 15:53:18

### Select extremes for given observational properties
###### cover min and max
###### extreme in one is normal in others
###### do not care about the continuity of ObsIDs

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotting

# >>>>>>>>>>>>>>> I/O

## Where to find the catalogues
inpath = './outputs/VIS_info_DR1_R1_F006_D0E1.csv'

## Total number of cells
N_cells = 61  

## Number of ROS each extreme
Nros = 2

## Properties used for selection (ordered)
col_list = ['cosmic_pixel_percent', 
            'avg_bkg', 
            'flagged_stars']

## Which sky
# south = True
south = False

## >>>>>>>>>>>>>>> Workhorse

## Load the data 
df_whole = pd.read_csv(inpath)
print('>>> Number total', len(df_whole))

## Select based on the sky
if south:
    df_whole = df_whole[(df_whole['DEC'].values<0)]
    print('>>>>> Number in south', len(df_whole))
    plot_title = f'{os.path.basename(inpath)} (south)'
    label_whole = 'DR1 (south)'
else:
    df_whole = df_whole[(df_whole['DEC'].values>0)]
    print('>>>>> Number in north', len(df_whole))
    plot_title = f'{os.path.basename(inpath)} (north)'
    label_whole = 'DR1 (north)'

## Remove the one with flagged_stars > 10000, assume sth weird happened there
df_whole = df_whole[(df_whole['flagged_stars'].values<10000)]
df_whole.reset_index(drop=True, inplace=True)
print('>>> Number after removing too many flagged_stars', len(df_whole))

## Quantile binning in given properties
for i_col, col in enumerate(col_list):
    df_whole[f'cell_{i_col}'] = pd.qcut(df_whole[col], 
                                  q=N_cells, 
                                  labels=False)

## Select extremes
Ncols = len(col_list)
df_selected_list = []
for i_col in range(Ncols):
    print("+++ For col", col_list[i_col])

    df_low = df_whole.loc[df_whole[f'cell_{i_col}'] == 0].copy()
    df_low.loc[:, 'dist'] = np.linalg.norm(df_low[[f'cell_{i_col_tmp}' for i_col_tmp in range(Ncols) if i_col_tmp!=i_col]] - [(N_cells-1)/2., (N_cells-1)/2.], axis=1)
    df_low = df_low.sort_values(by=['dist']).head(Nros)
    df_low.reset_index(drop=True, inplace=True)
    # print(">>>>>> low candidates")
    # print(df_low[['patch_id', 'obs_id']+[f'cell_{i_col_tmp}' for i_col_tmp in range(Ncols)]])
    print(">>>>>> low:")
    print(df_low[['patch_id', 'obs_id']].sort_values(by='obs_id').to_string(index=False))
    
    df_high = df_whole.loc[df_whole[f'cell_{i_col}'] == (N_cells-1)].copy()
    df_high.loc[:, 'dist'] = np.linalg.norm(df_high[[f'cell_{i_col_tmp}' for i_col_tmp in range(Ncols) if i_col_tmp!=i_col]] - [(N_cells-1)/2., (N_cells-1)/2.], axis=1)
    df_high = df_high.sort_values(by=['dist']).head(Nros)
    df_high.reset_index(drop=True, inplace=True)
    # print(">>>>>> high candidates")
    # print(df_high[['patch_id', 'obs_id']+[f'cell_{i_col_tmp}' for i_col_tmp in range(Ncols)]])
    print(">>>>>> high")
    print(df_high[['patch_id', 'obs_id']].sort_values(by='obs_id').to_string(index=False))

    ## Plot histogram
    for i_col_tmp in range(Ncols):
        outpath = 'show'
        XLABEL = col_list[i_col_tmp]
        XRANGE = [np.min(df_whole[col_list[i_col_tmp]]),
                np.max(df_whole[col_list[i_col_tmp]])]
        paras = [df_whole[col_list[i_col_tmp]], 
                df_low[col_list[i_col_tmp]],
                df_high[col_list[i_col_tmp]]]
        wgs = None
        COLORs = ['k', 'r', 'b']
        LABELs = [label_whole, 'low', 'high']
        nbins = 60

        YRANGE = None
        YLABEL = 'DENSITY'
        DENSITY = True

        plotting.HistPlotFunc(outpath,
                        paras, wgs, COLORs, LABELs,
                        nbins, XRANGE, YRANGE=YRANGE,
                        XLABEL=XLABEL, YLABEL=YLABEL,
                        DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
                        TITLE=plot_title, xtick_min_label=True, ytick_min_label=True,
                        xtick_spe=None, ytick_spe=None,
                        vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                        hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                        xlog=False, ylog=False,
                        loc_legend='best', 
                        font_size=10, usetex=False,
                        cumulative=False, 
                        FIGSIZE=[6.4, 4.8],
                        LINEs=None, LINEWs=None,
                        TIGHT=False,
                        alpha=None)

    ## Collect selections
    df_selected_list.append(df_high)
    df_selected_list.append(df_low)
    del df_low, df_high
df_selected = pd.concat(df_selected_list, ignore_index=True)
del df_selected_list
print(">>> total number of extremes selected", len(df_selected))
print(">>>> selected: ")
print(df_selected[['patch_id', 'obs_id']+[f'cell_{i_col_tmp}' for i_col_tmp in range(Ncols)]])

## Plot histogram
for i_col in range(Ncols):
    outpath = 'show'
    XLABEL = col_list[i_col]
    XRANGE = [np.min(df_whole[col_list[i_col]]),
              np.max(df_whole[col_list[i_col]])]
    paras = [df_whole[col_list[i_col]], 
             df_selected[col_list[i_col]]]
    wgs = None
    COLORs = ['k', 'red']
    LABELs = [label_whole, 'Selected']
    nbins = 60

    YRANGE = None
    YLABEL = 'DENSITY'
    DENSITY = True

    plotting.HistPlotFunc(outpath,
                    paras, wgs, COLORs, LABELs,
                    nbins, XRANGE, YRANGE=YRANGE,
                    XLABEL=XLABEL, YLABEL=YLABEL,
                    DENSITY=DENSITY, HISTTYPE='step', STACKED=False,
                    TITLE=plot_title, xtick_min_label=True, ytick_min_label=True,
                    xtick_spe=None, ytick_spe=None,
                    vlines=None, vline_styles=None, vline_colors=None, vline_labels=None, vline_widths=None,
                    hlines=None, hline_styles=None, hline_colors=None, hline_labels=None, hline_widths=None,
                    xlog=False, ylog=False,
                    loc_legend='best', 
                    font_size=10, usetex=False,
                    cumulative=False, 
                    FIGSIZE=[6.4, 4.8],
                    LINEs=None, LINEWs=None,
                    TIGHT=False,
                    alpha=None)

print(f">>>> selected step 2 ROS from {plot_title}: ")
print(df_selected[['patch_id', 'obs_id']].sort_values(by='obs_id').to_string(index=False))
print(np.sort(df_selected['obs_id'].values))

## Plot the footprint
fig, ax = plt.subplots()
## All ROS
plt.scatter(df_whole['RA'].values, df_whole['DEC'].values, 
                c='k', 
                s=10, alpha=0.7,
                label=label_whole)
## Selected ROS
plt.scatter(df_selected['RA'].values, df_selected['DEC'].values, 
                c='red', 
                s=20,
                label='Selected')
## Plot info
if south:
    ax.set_xlim((0, 100))
    ax.set_ylim((-80, -20))
else:
    ax.set_xlim((120, 290))
    ax.set_ylim((50, 85))
ax.xaxis.set_inverted(True)
plt.title(plot_title)
plt.legend()
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
plt.show()
plt.close()