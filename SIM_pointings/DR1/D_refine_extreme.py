# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2025-07-17 15:40:32
# @Last Modified by:   lshuns
# @Last Modified time: 2026-01-13 15:59:17

### Refine the selected extremes to avoid it to be too close to the edge
###### This is done manually

import os

import random
import colorsys
import pandas as pd
import matplotlib.pyplot as plt

## >>>>>>>>>>>>>>> I/O

## Where to find the catalogues
inpath = './outputs/VIS_info_DR1_R1_F006_D0E1.csv'

## Which sky
# south = True
south = False

## Selected obs_ids
### Original one
obs_id_list = [3218, 3241, 4222, 4299,
                7275, 7279, 7329, 7334, 
                7337, 7340, 7726, 7727]
### New
obs_id_list = [3218, 3240, 4222, 4299,
                7275, 7279, 7330, 7334, 
                7337, 7341, 7726, 7727]

## >>>>>>>>>>>>>>> Workhorse

## load the data 
df_whole = pd.read_csv(inpath)
print('>>> number total', len(df_whole))

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
print('>>> number after removing too many flagged_stars', len(df_whole))

## Check sky
random.seed(9001)
def generate_distinct_colors(n):
    hues = [i / n for i in range(n)]
    random.shuffle(hues)  
    colors = [
        colorsys.hsv_to_rgb(h, 0.8, 0.9) 
        for h in hues
    ]
    return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
colors = generate_distinct_colors(len(obs_id_list))

## Plot the footprint
fig, ax = plt.subplots()
## All ROS
plt.scatter(df_whole['RA'].values, df_whole['DEC'].values, 
                c='k', 
                s=10, alpha=0.7,
                label=label_whole)
## Selected ROS
for i_obs, obs_id in enumerate(obs_id_list):
    mask_tmp = df_whole['obs_id'] == obs_id
    plt.scatter(df_whole.loc[mask_tmp, 'RA'].values, 
                df_whole.loc[mask_tmp, 'DEC'].values, 
                c=colors[i_obs], 
                s=20,
                label=obs_id)
## Plot info
if south:
    ax.set_xlim((0, 100))
    ax.set_ylim((-80, -20))
else:
    ax.set_xlim((120, 360))
    ax.set_ylim((50, 85))
ax.xaxis.set_inverted(True)
plt.title(plot_title)
plt.legend()
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
plt.show()
plt.close()