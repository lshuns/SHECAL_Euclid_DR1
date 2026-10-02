# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-09-21 22:13:51
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-02 19:07:59

### Calculate m from constant shear simulations
### for lensmc catalogues
##### Five configurations for this purpose:
##### ShC1: "0,0.0354,0,0.0354"
##### ShC2: "0,0.0354,0,-0.0354"
##### ShC3: "0,-0.0354,0,-0.0354"
##### ShC4: "0,-0.0354,0,0.0354"
##### ShC6: "0,0,0,0"

import os
import re
import glob

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#### >>>>>>>>>>>>>>>>>>>>>>>> I/O
## Where to find all catalogues
main_dir = '/disks/shear10/ssli/euclid/DR1/SIM/STEP1b/preliminary_one_tile'

## Where to save the plots and the summary table
out_dir = './plots'
os.makedirs(out_dir, exist_ok=True)

## Shear info
shear_tags = ['STEP1B_ShC1', 
              'STEP1B_ShC2', 
              'STEP1B_ShC3', 
              'STEP1B_ShC4', 
              'STEP1B_ShC6']
## Each configuration was run twice with every galaxy rotated by 90 deg between the
## two runs: '_RG' and '_NRG'. 
rotation_tags = ['RG', 'NRG']
shear1_values = [0.0354, 
                 0.0354, 
                 -0.0354, 
                 -0.0354, 
                 0.0]
shear2_values = [0.0354, 
                 -0.0354, 
                 -0.0354, 
                 0.0354, 
                 0.0]

## Tomographic binning info
lower_redshift_list = [0.2, 0.48, 0.67, 0.86, 1.07, 1.42]
upper_redshift_list = [0.48, 0.67, 0.86, 1.07, 1.42, 2.5]

## which shear catalogue
shear_type = 'lenMC'
## the shear columns used for the bias estimate
E1_COL = 'SHE_LENSMC_E1'
E2_COL = 'SHE_LENSMC_E2'
WEIGHT_COL = 'SHE_LENSMC_WEIGHT'
## LensMC defines e1 in a frame whose x-axis runs along +RA, which is flipped with
## respect to the frame in which the input shear of the simulation is defined
E1_SIGN = -1.
E2_SIGN = +1.

## plotting colour 
COLOR_1 = 'blue'
COLOR_2 = 'orange'

## allowed MER quality values 
ALLOWED_DET_QUALITY = [0, 1, 2, 3, 512, 513, 514, 515]

## MW-corrected VIS magnitude range, expressed as uJy fluxes
FLUX_VIS_MAX = 575.44   # VIS >= 17.0
FLUX_VIS_MIN = 0.57544  # VIS <= 24.5

## columns to pull out of each catalogue
RA_COL = 'SHE_LENSMC_RA'
DEC_COL = 'SHE_LENSMC_DEC'
SHE_COLS = ['OBJECT_ID', E1_COL, E2_COL, WEIGHT_COL, RA_COL, DEC_COL]
MER_COLS = ['OBJECT_ID', 'DET_QUALITY_FLAG', 'SPURIOUS_FLAG']
PHZ_COLS = ['OBJECT_ID', 'PHZ_MODE_1', 'PHZ_FLAGS', 'FLUX_VIS_UNIF']

#### >>>>>>>>>>>>>>>>>>>>>>>> error estimation
## Every uncertainty quoted here is a delete-one-patch spatial jackknife over an
## N_PATCH_SIDE x N_PATCH_SIDE grid of equal-occupancy sky patches.
N_PATCH_SIDE = 8
n_patch = N_PATCH_SIDE**2
## realisation 0 is the full sample, 1..n_patch are the delete-one-patch samples
n_real = n_patch + 1

#### >>>>>>>>>>>>>>>>>>>>>>>> Helpers
def load_fits_cols(inpath, cols):
    """Load a few columns of a FITS table into a DataFrame (native byte order)."""
    with fits.open(inpath) as hdul:
        data = hdul[1].data
        out = {}
        for col in cols:
            values = np.asarray(data[col])
            if values.dtype.byteorder == '>':
                values = values.astype(values.dtype.newbyteorder('='))
            out[col] = values
    return pd.DataFrame(out)


def assign_patches(ra, dec, ra_edges, dec_edges):
    """Patch index of each object on the N_PATCH_SIDE x N_PATCH_SIDE sky grid."""
    return (np.digitize(ra, ra_edges[1:-1]) * N_PATCH_SIDE
            + np.digitize(dec, dec_edges[1:-1]))


def realisations(values, weights, patch_id):
    """Weighted mean of `values`, for the full sample and every delete-one-patch
    sample, returned as one array of length n_real (index 0 = full sample)."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(values)
    num = np.bincount(patch_id[good], weights=(weights[good] * values[good]),
                      minlength=n_patch)
    den = np.bincount(patch_id[good], weights=weights[good], minlength=n_patch)
    out = np.empty(n_real)
    out[0] = num.sum() / den.sum()
    out[1:] = (num.sum() - num) / (den.sum() - den)
    return out


def weight_realisations(weights, patch_id):
    """sum(w) for the full sample and every delete-one-patch sample."""
    den = np.bincount(patch_id, weights=np.asarray(weights, float), minlength=n_patch)
    out = np.empty(n_real)
    out[0] = den.sum()
    out[1:] = den.sum() - den
    return out


def jackknife_error(values):
    """Delete-one jackknife standard error of values[0], given values[1:] = the
    n_patch delete-one-patch repeats."""
    jk = np.asarray(values, float)[1:]
    return np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean())**2))


def linear_fit(x, y, yerr):
    """Weighted least-squares fit of y = slope * x + intercept, by the normal
    equations."""
    x, y, yerr = np.asarray(x, float), np.asarray(y, float), np.asarray(yerr, float)
    if not np.all(np.isfinite(yerr)) or np.any(yerr <= 0):
        yerr = np.ones_like(y)
    inv_var = 1. / yerr**2
    S = np.sum(inv_var)
    Sx = np.sum(x * inv_var)
    Sy = np.sum(y * inv_var)
    Sxx = np.sum(x * x * inv_var)
    Sxy = np.sum(x * y * inv_var)
    delta = S * Sxx - Sx**2
    slope = (S * Sxy - Sx * Sy) / delta
    intercept = (Sxx * Sy - Sx * Sxy) / delta
    return slope, intercept


def fit_realisations(shear_values, mean_real, sumw_real):
    """Fit <e> = (1 + m) g + c separately for the full sample and each
    delete-one-patch sample."""
    m_real, c_real = np.empty(n_real), np.empty(n_real)
    for i_real in range(n_real):
        slope, intercept = linear_fit(shear_values, mean_real[:, i_real],
                                      1. / np.sqrt(sumw_real[:, i_real]))
        m_real[i_real] = slope - 1.
        c_real[i_real] = intercept
    return m_real, c_real


#### >>>>>>>>>>>>>>>>>>>>>>>> Workhorse

## Loop over all shear configurations and both orientations
run_list = [(i_shear, f'{shear_tag}_{rotation}', rotation)
            for i_shear, shear_tag in enumerate(shear_tags)
            for rotation in rotation_tags]
cata_per_shear = [[] for _ in shear_tags]
for i_shear, run_tag, rotation in run_list:    
    ## find the shear folder
    tmp_dir = glob.glob(os.path.join(main_dir, run_tag, f'*_{shear_type}_*'))
    assert len(tmp_dir)==1, f"find {len(tmp_dir)} for {run_tag} {shear_type}"
    tmp_dir = tmp_dir[0]
    ## find catalogues
    inpath_list = glob.glob(os.path.join(tmp_dir, '*.fits.gz'))
    print("find ", len(inpath_list), "shear catalogues")

    ## find the MER folder
    #### '*_MER_*' also matches the MER_MORPHO product, which is not what we want
    mer_dir = [tmp for tmp in glob.glob(os.path.join(main_dir, run_tag, f'*_MER_*'))
               if 'MORPHO' not in os.path.basename(tmp)]
    assert len(mer_dir)==1, f"find {len(mer_dir)} MER (non-MORPHO) for {run_tag}"
    mer_dir = mer_dir[0]   

    ## find the PHZ folder
    phz_dir = glob.glob(os.path.join(main_dir, run_tag, f'*_PHZ_*'))
    assert len(phz_dir)==1, f"find {len(phz_dir)} for {run_tag}"
    phz_dir = phz_dir[0]

    ## loop over and load the cata
    cata_tag = []
    for inpath in inpath_list:
        ## Get the tile number
        tile_number = re.search(r'TILE-(\d+)', os.path.basename(inpath)).group(1)
        print(f"~~~ {run_tag} tile {tile_number}")

        ## Load the shear cata
        cata_shear = load_fits_cols(inpath, SHE_COLS)

        ## Load the MER cata
        inpath_mer = glob.glob(os.path.join(mer_dir, 
                                            f'EUC_MER_FINAL-CAT_TILE{tile_number}-*.fits.gz'))
        assert len(inpath_mer)==1, f"find {len(inpath_mer)} MER cata for {tile_number}"
        inpath_mer = inpath_mer[0]
        cata_mer = load_fits_cols(inpath_mer, MER_COLS)

        ## Load the PHZ cata
        #### the file name doesn't have tile number
        inpath_phz = glob.glob(os.path.join(phz_dir, f'EUC_PHZ_*.fits.gz'))
        assert len(inpath_phz)==1, f"find {len(inpath_phz)} PHZ cata"
        inpath_phz = inpath_phz[0]
        cata_phz = load_fits_cols(inpath_phz, PHZ_COLS)

        ## Merge the three catalogues
        cata = pd.merge(cata_shear, cata_mer, on='OBJECT_ID', how='inner')
        print(">>> cata_shear", len(cata_shear), 
              "cata_mer", len(cata_mer), 
              "after merge", len(cata))
        del cata_shear, cata_mer
        cata = pd.merge(cata, cata_phz, on='OBJECT_ID', how='inner')
        print(">>> cata_phz", len(cata_phz), 
              "after merge", len(cata))
        del cata_phz

        ## general selection
        mask = (cata['PHZ_FLAGS'] == 0)  # no problems with PHZ
        mask &= cata['DET_QUALITY_FLAG'].isin(ALLOWED_DET_QUALITY)  # allowed MER quality values
        mask &= (cata['SPURIOUS_FLAG'] == 0)  # not marked as spurious detection
        mask &= (cata['FLUX_VIS_UNIF'] <= FLUX_VIS_MAX)  # MW-corrected VIS >= 17.0
        mask &= (cata['FLUX_VIS_UNIF'] >= FLUX_VIS_MIN)  # MW-corrected VIS <= 24.5
        ## usable shear measurement
        mask &= (cata[WEIGHT_COL] > 0)
        mask &= np.isfinite(cata[E1_COL]) & np.isfinite(cata[E2_COL])
        cata = cata[mask].copy()
        print(">>> after the general selection", len(cata))

        ## apply the LensMC -> input-shear convention
        cata['e1'] = E1_SIGN * cata[E1_COL]
        cata['e2'] = E2_SIGN * cata[E2_COL]
        cata['weight'] = cata[WEIGHT_COL]
        cata['RA'] = cata[RA_COL]
        cata['DEC'] = cata[DEC_COL]
        ## which orientation this row came from (0 = RG, 1 = NRG)
        cata['rot'] = rotation_tags.index(rotation)

        cata_tag.append(cata[['e1', 'e2', 'weight', 'PHZ_MODE_1', 'rot', 'RA', 'DEC']])
        del cata

    cata_per_shear[i_shear].append(pd.concat(cata_tag, ignore_index=True))
    del cata_tag
    print(f">>> {run_tag}: {len(cata_per_shear[i_shear][-1])} sources\n")

## pool the orientations of each configuration into one sample
cata_list = [pd.concat(per_rot, ignore_index=True) for per_rot in cata_per_shear]
del cata_per_shear
e1k, e2k = 'e1', 'e2'
print(f">>> pooled over {rotation_tags}: "
      + ', '.join(f'{t}={len(c)}' for t, c in zip(shear_tags, cata_list)))
print(">>> shape-noise cancellation check -- <e> per orientation and pooled")
for shear_tag, cata in zip(shear_tags, cata_list):
    row = f'    {shear_tag}:'
    for i_rot, rot in enumerate(rotation_tags):
        sel = cata[cata['rot'] == i_rot]
        row += (f"  {rot}: <e1>={np.average(sel[e1k], weights=sel['weight']):+.5f}"
                f" <e2>={np.average(sel[e2k], weights=sel['weight']):+.5f}")
    row += (f"  | pooled: <e1>={np.average(cata[e1k], weights=cata['weight']):+.5f}"
            f" <e2>={np.average(cata[e2k], weights=cata['weight']):+.5f}")
    print(row)

## Define the jackknife patches once, from every configuration pooled
ra_all = np.concatenate([cata['RA'].values for cata in cata_list])
dec_all = np.concatenate([cata['DEC'].values for cata in cata_list])
ra_edges = np.quantile(ra_all, np.linspace(0, 1, N_PATCH_SIDE + 1))
dec_edges = np.quantile(dec_all, np.linspace(0, 1, N_PATCH_SIDE + 1))
del ra_all, dec_all
for cata in cata_list:
    cata['patch'] = assign_patches(cata['RA'].values, cata['DEC'].values,
                                   ra_edges, dec_edges)
print(f">>> {n_patch} jackknife patches "
      f"({N_PATCH_SIDE}x{N_PATCH_SIDE}); per-patch counts in the first configuration: "
      f"{np.bincount(cata_list[0]['patch'].values, minlength=n_patch)}")

## the samples to fit: the non-tomographic ones first, then the tomographic bins
#### 'NoZcut': everything passing the general selection, no cut on PHZ_MODE_1 at all
#### 'All'   : the union of the tomographic bins, i.e. the PHZ_MODE_1 range they span
bin_labels = ['NoZcut', 'All']
n_extra = len(bin_labels)
bin_edges = [(None, None),
             (lower_redshift_list[0], upper_redshift_list[-1])]
bin_titles = ['no cut on $z_{\\rm PHZ}$',
              f'{lower_redshift_list[0]} < $z_{{\\rm PHZ}}$ < {upper_redshift_list[-1]}']
## the sample used as the dashed reference in the m/c-vs-bin plot 
i_ref = bin_labels.index('All')
for i_bin, (lower_redshift, upper_redshift) in enumerate(zip(lower_redshift_list, upper_redshift_list)):
    bin_labels.append(f'B{i_bin+1}')
    bin_edges.append((lower_redshift, upper_redshift))
    bin_titles.append(f'{lower_redshift} < $z_{{\\rm PHZ}}$ < {upper_redshift}')

## calculate mean shear values and fit for m and c
mean_shear = []
fit_results = []
for label, (lower_redshift, upper_redshift) in zip(bin_labels, bin_edges):

    n_config = len(cata_list)
    e1_real = np.empty((n_config, n_real))
    e2_real = np.empty((n_config, n_real))
    sumw_real = np.empty((n_config, n_real))
    n_eff_list = []
    for i_config, cata in enumerate(cata_list):

        ## selection for the tomographic bin (None, None = keep everything)
        if lower_redshift is None:
            cata_bin = cata
        else:
            cata_bin = cata[(cata['PHZ_MODE_1'] >= lower_redshift)
                            & (cata['PHZ_MODE_1'] < upper_redshift)]
        weight = cata_bin['weight'].values
        patch = cata_bin['patch'].values

        ## weighted mean ellipticity = the shear estimate of this configuration,
        ## for the full sample and each delete-one-patch repeat
        e1_real[i_config] = realisations(cata_bin['e1'].values, weight, patch)
        e2_real[i_config] = realisations(cata_bin['e2'].values, weight, patch)
        sumw_real[i_config] = weight_realisations(weight, patch)

        n_eff_list.append(np.sum(weight)**2 / np.sum(weight**2))

    mean_shear.append({
            'e1': e1_real[:, 0],
            'e1_err': np.array([jackknife_error(row) for row in e1_real]),
            'e2': e2_real[:, 0],
            'e2_err': np.array([jackknife_error(row) for row in e2_real]),
            'n_eff': np.array(n_eff_list)})

    ## perform the linear fit to get m and c: <e_i> = (1 + m_i) g_i + c_i
    res = {'bin': label,
           'z_low': np.nan if lower_redshift is None else lower_redshift,
           'z_high': np.nan if upper_redshift is None else upper_redshift,
           'n_eff': float(np.sum(n_eff_list)),
           'n_patch': n_patch}
    for i_comp, shear_values in enumerate([shear1_values, shear2_values], start=1):
        mean_real = e1_real if i_comp == 1 else e2_real
        m_real, c_real = fit_realisations(shear_values, mean_real, sumw_real)
        res[f'm{i_comp}'] = m_real[0]
        res[f'm{i_comp}_err'] = jackknife_error(m_real)
        res[f'c{i_comp}'] = c_real[0]
        res[f'c{i_comp}_err'] = jackknife_error(c_real)
    fit_results.append(res)

## the summary table
fit_results = pd.DataFrame(fit_results)
outpath = os.path.join(out_dir, f'm_c_{shear_type}.csv')
fit_results.to_csv(outpath, index=False, float_format='%.6g')
print(f">>> summary table saved to {outpath}")
print(f"\n### shear bias, errors from the {n_patch}-patch spatial jackknife")
print(fit_results[['bin', 'z_low', 'z_high', 'n_eff',
                   'm1', 'm1_err', 'c1', 'c1_err',
                   'm2', 'm2_err', 'c2', 'c2_err']].to_string(index=False,
                                                              float_format='%.5f'))
print("\n### significance")
significance = pd.DataFrame({'bin': fit_results['bin']})
for key in ('m1', 'm2', 'c1', 'c2'):
    significance[f'{key}/sig'] = fit_results[key] / fit_results[f'{key}_err']
print(significance.to_string(index=False, float_format='%+.2f'))

#### >>>>>>>>>>>>>>>>>>>>>>>> Plots

## Plot the measured shear vs. input shear
for i_sample in range(n_extra):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex='col',
                             gridspec_kw={'height_ratios': [2.2, 1]})
    for i_comp, (shear_values, e_key, color) in enumerate([(shear1_values, 'e1', COLOR_1),
                                                           (shear2_values, 'e2', COLOR_2)]):
        ax_top, ax_bot = axes[0, i_comp], axes[1, i_comp]
        g_in = np.array(shear_values, float)
        e_out = mean_shear[i_sample][e_key]
        e_err = mean_shear[i_sample][e_key + '_err']
        res = fit_results.iloc[i_sample]
        m, m_err = res[f'm{i_comp+1}'], res[f'm{i_comp+1}_err']
        c, c_err = res[f'c{i_comp+1}'], res[f'c{i_comp+1}_err']

        g_line = np.linspace(g_in.min() - 0.008, g_in.max() + 0.008, 10)
        ax_top.plot(g_line, g_line, color='grey', lw=1, ls=':', zorder=1)
        ax_top.plot(g_line, (1 + m) * g_line + c, color=color, lw=2, zorder=2,
                    label='fit')
        ax_top.errorbar(g_in, e_out, yerr=e_err, fmt='o', ms=8, color=color,
                        mec='white', mew=1, ecolor=color, capsize=3, zorder=3,
                        label='tile mean')
        ax_top.set_ylabel(f'$\\langle e_{i_comp+1} \\rangle$')
        ## the fitted bias of this component, rather than just its name
        ax_top.set_title(f'$m_{i_comp+1}$ = {m:.4f} $\\pm$ {m_err:.4f}\n'
                         f'$c_{i_comp+1}$ = {c:.5f} $\\pm$ {c_err:.5f}', fontsize=11)
        ax_top.legend(loc='upper left', fontsize=8, frameon=False)
        ax_top.grid(alpha=0.2, lw=0.6)

        ax_bot.axhline(0, color='grey', lw=1, ls=':', zorder=1)
        ax_bot.plot(g_line, m * g_line + c, color=color, lw=2, zorder=2)
        ax_bot.errorbar(g_in, e_out - g_in, yerr=e_err, fmt='o', ms=8, color=color,
                        mec='white', mew=1, ecolor=color, capsize=3, zorder=3)
        ax_bot.set_xlabel(f'$g_{i_comp+1}^{{\\rm true}}$')
        ax_bot.set_ylabel(f'$\\langle e_{i_comp+1} \\rangle - g_{i_comp+1}$')
        ax_bot.grid(alpha=0.2, lw=0.6)

    fig.suptitle(f'{shear_type}: measured vs. input shear, '
                 f'{bin_labels[i_sample]} ({bin_titles[i_sample]})')
    fig.tight_layout()
    outpath = os.path.join(out_dir,
                           f'01_shear_bias_{bin_labels[i_sample]}_{shear_type}.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f">>> plot saved to {outpath}")

## the same shear response, one panel per tomographic bin
n_tomo = len(lower_redshift_list)
n_col = 3
n_row = int(np.ceil(n_tomo / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 3.2 * n_row),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes).flatten()
for i_bin in range(n_tomo):
    ax = axes[i_bin]
    ## the non-tomographic samples are the first n_extra entries
    shear = mean_shear[i_bin + n_extra]
    res = fit_results.iloc[i_bin + n_extra]
    ax.axhline(0, color='grey', lw=1, ls=':', zorder=1)
    for i_comp, (shear_values, e_key, color) in enumerate([(shear1_values, 'e1', COLOR_1),
                                                           (shear2_values, 'e2', COLOR_2)]):
        g_in = np.array(shear_values, float)
        m, c = res[f'm{i_comp+1}'], res[f'c{i_comp+1}']
        m_err = res[f'm{i_comp+1}_err']
        g_line = np.linspace(g_in.min() - 0.008, g_in.max() + 0.008, 10)
        ax.plot(g_line, m * g_line + c, color=color, lw=2, zorder=2)
        ax.errorbar(g_in, shear[e_key] - g_in, yerr=shear[e_key + '_err'], fmt='o',
                    ms=7, color=color, mec='white', mew=1, ecolor=color, capsize=3,
                    zorder=3,
                    label=f'$m_{i_comp+1}$ = {m:.4f} $\\pm$ {m_err:.4f}')
    ax.set_title(f'B{i_bin+1}: {res["z_low"]} < $z_{{\\rm PHZ}}$ < {res["z_high"]}',
                 fontsize=10)
    ax.legend(loc='best', fontsize=8, frameon=False)
    ax.grid(alpha=0.2, lw=0.6)
    if i_bin % n_col == 0:
        ax.set_ylabel(r'$\langle e_i \rangle - g_i$')
    if i_bin >= n_tomo - n_col:
        ax.set_xlabel(r'$g_i^{\rm true}$')
## hide the unused panels
for ax in axes[n_tomo:]:
    ax.set_visible(False)
fig.suptitle(f'{shear_type}: shear residual per tomographic bin')
fig.tight_layout()
outpath = os.path.join(out_dir, f'02_shear_bias_tomo_{shear_type}.png')
fig.savefig(outpath, dpi=150)
plt.close(fig)
print(f">>> plot saved to {outpath}")

## Plot the m values for the tomographic bins
x_bin = np.arange(1, n_tomo + 1)
tomo = fit_results.iloc[n_extra:]
fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
for i_panel, (key, ylabel) in enumerate([('m', 'multiplicative bias $m$'),
                                         ('c', 'additive bias $c$')]):
    ax = axes[i_panel]
    ax.axhline(0, color='grey', lw=1, ls=':', zorder=1)
    for i_comp, color in enumerate([COLOR_1, COLOR_2], start=1):
        ## the reference sample: dashed line at its value, shaded band at +/-1 sigma
        res_ref = fit_results.iloc[i_ref]
        ref_value = res_ref[f'{key}{i_comp}']
        ref_err = res_ref[f'{key}{i_comp}_err']
        ax.axhspan(ref_value - ref_err, ref_value + ref_err, color=color,
                   alpha=0.12, lw=0, zorder=0)
        ax.axhline(ref_value, color=color, lw=1, ls='--', alpha=0.6, zorder=1)
        ## offset the two components slightly so the error bars stay readable
        ax.errorbar(x_bin + 0.06 * (2 * i_comp - 3), tomo[f'{key}{i_comp}'].values,
                    yerr=tomo[f'{key}{i_comp}_err'].values, fmt='o', ms=8, color=color,
                    mec='white', mew=1, ecolor=color, capsize=3, zorder=3,
                    label=f'${key}_{i_comp}$')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, lw=0.6)
    ax.legend(loc='best', fontsize=9, frameon=False,
              title=f'dashed + band ($\\pm 1\\sigma$): {bin_labels[i_ref]}'
                    f' ({bin_titles[i_ref]})',
              title_fontsize=8)
axes[0].set_ylabel('multiplicative bias $m$')
axes[-1].set_xlabel('tomographic bin')
axes[-1].set_xticks(x_bin)
axes[-1].set_xticklabels([f'B{i}\n{lo}-{hi}' for i, (lo, hi)
                          in enumerate(zip(lower_redshift_list, upper_redshift_list), start=1)])
fig.suptitle(f'{shear_type}: shear bias per tomographic bin')
fig.tight_layout()
outpath = os.path.join(out_dir, f'03_m_c_tomo_{shear_type}.png')
fig.savefig(outpath, dpi=150)
plt.close(fig)
print(f">>> plot saved to {outpath}")

print("\n>>> Done.")