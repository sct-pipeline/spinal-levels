"""
TL;DR: Compute the cervical spinal cord length from the top of C2 to the bottom of C8 along the centerline.

This script computes the cervical spinal cord length (top of the C2 spinal level to the bottom of the C8 spinal level)
for the KCL cervical dataset (sub-HFSPAIN* and sub-SPAIN*, two sessions per subject: ses-A and ses-B) from the
spinal cord and rootlets segmentations, and plots it per subject.

    1. The spinal levels are obtained from the rootlets as in ../rootlets_to_spinal_levels.py: the spinal cord
       segmentation is dilated (default: 3 voxels) and intersected with the rootlets segmentation. The extent of each
       spinal level is given by the most superior and the most inferior slice of the intersection.
    2. The centerline is fitted to the spinal cord segmentation using SCT's get_centerline() with ParamCenterline
       (default: bspline, smooth=30, as in sct_process_segmentation) and converted to physical space (mm).
    3. The length is the arc length of the centerline from the top edge of the most superior slice of C2 to the bottom
       edge of the most inferior slice of C8 (i.e., both end slices included), as in
       ../lumbar-balgrist/lumbar_length_along_centerline.py.

Input:
- folder with the segmentations (sub-*/ses-*/*_T2w_seg.nii.gz and *_T2w_rootlets.nii.gz or
  *_T2w_rootlets_labels.nii.gz); rootlets values: 2=C2, 3=C3, ..., 8=C8, 9=T1
- participants.csv with the subject sex and age (columns: id, sex, age); no height is available for this dataset

Output (saved to the -o folder, default: ~/data/dwi_normative_database/Olivia_cord_length/results_and_figures):
    - cervical_length_C2-C8_centerline.csv: length along the centerline per subject and session, with sex and age
      (only subjects with a valid length in both sessions)
    - figure_cervical_length_C2-C8_centerline.png: length per subject (ses-A and ses-B) and test-retest (ses-A vs ses-B,
      with Spearman correlation for sessions on the same day (sub-HFSPAIN*, circles) and on separate days (sub-SPAIN*, squares))
    - qc/*_spinal_levels.nii.gz: spinal levels projected on the spinal cord segmentation, for visual QC
    - qc/*_seg_centerline.nii.gz: fitted centerline, for visual QC

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python cervical_length_along_centerline.py
    python cervical_length_along_centerline.py -i <path/to/seg_cord_rootlet> -o <output folder>
    python cervical_length_along_centerline.py -plot-only    # only regenerate the figure from the saved CSV file

Author: Jan Valosek
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.math import dilate
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

# First and last spinal level (values in the rootlets segmentation)
FIRST_LEVEL = 2  # C2
LAST_LEVEL = 8   # C8
METRIC_NAME = f'length_C{FIRST_LEVEL}-C{LAST_LEVEL}_centerline'
METRIC_LABEL = f'Length top C{FIRST_LEVEL} – bottom C{LAST_LEVEL}'

DATA_DIR = os.path.expanduser('~/data/dwi_normative_database/Olivia_cord_length')
INPUT_DIR = os.path.join(DATA_DIR, 'seg_cord_rootlet')
PARTICIPANTS_CSV = os.path.join(DATA_DIR, 'participants.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'results_and_figures')

FONT_SIZE = 14

# Female and male colors
SEX_STYLE = {'F': {'color': '#d62728', 'label': 'Female'},
             'M': {'color': '#2a78d6', 'label': 'Male'}}

# Subject prefix: when the two sessions (ses-A and ses-B) were acquired, and the marker
RETEST = {'sub-HFSPAIN': {'label': 'same day', 'marker': 'o'},
          'sub-SPAIN': {'label': 'separate days', 'marker': 's'}}


def get_spinal_levels(im_rootlets, im_seg, dilate_size):
    """
    Project the rootlets on the spinal cord segmentation to obtain the spinal levels (same as
    ../rootlets_to_spinal_levels.py): intersect the dilated spinal cord segmentation with the rootlets.
    :param im_rootlets: rootlets segmentation (Image) in RPI orientation
    :param im_seg: binary spinal cord segmentation (Image) in RPI orientation
    :param dilate_size: size of the spinal cord segmentation dilation in voxels
    :return: intersection data (rootlets values within the dilated spinal cord segmentation), and the spinal levels
    projected on the spinal cord segmentation (for visual QC)
    """
    seg_dil = dilate(im_seg.data, dilate_size, 'ball')
    intersect = im_rootlets.data * (seg_dil > 0)
    spinal_levels = np.zeros_like(im_seg.data, dtype=np.uint8)
    for level in np.unique(intersect[intersect > 0]).astype(int):
        z = np.where(intersect == level)[2]
        spinal_levels[:, :, z.min():z.max() + 1][im_seg.data[:, :, z.min():z.max() + 1] > 0] = level
    return intersect, spinal_levels


def get_level_slice(data, value, point):
    """
    Get a slice position of a spinal level along the superior-inferior (S-I) axis, in RPI slice index.
    The top and bottom positions are at the edge of the end slice (+/- half a slice), so that both end slices are
    included in the length.
    :param data: data in RPI orientation
    :param value: value of the spinal level
    :param point: 'top': top edge of the most superior slice; 'bottom': bottom edge of the most inferior slice
    :return: slice position (float, in slices); NaN if the level is missing
    """
    z = np.where(data == value)[2]  # S-I index of each voxel of this level
    if len(z) == 0:
        return np.nan
    return {'top': z.max() + 0.5, 'bottom': z.min() - 0.5}[point]


def fit_centerline(im_seg, param_centerline, fname_out=None, orientation_out='RPI'):
    """
    Fit the spinal cord centerline to the spinal cord segmentation and convert it to physical space.
    :param im_seg: binary spinal cord segmentation (Image) in RPI orientation
    :param param_centerline: ParamCenterline
    :param fname_out: if provided, the centerline image is saved there (for visual QC)
    :param orientation_out: orientation of the saved centerline image (native orientation of the segmentation)
    :return: Centerline (SCT) in physical space (mm), and the RPI slice index of each centerline point
    """
    # space='pix' keeps the slice index of each point (third row of arr_ctl); the points are then converted to
    # physical space so that the length is in mm and the voxel size and image obliquity are accounted for
    im_ctl, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=param_centerline, verbose=0, space='pix')
    points = im_seg.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_seg.transfo_pix2phys(arr_ctl_der.T, mode='relative')
    if fname_out:
        im_ctl.change_orientation(orientation_out).save(fname_out, mutable=True, verbose=0)
    return Centerline(*points.T, *deriv.T), arr_ctl[2]


def compute_length(fname_seg, fname_rootlets, param_centerline, dilate_size, dir_qc):
    """
    Compute the length from the top of C2 to the bottom of C8 along the centerline (in mm) for one session.
    :param fname_seg: path to the spinal cord segmentation
    :param fname_rootlets: path to the rootlets segmentation
    :param param_centerline: ParamCenterline
    :param dilate_size: size of the spinal cord segmentation dilation in voxels
    :param dir_qc: folder where the spinal levels and the fitted centerline are saved (for visual QC)
    :return: dict with the length in mm (NaN if one of the levels is missing) and the C2 top and C8 bottom slices
    """
    # Reorient to RPI so that the third axis is the S-I axis (slice 0 is the most inferior)
    orientation = Image(fname_seg).orientation
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_seg.data = (im_seg.data > 0.5).astype(np.uint8)
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')

    intersect, spinal_levels = get_spinal_levels(im_rootlets, im_seg, dilate_size)
    im_levels = im_seg.copy()
    im_levels.data = spinal_levels
    im_levels.change_orientation(orientation).save(
        os.path.join(dir_qc, os.path.basename(fname_rootlets).replace('.nii.gz', '_spinal_levels.nii.gz')),
        mutable=True, verbose=0)

    z_first = get_level_slice(intersect, FIRST_LEVEL, 'top')
    z_last = get_level_slice(intersect, LAST_LEVEL, 'bottom')
    out = {'slice_top_C2_I-S': z_first, 'slice_bottom_C8_I-S': z_last, f'{METRIC_NAME}_mm': np.nan}
    if np.isnan(z_first) or np.isnan(z_last):
        print(f"  WARNING: level {FIRST_LEVEL} or {LAST_LEVEL} missing; the length will be NaN.")
        return out

    # Centerline in physical space; cumulative length (mm) at each centerline point, from the most inferior one
    centerline, z_ctl = fit_centerline(im_seg, param_centerline,
                                       fname_out=os.path.join(dir_qc, os.path.basename(fname_seg).replace('.nii.gz', '_centerline.nii.gz')),
                                       orientation_out=orientation)
    # The end slices must be within the centerline (i.e., within the spinal cord segmentation)
    if z_first > z_ctl.max() + 0.5 or z_last < z_ctl.min() - 0.5:
        print(f"  WARNING: top of C{FIRST_LEVEL} or bottom of C{LAST_LEVEL} outside the spinal cord segmentation.")
    length_at_slice = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
    out[f'{METRIC_NAME}_mm'] = float(length_at_slice(z_first) - length_at_slice(z_last))
    print(f"  {METRIC_LABEL}: {out[f'{METRIC_NAME}_mm']:.1f} mm along the centerline")
    return out


def add_sex_age(df, fname_csv):
    """
    Add the subject sex ('F'/'M') and age from participants.csv to the dataframe.
    :param df: dataframe with a 'subject' column
    :param fname_csv: CSV file with the 'id', 'sex' ('female'/'male') and 'age' columns
    :return: dataframe with 'sex' and 'age' columns (NaN for subjects not in the CSV file)
    """
    # utf-8-sig: the CSV file starts with a byte order mark
    df_participants = pd.read_csv(fname_csv, encoding='utf-8-sig')
    df_participants['sex'] = df_participants['sex'].str.strip().str[0].str.upper()   # 'female'/'male' -> 'F'/'M'
    return df.merge(df_participants[['id', 'sex', 'age']], how='left', left_on='subject',
                    right_on='id').drop(columns='id')


def generate_figure(df, dir_out):
    """
    Plot the length per subject (ses-A and ses-B connected by a line, colored by sex; subjects sorted by the mean
    length) and the test-retest agreement (ses-A vs ses-B). The subjects with both sessions on the
    same day (sub-HFSPAIN*, circles) and on separate days (sub-SPAIN*, squares) are shown with different markers and
    the test-retest statistics are computed separately for them.
    :param df: dataframe with 'subject', 'session', 'sex' and the length columns
    :param dir_out: output folder
    """
    col = f'{METRIC_NAME}_mm'
    df_wide = df.pivot_table(index=['subject', 'sex'], columns='session', values=col).reset_index()
    df_wide['mean'] = df_wide[['ses-A', 'ses-B']].mean(axis=1)
    df_wide = df_wide.sort_values('mean').reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [2.5, 1.2]})

    # A: length per subject
    # Color = sex, marker = when the sessions were acquired, filled = ses-A, hollow = ses-B
    x = np.arange(1, len(df_wide) + 1)
    for sex, style in SEX_STYLE.items():
        for prefix, retest in RETEST.items():
            mask = ((df_wide['sex'] == sex) & df_wide['subject'].str.startswith(prefix)).to_numpy()
            for ses, fill in [('ses-A', style['color']), ('ses-B', 'white')]:
                ax1.scatter(x[mask], df_wide.loc[mask, ses] / 10, s=25, zorder=3, marker=retest['marker'],
                            facecolors=fill, edgecolors=style['color'])
            ax1.vlines(x[mask], df_wide.loc[mask, ['ses-A', 'ses-B']].min(axis=1) / 10,
                       df_wide.loc[mask, ['ses-A', 'ses-B']].max(axis=1) / 10, color=style['color'], lw=1, zorder=2)
    # Legend: one entry per color, marker and fill
    handles = [Line2D([], [], ls='', marker='o', color=style['color'],
                      label=f"{style['label']} (n = {(df_wide['sex'] == sex).sum()})")
               for sex, style in SEX_STYLE.items()]
    handles += [Line2D([], [], ls='', marker=retest['marker'], mfc='white', mec='gray',
                       label=f"{retest['label'].capitalize()} (n = {df_wide['subject'].str.startswith(prefix).sum()})")
                for prefix, retest in RETEST.items()]
    handles += [Line2D([], [], ls='', marker='o', mfc=fill, mec='gray', label=ses)
                for ses, fill in [('ses-A', 'gray'), ('ses-B', 'white')]]
    ax1.legend(handles=handles, loc='lower right', ncol=3, frameon=False, fontsize=FONT_SIZE - 4)
    ax1.set_xlim(0.5, len(df_wide) + 0.5)
    ax1.set_ylim(0, 13)
    ax1.set_xlabel(f'Subject (n = {len(df_wide)}), sorted by C{FIRST_LEVEL}–C{LAST_LEVEL} length', fontsize=FONT_SIZE)
    ax1.set_ylabel(f'{METRIC_LABEL} [cm]', fontsize=FONT_SIZE)

    # B: test-retest; same colors and markers as in A, statistics separately for the sessions on the same day and on
    # separate days
    df_both = df_wide.dropna(subset=['ses-A', 'ses-B'])
    lims = [np.floor(df_both[['ses-A', 'ses-B']].min().min() / 10) - 0.5,
            np.ceil(df_both[['ses-A', 'ses-B']].max().max() / 10) + 0.5]
    stats_text = []
    for prefix, retest in RETEST.items():
        d_retest = df_both[df_both['subject'].str.startswith(prefix)]
        for sex, style in SEX_STYLE.items():
            d = d_retest[d_retest['sex'] == sex]
            ax2.scatter(d['ses-A'] / 10, d['ses-B'] / 10, color=style['color'], marker=retest['marker'], s=25,
                        zorder=3)
        diff = d_retest['ses-B'] - d_retest['ses-A']
        rho, p = stats.spearmanr(d_retest['ses-A'], d_retest['ses-B'])
        print(f"Test-retest, {retest['label']} ({prefix}*): n = {len(d_retest)}, Spearman rho = {rho:.2f}, "
              f"p = {p:.2g}, ses-B - ses-A = {diff.mean():.1f} ± {diff.std():.1f} mm")
        p_text = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        stats_text.append(f"{retest['label'].capitalize()} (n = {len(d_retest)}): ρ = {rho:.2f}, {p_text}\n"
                          f"  ses-B – ses-A: {diff.mean():.1f} ± {diff.std():.1f} mm")
    ax2.plot(lims, lims, color='gray', ls='--', lw=1, zorder=1)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect('equal')
    ax2.text(0.03, 0.97, 'Spearman correlation\n' + '\n'.join(stats_text), transform=ax2.transAxes, va='top',
             fontsize=FONT_SIZE - 5)
    ax2.set_xlabel('ses-A [cm]', fontsize=FONT_SIZE)
    ax2.set_ylabel('ses-B [cm]', fontsize=FONT_SIZE)
    ax2.legend(handles=handles[:4], loc='lower right', frameon=False, fontsize=FONT_SIZE - 5)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    fname_figure = os.path.join(dir_out, f'figure_cervical_{METRIC_NAME}.png')
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=f'Compute the cervical spinal cord length (top of C{FIRST_LEVEL} to '
                                                 f'bottom of C{LAST_LEVEL}) along the spinal cord centerline from the '
                                                 f'spinal cord and rootlets segmentations.')
    parser.add_argument('-i', default=INPUT_DIR,
                        help=f'Folder with the segmentations (sub-*/ses-*/*_T2w_seg.nii.gz and *_T2w_rootlets*.nii.gz). '
                             f'Default: {INPUT_DIR}')
    parser.add_argument('-o', default=OUTPUT_DIR,
                        help=f'Output folder for the CSV file and the figure. Default: {OUTPUT_DIR}')
    parser.add_argument('-participants', default=PARTICIPANTS_CSV,
                        help=f'CSV file with the subject sex and age. Default: {PARTICIPANTS_CSV}')
    parser.add_argument('-dilate', type=int, default=3,
                        help='Size of the spinal cord segmentation dilation in voxels used to intersect it with the '
                             'rootlets (as in ../rootlets_to_spinal_levels.py). Default: 3.')
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline',
                        help='Algorithm for centerline fitting (as in sct_process_segmentation). Default: bspline.')
    parser.add_argument('-centerline-smooth', type=int, default=30,
                        help='Degree of smoothing for centerline fitting, only for bspline and linear (as in '
                             'sct_process_segmentation). Default: 30.')
    parser.add_argument('-plot-only', action='store_true',
                        help='Only regenerate the figure from the CSV file saved in the -o folder by a previous run '
                             '(no length computation).')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    fname_csv = os.path.join(dir_out, f'cervical_{METRIC_NAME}.csv')

    if args.plot_only:
        generate_figure(pd.read_csv(fname_csv), dir_out)
        return

    dir_qc = os.path.join(dir_out, 'qc')
    os.makedirs(dir_qc, exist_ok=True)
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    # Compute the length for each subject and session
    rows = []
    for fname_seg in sorted(glob.glob(os.path.join(dir_path, 'sub-*', 'ses-*', '*_T2w_seg.nii.gz'))):
        # The rootlets file is named either *_T2w_rootlets.nii.gz or *_T2w_rootlets_labels.nii.gz
        fname_rootlets = glob.glob(fname_seg.replace('_seg.nii.gz', '_rootlets*.nii.gz'))
        subject, session = re.search(r'(sub-[^_]+)_(ses-[^_]+)', os.path.basename(fname_seg)).groups()
        print(f'{subject} {session}')
        if len(fname_rootlets) != 1:
            print(f'  WARNING: expected one rootlets file, found {len(fname_rootlets)}; skipping.')
            continue
        rows.append({'subject': subject, 'session': session,
                     **compute_length(fname_seg, fname_rootlets[0], param_centerline, args.dilate, dir_qc)})
    if not rows:
        raise FileNotFoundError(f'No sub-*/ses-*/*_T2w_seg.nii.gz files found in {dir_path}. Check the -i path.')
    df = pd.DataFrame(rows)

    # Keep only the subjects with a valid length in both sessions
    n_sessions = df.dropna(subset=[f'{METRIC_NAME}_mm']).groupby('subject')['session'].nunique()
    excluded = sorted(n_sessions.index[n_sessions != 2].union(set(df['subject']) - set(n_sessions.index)))
    if excluded:
        print(f'Excluded (no valid length in both sessions): {excluded}')
    df = df[~df['subject'].isin(excluded)]

    # Add the subject sex and age
    df = add_sex_age(df, args.participants)

    # Save the lengths, with sex and age, to a CSV file
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Summary per session and sex
    col = f'{METRIC_NAME}_mm'
    print(df.groupby(['session', 'sex'])[col].describe().round(1))

    # Plot the length per subject and the test-retest agreement
    generate_figure(df, dir_out)


if __name__ == '__main__':
    main()
