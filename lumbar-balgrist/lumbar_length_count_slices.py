"""
TL;DR: Compute a lumbar spinal cord length by counting axial slices and multiplying by slice thickness.

This script computes a lumbar spinal cord length from labeled spinal cord segmentations. Two metrics are available
(-metric); both count slices along the superior-inferior (S-I) axis and multiply by the slice thickness (S-I voxel
size), assuming that the spinal cord is approximately straight:
    - top-bottom (default): length from the top of L1 to the bottom of L5 = (most superior slice of L1 - most inferior
      slice of L5 + 1) x slice thickness, i.e., both end slices included
    - midpoints: distance between the L1 and S1 midpoints = (center of mass slice of L1 - center of mass slice of S1)
      x slice thickness

TODO: include also the centerline-based length to account for spinal cord curvature

Input:
- folder with manually labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz) with one value per
spinal level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=S1, 7=S2.

Output (saved to the -o folder, default: ~/data/spinal-levels/lumbar_Balgrist/results_and_figures;
<metric> is length_L1-L5 or midpoints_distance_L1-S1):
    - lumbar_<metric>.csv: length for each subject, the min slice, max slice and midpoint slice of each spinal
      level, and the subject sex and height
    - figure_lumbar_<metric>.png: the same length plotted per subject

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python lumbar_length_count_slices.py
    python lumbar_length_count_slices.py -metric midpoints

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

from spinalcordtoolbox.image import Image

# Metric definitions, Excel file with the subject sex and height, and plotting shared by the lumbar length scripts
from utils import LEVELS, METRICS, INPUT_DIR, OUTPUT_DIR, PARTICIPANTS_XLSX, add_sex_height, generate_figure


def compute_length(fname, metric_name):
    """
    Compute the length (in mm) for one segmentation.
    :param fname: path to the labeled spinal cord segmentation
    :param metric_name: key of METRICS
    :return: dict with the length in mm (NaN if one of the levels is missing) and, for each level, its min slice,
    max slice and midpoint slice (NaN if the level is missing)
    """
    metric = METRICS[metric_name]
    first, last = LEVELS[metric['first']], LEVELS[metric['last']]
    # Load the image and reorient it to RPI, so that the third axis is the S-I axis (slice 0 is the most inferior)
    img = Image(fname).change_orientation('RPI')
    data = img.data
    slice_thickness = img.dim[6]  # pz, slice thickness in mm

    # Min (most inferior) slice, max (most superior) slice and center of mass of each spinal level along the S-I axis
    row = {f"{metric['name']}_mm": np.nan}
    for value, name in LEVELS.items():
        z = np.where(data == value)[2]  # S-I index of each voxel of this level
        if len(z) == 0:
            print(f'  {name}: not found')
            row.update({f'{name}_slice_min': np.nan, f'{name}_slice_max': np.nan, f'{name}_midpoint_slice': np.nan})
            continue
        row.update({f'{name}_slice_min': z.min(), f'{name}_slice_max': z.max(), f'{name}_midpoint_slice': z.mean()})
        print(f'  {name}: midpoint at slice {z.mean():.1f} (slices {z.min()}-{z.max()})')

    if np.isnan(row[f'{first}_slice_min']) or np.isnan(row[f'{last}_slice_min']):
        print(f'  WARNING: {first} or {last} missing; the length will be NaN.')
        return row

    if metric_name == 'top-bottom':
        # Number of slices from the top of L1 to the bottom of L5 (both included)
        n_slices = row[f'{first}_slice_max'] - row[f'{last}_slice_min'] + 1
    else:
        # Number of slices between the L1 and S1 midpoints
        n_slices = row[f'{first}_midpoint_slice'] - row[f'{last}_midpoint_slice']
    row[f"{metric['name']}_mm"] = n_slices * slice_thickness
    print(f"  {metric['label']}: {n_slices:.1f} slices x {slice_thickness:.2f} mm = "
          f"{n_slices * slice_thickness:.1f} mm")

    return row


def main():
    parser = argparse.ArgumentParser(description='Compute a lumbar spinal cord length from labeled spinal cord '
                                                 'segmentations by counting axial slices.')
    parser.add_argument('-i', default=INPUT_DIR,
                        help='Folder with labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz). '
                             f'Default: {INPUT_DIR}')
    parser.add_argument('-o', default=OUTPUT_DIR,
                        help=f'Output folder for the CSV file and the figure. Default: {OUTPUT_DIR}')
    parser.add_argument('-metric', choices=list(METRICS), default='top-bottom',
                        help='top-bottom: length from the top of L1 to the bottom of L5; midpoints: distance between '
                             'the L1 and S1 midpoints. Default: top-bottom.')
    parser.add_argument('-xlsx', default=PARTICIPANTS_XLSX,
                        help=f'Excel file with the subject sex and height (used to color and sort the points). '
                             f'Default: {PARTICIPANTS_XLSX}')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    os.makedirs(dir_out, exist_ok=True)
    metric = METRICS[args.metric]

    # Compute the length for each subject
    rows = []
    for fname in sorted(glob.glob(os.path.join(dir_path, '**', '*_spinallevels_dseg.nii.gz'), recursive=True)):
        subject = os.path.basename(fname).split('_')[0]
        print(subject)
        rows.append({'subject': subject, **compute_length(fname, args.metric)})
    if not rows:
        raise FileNotFoundError(f'No *_spinallevels_dseg.nii.gz files found in {dir_path}. Check the -i path.')
    df = pd.DataFrame(rows)

    # Add the subject sex and height (also used to color and sort the subjects in the figure)
    df = add_sex_height(df, args.xlsx)

    # Save the lengths, with sex and height, to a CSV file
    fname_csv = os.path.join(dir_out, f"lumbar_{metric['name']}.csv")
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Plot the length per subject
    generate_figure(df, dir_out, metric)


if __name__ == '__main__':
    main()
