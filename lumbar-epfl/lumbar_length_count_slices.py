"""
TL;DR: Compute a lumbar spinal cord length by counting axial slices and multiplying by slice thickness.

This script computes a lumbar spinal cord length from spinal level masks saved one level per file
(sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz, e.g. desc-L1 ... desc-S2), and plots:
    - the length per subject
    - the correlation between the length and subject height

NOTE:
Axial slices are counted along the superior-inferior (S-I) axis and multiplied by the slice thickness (S-I voxel size),
i.e., the same logic as ../lumbar-balgrist/lumbar_length_count_slices.py (Balgrist data, same -metric options),
assuming that the spinal cord is approximately straight:

Two metrics are available (argument -metric);
    - top-bottom (default): length from the top of L1 to the bottom of L5 = (most superior slice of L1 - most inferior
      slice of L5 + 1) x slice thickness, i.e., both end slices included
    - midpoints: distance between the L1 and S1 midpoints = (center of mass slice of L1 - center of mass slice of S1)
      x slice thickness

Input:
- folder with the spinal level masks (one level per file)
- participants.tsv with the subject sex and height (default: participants.tsv one level above the input folder)

Output (saved to the current working directory; <metric> is e.g. length_L1-L5 or midpoints_distance_L1-S1):
    - lumbar_<metric>_masks.csv: length per subject, with sex and height
    - figure_lumbar_<metric>_masks.png: length per subject, colored by sex
    - figure_correlation_lumbar_<metric>_masks_height.png: length vs. height with a linear fit (visual guide), its 95%
      confidence band, and the Spearman correlation coefficient with its 95% CI

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python lumbar_length_count_slices.py -i ~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks
    python lumbar_length_count_slices.py -i ~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks -metric midpoints

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spinalcordtoolbox.image import Image

# Metric definitions, statistics and plotting shared by the lumbar length scripts
from utils import METRICS, generate_length_figure, generate_correlation_figure


def get_slice(fname, point):
    """
    Get a slice position of a binary mask along the superior-inferior (S-I) axis.
    :param fname: path to the mask
    :param point: 'top': most superior slice; 'bottom': most inferior slice; 'center': center of mass along the S-I axis
    :return: slice position (in slices) and slice thickness (S-I voxel size, in mm)
    """
    # Reorient to RPI so that the third axis is the S-I axis (slice 0 is the most inferior)
    img = Image(fname).change_orientation('RPI')
    z = np.where(img.data > 0)[2]   # S-I index of each voxel of the mask
    position = {'top': z.max(), 'bottom': z.min(), 'center': z.mean()}[point]
    return position, img.dim[6]


def compute_length(dir_path, metric):
    """
    Compute the length for each subject: number of slices between the two points x slice thickness.
    :param dir_path: folder with the spinal level masks (one level per file)
    :param metric: dict from METRICS
    :return: dataframe with 'subject' and '<metric name>_mm' columns
    """
    first, last = metric['first'], metric['last']
    rows = []
    for fname_first in sorted(glob.glob(os.path.join(dir_path, '**', f'*_desc-{first}_mask.nii.gz'), recursive=True)):
        fname_last = fname_first.replace(f'_desc-{first}_mask', f'_desc-{last}_mask')
        subject = os.path.basename(fname_first).split('_')[0]
        if not os.path.exists(fname_last):
            print(f'WARNING: {subject}: {last} mask not found; skipping.')
            continue
        slice_first, slice_thickness = get_slice(fname_first, metric['point_first'])
        slice_last, _ = get_slice(fname_last, metric['point_last'])
        n_slices = slice_first - slice_last + (1 if metric['add_slice'] else 0)
        length = n_slices * slice_thickness
        print(f"{subject}: {metric['label']} = {n_slices:.1f} slices x {slice_thickness:.2f} mm = {length:.1f} mm")
        rows.append({'subject': subject, f"{metric['name']}_mm": length})
    if not rows:
        raise FileNotFoundError(f'No *_desc-{first}_mask.nii.gz files found in {dir_path}. Check the -i path.')
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Compute and plot a lumbar spinal cord length from spinal level masks '
                                                 'saved one level per file.')
    parser.add_argument('-i', required=True,
                        help='Folder with the spinal level masks '
                             '(sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz).')
    parser.add_argument('-metric', choices=list(METRICS), default='top-bottom',
                        help='top-bottom: length from the top of L1 to the bottom of L5; midpoints: distance between '
                             'the L1 and S1 midpoints. Default: top-bottom.')
    parser.add_argument('-participants', required=False,
                        help='participants.tsv with the subject sex and height. Default: participants.tsv one level '
                             'above the input folder.')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    fname_participants = args.participants or os.path.join(os.path.dirname(dir_path), 'participants.tsv')
    metric = f'{METRICS[args.metric]}_slice_count'

    # Compute the length for each subject
    df = compute_length(dir_path, metric)

    # Add the subject sex and height
    df_participants = pd.read_csv(fname_participants, sep='\t')
    df = df.merge(df_participants[['participant_id', 'sex', 'height']], how='left', left_on='subject',
                  right_on='participant_id').drop(columns='participant_id')
    df['sex'] = df['sex'].str.upper()

    # Save the lengths to a CSV file
    fname_csv = os.path.join(os.getcwd(), f"lumbar_{metric['name']}_masks.csv")
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Plot the distance per subject and its correlation with height (only subjects with known height)
    generate_length_figure(df, metric)
    generate_correlation_figure(df.dropna(subset=['height']).reset_index(drop=True), metric)
    plt.show()


if __name__ == '__main__':
    main()
