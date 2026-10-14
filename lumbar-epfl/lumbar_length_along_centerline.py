"""
TL;DR: Compute a lumbar spinal cord length along the centerline to account for the curvature of the spinal cord.

This script computes a lumbar spinal cord length from spinal level masks saved one level per file
(sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz, e.g. desc-L1 ... desc-S2), and plots:
    - the length per subject
    - the correlation between the length and subject height

NOTE:
Unlike lumbar_length_count_slices.py, which multiplies the number of axial slices by the slice thickness (i.e.,
assumes a straight spinal cord), this script measures the length along the spinal cord centerline, which accounts
for the spinal cord curvature.

    1. The centerline is fitted to the whole spinal cord mask (*_label-SC_mask.nii.gz) using SCT's get_centerline()
       with ParamCenterline (default: bspline, smooth=30, as in sct_process_segmentation). The centerline has one point
       per axial slice and is converted to physical space (mm).
    2. The two end points are found on the axial slices as in lumbar_length_count_slices.py (-metric):
        - top-bottom (default): from the top edge of the most superior slice of L1 to the bottom edge of the most
          inferior slice of L5 (i.e., both end slices included)
        - midpoints: from the center of mass slice of L1 to the center of mass slice of S1
    3. The length is the arc length of the centerline between the two end slices (cumulative length of the centerline
       segments, computed by SCT's Centerline class, linearly interpolated at the end slices).

Input:
- folder with the spinal level masks (one level per file) and the whole spinal cord mask (*_label-SC_mask.nii.gz)
- participants.tsv with the subject sex and height (default: participants.tsv one level above the input folder)

Output (saved to the current working directory; <metric> is e.g. length_L1-L5 or midpoints_distance_L1-S1):
    - lumbar_<metric>_along_centerline.csv: length along the centerline per subject, with sex and height
    - figure_lumbar_<metric>_along_centerline_masks.png: length per subject, colored by sex
    - figure_correlation_lumbar_<metric>_along_centerline_masks_height.png: length vs. height with a linear fit (visual
      guide), its 95% confidence band, and the Spearman correlation coefficient with its 95% CI
    - centerline/<subject>_<session>_label-SC_centerline.nii.gz: fitted centerline for visual QC, e.g.:
        ff <T2w> centerline/<subject>_<session>_label-SC_centerline.nii.gz

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python lumbar_length_along_centerline.py -i ~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks
    python lumbar_length_along_centerline.py -i ~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks -metric midpoints

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

# Metric definitions, statistics and plotting shared by the lumbar length scripts
from utils import METRICS, generate_length_figure, generate_correlation_figure


def get_level_slice(fname, point):
    """
    Get a slice position of a spinal level mask along the superior-inferior (S-I) axis, in RPI slice index.
    The top and bottom positions are at the edge of the end slice (+/- half a slice), so that both end slices are
    included in the length.
    :param fname: path to the mask
    :param point: 'top': top edge of the most superior slice; 'bottom': bottom edge of the most inferior slice;
    'center': center of mass along the S-I axis
    :return: slice position (float, in slices)
    """
    # Reorient to RPI so that the third axis is the S-I axis (slice 0 is the most inferior)
    z = np.where(Image(fname).change_orientation('RPI').data > 0)[2]
    return {'top': z.max() + 0.5, 'bottom': z.min() - 0.5, 'center': z.mean()}[point]


def fit_centerline(fname_seg, param_centerline, fname_out=None):
    """
    Fit the spinal cord centerline to the spinal cord mask and convert it to physical space.
    :param fname_seg: path to the whole spinal cord mask
    :param param_centerline: ParamCenterline
    :param fname_out: if provided, the centerline image is saved there (for visual QC)
    :return: Centerline (SCT) in physical space (mm), and the RPI slice index of each centerline point
    """
    im_seg = Image(fname_seg).change_orientation('RPI')
    # space='pix' keeps the slice index of each point (third row of arr_ctl); the points are then converted to
    # physical space so that the length is in mm and the voxel size and image obliquity are accounted for
    im_ctl, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=param_centerline, verbose=0, space='pix')
    points = im_seg.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_seg.transfo_pix2phys(arr_ctl_der.T, mode='relative')
    if fname_out:
        im_ctl.change_orientation(Image(fname_seg).orientation).save(fname_out, mutable=True, verbose=0)
    return Centerline(*points.T, *deriv.T), arr_ctl[2]


def compute_length(dir_path, metric, param_centerline):
    """
    Compute the length along the centerline for each subject.
    :param dir_path: folder with the spinal level masks (one level per file) and the whole spinal cord mask
    :param metric: dict from METRICS
    :param param_centerline: ParamCenterline
    :return: dataframe with 'subject' and '<metric name>_along_centerline_mm' columns
    """
    first, last = metric['first'], metric['last']
    dir_qc = os.path.join(os.getcwd(), 'centerline')
    os.makedirs(dir_qc, exist_ok=True)
    rows = []
    for fname_first in sorted(glob.glob(os.path.join(dir_path, '**', f'*_desc-{first}_mask.nii.gz'), recursive=True)):
        fname_last = fname_first.replace(f'_desc-{first}_mask', f'_desc-{last}_mask')
        fname_seg = fname_first.replace(f'_desc-{first}_mask', '_mask')
        prefix = os.path.basename(fname_first).split('_desc-')[0]   # e.g. sub-01_ses-ankle_label-SC
        subject = prefix.split('_')[0]
        missing = [f for f in (fname_last, fname_seg) if not os.path.exists(f)]
        if missing:
            print(f'WARNING: {subject}: {", ".join(map(os.path.basename, missing))} not found; skipping.')
            continue

        # Centerline in physical space; cumulative length (mm) at each centerline point, from the most inferior one
        centerline, z_ctl = fit_centerline(fname_seg, param_centerline,
                                           fname_out=os.path.join(dir_qc, f'{prefix}_centerline.nii.gz'))
        length_at_slice = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')

        z_first = get_level_slice(fname_first, metric['point_first'])
        z_last = get_level_slice(fname_last, metric['point_last'])
        length = float(length_at_slice(z_first) - length_at_slice(z_last))
        print(f"{subject}: {metric['label']} = {length:.1f} mm along the centerline")
        rows.append({'subject': subject, f"{metric['name']}_along_centerline_mm": length})
    if not rows:
        raise FileNotFoundError(f'No *_desc-{first}_mask.nii.gz files found in {dir_path}. Check the -i path.')
    print(f'Centerlines saved to {dir_qc}')
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Compute and plot a lumbar spinal cord length along the spinal cord '
                                                 'centerline from spinal level masks saved one level per file.')
    parser.add_argument('-i', required=True,
                        help='Folder with the spinal level masks (sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz)'
                             ' and the whole spinal cord mask (*_label-SC_mask.nii.gz).')
    parser.add_argument('-metric', choices=list(METRICS), default='top-bottom',
                        help='top-bottom: length from the top of L1 to the bottom of L5; midpoints: distance between '
                             'the L1 and S1 midpoints. Default: top-bottom.')
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline',
                        help='Algorithm for centerline fitting (as in sct_process_segmentation). Default: bspline.')
    parser.add_argument('-centerline-smooth', type=int, default=30,
                        help='Degree of smoothing for centerline fitting, only for bspline and linear (as in '
                             'sct_process_segmentation). Default: 30.')
    parser.add_argument('-participants', required=False,
                        help='participants.tsv with the subject sex and height. Default: participants.tsv one level '
                             'above the input folder.')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    fname_participants = args.participants or os.path.join(os.path.dirname(dir_path), 'participants.tsv')
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    # Compute the length along the centerline for each subject
    df = compute_length(dir_path, METRICS[args.metric], param_centerline)

    # Plot the length along the centerline: same figures as the straight version, with '_centerline' in the
    # file names and '(centerline)' in the axis labels
    metric = dict(METRICS[args.metric])
    metric['name'] = f"{metric['name']}_along_centerline"
    metric['label'] = f"{metric['label']}"

    # Add the subject sex and height
    df_participants = pd.read_csv(fname_participants, sep='\t')
    df = df.merge(df_participants[['participant_id', 'sex', 'height']], how='left', left_on='subject',
                  right_on='participant_id').drop(columns='participant_id')
    df['sex'] = df['sex'].str.upper()

    # Save the lengths to a CSV file
    fname_csv = os.path.join(os.getcwd(), f"lumbar_{metric['name']}.csv")
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Plot the length per subject and its correlation with height (only subjects with known height)
    generate_length_figure(df, metric)
    generate_correlation_figure(df.dropna(subset=['height']).reset_index(drop=True), metric)
    plt.show()


if __name__ == '__main__':
    main()
