"""
TL;DR: Compute a lumbar spinal cord length along the centerline to account for the curvature of the spinal cord.

This script computes a lumbar spinal cord length from labeled spinal cord segmentations and plots it per subject.

NOTE:
Unlike lumbar_length_count_slices.py, which multiplies the number of axial slices by the slice thickness (i.e.,
assumes a straight spinal cord), this script measures the length along the spinal cord centerline, which accounts
for the spinal cord curvature (same method as ../lumbar-epfl/lumbar_length_along_centerline.py).

    1. The centerline is fitted to the labeled spinal cord segmentation (all levels L1-S2 together, binarized) using
       SCT's get_centerline() with ParamCenterline (default: bspline, smooth=30, as in sct_process_segmentation). The
       centerline has one point per axial slice and is converted to physical space (mm).
       Note that the segmentation covers only L1-S2, so the centerline covers only L1-S2 too.
    2. The two end points are found on the axial slices as in lumbar_length_count_slices.py (-metric):
        - top-bottom (default): from the top edge of the most superior slice of L1 to the bottom edge of the most
          inferior slice of L5 (i.e., both end slices included)
        - midpoints: from the center of mass slice of L1 to the center of mass slice of S1
    3. The length is the arc length of the centerline between the two end slices (cumulative length of the centerline
       segments, computed by SCT's Centerline class, linearly interpolated at the end slices).

Input:
- folder with manually labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz) with one value per
spinal level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=S1, 7=S2.

Output (saved to the -o folder, default: ~/data/spinal-levels/lumbar_rootlets_Balgrist/results_and_figures;
<metric> is length_L1-L5 or midpoints_distance_L1-S1):
    - lumbar_<metric>_centerline.csv: length along the centerline per subject, with sex and height
    - figure_lumbar_<metric>_centerline.png: the same length plotted per subject
    - centerline/*_spinallevels_dseg_centerline.nii.gz: fitted centerline for visual QC

The correlation with height can then be plotted with correlation_length_height.py.

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python lumbar_length_along_centerline.py
    python lumbar_length_along_centerline.py -metric midpoints

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

# Metric definitions, Excel file with the subject sex and height, and plotting shared by the lumbar length scripts
from utils import METRICS, INPUT_DIR, OUTPUT_DIR, PARTICIPANTS_XLSX, add_sex_height, generate_figure


def get_level_slice(data, value, point):
    """
    Get a slice position of a spinal level along the superior-inferior (S-I) axis, in RPI slice index.
    The top and bottom positions are at the edge of the end slice (+/- half a slice), so that both end slices are
    included in the length.
    :param data: labeled segmentation data in RPI orientation
    :param value: value of the spinal level in the segmentation
    :param point: 'top': top edge of the most superior slice; 'bottom': bottom edge of the most inferior slice;
    'center': center of mass along the S-I axis
    :return: slice position (float, in slices); NaN if the level is missing
    """
    z = np.where(data == value)[2]  # S-I index of each voxel of this level
    if len(z) == 0:
        return np.nan
    return {'top': z.max() + 0.5, 'bottom': z.min() - 0.5, 'center': z.mean()}[point]


def fit_centerline(im_seg, param_centerline, fname_out=None, orientation_out='RPI'):
    """
    Fit the spinal cord centerline to the spinal cord segmentation and convert it to physical space.
    :param im_seg: labeled spinal cord segmentation (Image) in RPI orientation
    :param param_centerline: ParamCenterline
    :param fname_out: if provided, the centerline image is saved there (for visual QC)
    :param orientation_out: orientation of the saved centerline image (native orientation of the segmentation)
    :return: Centerline (SCT) in physical space (mm), and the RPI slice index of each centerline point
    """
    # Binarize the segmentation: the centerline is fitted to all levels together
    im_bin = im_seg.copy()
    im_bin.data = (im_seg.data > 0).astype(np.uint8)
    # space='pix' keeps the slice index of each point (third row of arr_ctl); the points are then converted to
    # physical space so that the length is in mm and the voxel size and image obliquity are accounted for
    im_ctl, arr_ctl, arr_ctl_der, _ = get_centerline(im_bin, param=param_centerline, verbose=0, space='pix')
    points = im_bin.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_bin.transfo_pix2phys(arr_ctl_der.T, mode='relative')
    if fname_out:
        im_ctl.change_orientation(orientation_out).save(fname_out, mutable=True, verbose=0)
    return Centerline(*points.T, *deriv.T), arr_ctl[2]


def compute_length(fname, metric_name, param_centerline, dir_qc):
    """
    Compute the length along the centerline (in mm) for one segmentation.
    :param fname: path to the labeled spinal cord segmentation
    :param metric_name: key of METRICS
    :param param_centerline: ParamCenterline
    :param dir_qc: folder where the fitted centerline is saved (for visual QC)
    :return: length in mm (NaN if one of the levels is missing)
    """
    metric = METRICS[metric_name]
    # Reorient to RPI so that the third axis is the S-I axis (slice 0 is the most inferior)
    im_seg = Image(fname).change_orientation('RPI')
    point_first, point_last = ('top', 'bottom') if metric_name == 'top-bottom' else ('center', 'center')
    z_first = get_level_slice(im_seg.data, metric['first'], point_first)
    z_last = get_level_slice(im_seg.data, metric['last'], point_last)
    if np.isnan(z_first) or np.isnan(z_last):
        print(f"  WARNING: level {metric['first']} or {metric['last']} missing; the length will be NaN.")
        return np.nan

    # Centerline in physical space; cumulative length (mm) at each centerline point, from the most inferior one
    centerline, z_ctl = fit_centerline(im_seg, param_centerline,
                                       fname_out=os.path.join(dir_qc, os.path.basename(fname).replace('.nii.gz', '_centerline.nii.gz')),
                                       orientation_out=Image(fname).orientation)
    length_at_slice = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
    length = float(length_at_slice(z_first) - length_at_slice(z_last))
    print(f"  {metric['label']}: {length:.1f} mm along the centerline")
    return length


def main():
    parser = argparse.ArgumentParser(description='Compute a lumbar spinal cord length along the spinal cord centerline '
                                                 'from labeled spinal cord segmentations.')
    parser.add_argument('-i', default=INPUT_DIR,
                        help='Folder with labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz). '
                             f'Default: {INPUT_DIR}')
    parser.add_argument('-o', default=OUTPUT_DIR,
                        help=f'Output folder for the CSV file and the figure. Default: {OUTPUT_DIR}')
    parser.add_argument('-metric', choices=list(METRICS), default='top-bottom',
                        help='top-bottom: length from the top of L1 to the bottom of L5; midpoints: distance between '
                             'the L1 and S1 midpoints. Default: top-bottom.')
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline',
                        help='Algorithm for centerline fitting (as in sct_process_segmentation). Default: bspline.')
    parser.add_argument('-centerline-smooth', type=int, default=30,
                        help='Degree of smoothing for centerline fitting, only for bspline and linear (as in '
                             'sct_process_segmentation). Default: 30.')
    parser.add_argument('-xlsx', default=PARTICIPANTS_XLSX,
                        help=f'Excel file with the subject sex and height (used to color and sort the points). '
                             f'Default: {PARTICIPANTS_XLSX}')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    dir_qc = os.path.join(dir_out, 'centerline')
    os.makedirs(dir_qc, exist_ok=True)
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    # Same outputs as the straight version, with '_centerline' in the file and column names
    metric = dict(METRICS[args.metric])
    metric['name'] = f"{metric['name']}_centerline"

    # Compute the length for each subject
    rows = []
    for fname in sorted(glob.glob(os.path.join(dir_path, '**', '*_spinallevels_dseg.nii.gz'), recursive=True)):
        subject = os.path.basename(fname).split('_')[0]
        print(subject)
        rows.append({'subject': subject,
                     f"{metric['name']}_mm": compute_length(fname, args.metric, param_centerline, dir_qc)})
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
