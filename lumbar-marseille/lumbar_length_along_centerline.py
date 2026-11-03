"""
TL;DR: Compute the lumbar spinal cord length from the top of L1 to the bottom of L5 along the spinal cord centerline
for the lumbar-marseille dataset.

Same measurement as ../lumbar-epfl/lumbar_length_along_centerline.py (-metric top-bottom), so the numbers are
comparable to the other lumbar datasets:
    1. The centerline is fitted to the whole spinal cord segmentation (derivatives/labels/*_sc_seg.nii.gz) using SCT's
       get_centerline() with ParamCenterline (default: bspline, smooth=30, as in sct_process_segmentation), and
       converted to physical space (mm).
    2. The end points are taken from the spinal levels (rootlets_to_spinal_levels.py output, L1 = 21, L5 = 25):
       top edge of the most superior L1 slice and bottom edge of the most inferior L5 slice (both end slices included).
    3. The length is the arc length of the centerline between the two end points.

Input:
    -i    : folder with the rootlets_to_spinal_levels.py outputs (*_label-rootlets_dseg_spinal_levels.nii.gz)
    -seg  : derivatives/labels folder with the spinal cord segmentations (*_sc_seg.nii.gz)
    -participants : participants.tsv; sex and height are added when available (columns 'sex' and 'height')

Output (in -i):
    lumbar_length_L1-L5_along_centerline.csv : subject, length_L1-L5_along_centerline_mm, sex, height
    centerline/<subject>_..._centerline.nii.gz : fitted centerline for visual QC

The script requires the SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh && conda activate venv_sct

Example:
    python lumbar_length_along_centerline.py -i ~/data/spinal-levels/lumbar-marseille

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

DATA_DIR = os.path.expanduser('~/data/data.neuro.polymtl.ca/lumbar-marseille')
L1, L5 = 21, 25     # SCT spinal level values
LEVELS_SUFFIX = '_label-rootlets_dseg_spinal_levels.nii.gz'
SEG_SUFFIX = '_sc_seg.nii.gz'


def fit_centerline(fname_seg, param_centerline, fname_out=None):
    """
    Fit the centerline to the spinal cord segmentation (as in ../lumbar-epfl/lumbar_length_along_centerline.py).
    :return: Centerline (SCT) in physical space (mm), and the RPI slice index of each centerline point
    """
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_ctl, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=param_centerline, verbose=0, space='pix')
    points = im_seg.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_seg.transfo_pix2phys(arr_ctl_der.T, mode='relative')
    if fname_out:
        im_ctl.change_orientation(Image(fname_seg).orientation).save(fname_out, mutable=True, verbose=0)
    return Centerline(*points.T, *deriv.T), arr_ctl[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, help='Folder with the *' + LEVELS_SUFFIX + ' files.')
    parser.add_argument('-seg', default=os.path.join(DATA_DIR, 'derivatives', 'labels'),
                        help='Folder with the spinal cord segmentations. Default: %(default)s')
    parser.add_argument('-participants', default=os.path.join(DATA_DIR, 'participants.tsv'),
                        help='participants.tsv. Default: %(default)s')
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline')
    parser.add_argument('-centerline-smooth', type=int, default=30)
    args = parser.parse_args()
    dir_in = os.path.abspath(os.path.expanduser(args.i))
    dir_qc = os.path.join(dir_in, 'centerline')
    os.makedirs(dir_qc, exist_ok=True)
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    rows = []
    for fname_levels in sorted(glob.glob(os.path.join(dir_in, '*' + LEVELS_SUFFIX))):
        prefix = os.path.basename(fname_levels).replace(LEVELS_SUFFIX, '')   # e.g. sub-CTS04_ses-SPpre_acq-ax_T2w
        subject = prefix.split('_')[0]
        fname_seg = glob.glob(os.path.join(os.path.expanduser(args.seg), subject, '**', prefix + SEG_SUFFIX),
                              recursive=True)
        if not fname_seg:
            print(f'WARNING: {subject}: {prefix + SEG_SUFFIX} not found; skipping.')
            continue

        # Top edge of the most superior L1 slice and bottom edge of the most inferior L5 slice (RPI slice index)
        levels = Image(fname_levels).change_orientation('RPI').data.round()
        z_l1, z_l5 = np.where(levels == L1)[2], np.where(levels == L5)[2]
        if not len(z_l1) or not len(z_l5):
            print(f'WARNING: {subject}: L1 or L5 missing in {os.path.basename(fname_levels)}; skipping.')
            continue
        z_top, z_bottom = z_l1.max() + 0.5, z_l5.min() - 0.5

        centerline, z_ctl = fit_centerline(fname_seg[0], param_centerline,
                                           fname_out=os.path.join(dir_qc, f'{prefix}_centerline.nii.gz'))
        length_at_slice = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
        length = float(length_at_slice(z_top) - length_at_slice(z_bottom))
        print(f'{subject}: length top L1 – bottom L5 = {length:.1f} mm along the centerline')
        rows.append({'subject': subject, 'length_L1-L5_along_centerline_mm': length})
    df = pd.DataFrame(rows)

    # Sex and height (NaN when not in participants.tsv)
    df_participants = pd.read_csv(os.path.expanduser(args.participants), sep='\t')
    for col in ('sex', 'height'):
        if col not in df_participants:
            print(f'WARNING: no "{col}" column in {args.participants}')
            df_participants[col] = np.nan
    df = df.merge(df_participants[['participant_id', 'sex', 'height']], how='left', left_on='subject',
                  right_on='participant_id').drop(columns='participant_id')

    fname_csv = os.path.join(dir_in, 'lumbar_length_L1-L5_along_centerline.csv')
    df.to_csv(fname_csv, index=False)
    print(df.to_string())
    print(f'CSV file saved in {fname_csv}. Centerlines saved in {dir_qc}.')


if __name__ == '__main__':
    main()
