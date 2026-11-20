"""
TL;DR: Compute the Balgrist sacral spinal cord length (top of S1 to the conus tip) along the spinal cord centerline,
directly on the axial CISS image (no localizer).

Unlike sacral_length_along_centerline.py (which measures along the whole-spine localizer centerline), everything is
measured on the CISS:

    1. Centerline. Fitted to the existing cord segmentation of the CISS (-seg) with SCT's get_centerline() (bspline,
       smooth=30, as in sct_process_segmentation and ../lumbar-balgrist/lumbar_length_along_centerline.py).
    2. Bounds.
         - top of S1: top edge of the most superior S1 (value 6) slice of the spinal-level segmentation
                      (*_spinallevels_dseg.nii.gz);
         - conus:     two versions, giving two lengths:
                        - auto:      bottom edge of the most inferior slice of the cord segmentation (cord tip);
                        - conus CSV: conus world-Z read on the CISS (-conus-csv), as for the whole-cord length.
    3. Length. Difference of the cumulative centerline length (Centerline.incremental_length) at the two bounds.

Input:
    -seg   : folder with the cord segmentations of the CISS (*_acq-ciss_T2w<-seg-suffix>.nii.gz, default suffix '_seg',
             searched recursively)
    -dseg  : folder with the spinal-level segmentations (<CISS name>_spinallevels_dseg.nii.gz, searched recursively).
             Default: same as -seg.
    -xlsx  : Excel file with the subject sex and height ('Spinal level heights (mm)' sheet)
    -conus-csv : CSV file with the conus world-Z read on the CISS (columns 'subject' (number, e.g. 3 for sub-ltr03)
                 and 'world_z_mm_ciss_tip'), as used for the whole-cord length (../lumbar-balgrist/snap_conus.py)

Output (-o):
    - sacral_length_S1-conus_CISS.csv: per subject, sex, height, the two lengths and QC columns:
        - length_S1-conus_centerline_mm: top of S1 to the conus from -conus-csv (same column name as the other
          sacral scripts, so correlation_length_height.py and ../sacral-combined/combine_sacral_lengths.py use it)
        - length_S1-conus_auto_centerline_mm: top of S1 to the bottom of the cord segmentation (purely automatic)
        - extrap_mm: how far the conus from -conus-csv lies below the cord segmentation (the length is extrapolated
          there); re-check if > 3 mm
        - bottom_area_mm2: cord area on the most inferior slice of the cord segmentation (should be small, i.e. the
          conus tip)
        - conus_csv_vs_seg_mm: conus from -conus-csv minus the bottom of the cord segmentation (S-I, mm); large values
          mean the segmentation stops too early (< 0) or too late (> 0) compared with the manually read conus
    - centerline/<CISS name><-seg-suffix>_centerline.nii.gz: fitted centerline for QC

QC: open the CISS with the cord segmentation and the centerline, e.g.:
    ff sub-ltr03_ses-03_acq-ciss_T2w.nii.gz sub-ltr03_ses-03_acq-ciss_T2w_seg.nii.gz \
        centerline/sub-ltr03_ses-03_acq-ciss_T2w_seg_centerline.nii.gz &

The script requires the SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python sacral_length_ciss.py \
        -seg ~/data/spinal-levels/lumbar_Balgrist/bids/derivatives/labels \
        -xlsx ~/data/spinal-levels/lumbar_Balgrist/bids/spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx \
        -conus-csv tip_conus_CISS_worldZ.csv \
        -o ~/data/spinal-levels/sacral_Balgrist_CISS
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

S1_VALUE = 6
LENGTH_COL = 'length_S1-conus_centerline_mm'                # conus CSV; same column name as the other sacral scripts
LENGTH_AUTO_COL = 'length_S1-conus_auto_centerline_mm'      # bottom of the cord segmentation
PARTICIPANTS_SHEET = 'Spinal level heights (mm)'    # sheet with the 'sex' and 'Height (cm)' columns


def subject_id(fname):
    """sub-ltrNN from a BIDS file name."""
    return re.search(r'(sub-[a-zA-Z0-9]+)', os.path.basename(fname)).group(1)


def find_one(folder, fname):
    """First file named fname under folder (recursive), or None."""
    found = sorted(glob.glob(os.path.join(folder, '**', fname), recursive=True))
    return found[0] if found else None


def s1_top_phys(fname_dseg):
    """Physical point (mm) at the top edge of the most superior S1 slice of the spinal-level segmentation."""
    im = Image(fname_dseg).change_orientation('RPI')
    dseg = np.round(im.data).astype(int)
    if not (dseg == S1_VALUE).any():
        raise ValueError(f'S1 ({S1_VALUE}) missing in {fname_dseg}')
    k = np.where((dseg == S1_VALUE).any((0, 1)))[0].max()
    ii, jj = np.nonzero(dseg[:, :, k] == S1_VALUE)
    return im.transfo_pix2phys([[ii.mean(), jj.mean(), k + 0.5]])[0]


def read_conus_csv(fname):
    """{sub-ltrNN: conus world-Z (mm)} from the CSV with the 'subject' (number) and 'world_z_mm_ciss_tip' columns."""
    df = pd.read_csv(fname)
    return {f"sub-ltr{int(r['subject']):02d}": float(r['world_z_mm_ciss_tip']) for _, r in df.iterrows()}


def add_sex_height(df, fname_xlsx):
    """Add the subject sex ('F'/'M') and height (cm) from the participants Excel file."""
    xl = pd.read_excel(fname_xlsx, sheet_name=PARTICIPANTS_SHEET)
    xl = xl[xl['Participants'].astype(str).str.startswith('sub-')]
    xl['sex'] = xl['sex'].str.upper()
    xl = xl.rename(columns={'Height (cm)': 'height'})
    return df.merge(xl[['Participants', 'sex', 'height']], how='left', left_on='subject',
                    right_on='Participants').drop(columns='Participants')


def measure_subject(name, fname_seg, fname_dseg, dir_out, param_centerline, conus_z_csv=None):
    """Sacral length (S1 top to conus) along the CISS cord centerline for one subject. :return: dict (CSV row)"""
    subject = subject_id(name)

    im_seg = Image(fname_seg)
    orientation_native = im_seg.orientation
    im_seg.change_orientation('RPI')
    im_seg.data = (im_seg.data > 0).astype(np.uint8)
    if not im_seg.data.any():
        raise ValueError(f'empty cord segmentation {fname_seg}')
    px, py, pz = im_seg.dim[4:7]
    k_bottom = np.where(im_seg.data.any((0, 1)))[0].min()
    k_seg_bottom = k_bottom - 0.5                                     # bottom edge of the most inferior slice
    bottom_area = im_seg.data[:, :, k_bottom].sum() * px * py

    # Centerline fitted to the cord segmentation (RPI pixel space) and its cumulative length in mm
    im_ctl, arr_ctl, arr_der, _ = get_centerline(im_seg, param=param_centerline, verbose=0, space='pix')
    points = im_seg.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_seg.transfo_pix2phys(arr_der.T, mode='relative')
    centerline = Centerline(*points.T, *deriv.T)
    z_ctl = arr_ctl[2]
    length_at = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
    slice_at_world_z = interp1d(points[:, 2], z_ctl, fill_value='extrapolate')   # world-Z -> (real) slice index

    dir_ctl = os.path.join(dir_out, 'centerline')
    os.makedirs(dir_ctl, exist_ok=True)
    im_ctl.change_orientation(orientation_native).save(
        os.path.join(dir_ctl, os.path.basename(fname_seg).replace('.nii.gz', '_centerline.nii.gz')),
        mutable=True, verbose=0)

    # Top of S1: top edge of the most superior S1 slice (via physical space, so the spinal-level segmentation does not
    # need to be on the same grid as the cord segmentation)
    k_s1 = im_seg.transfo_phys2pix([s1_top_phys(fname_dseg)], real=False)[0][2]
    k_seg_top = np.where(im_seg.data.any((0, 1)))[0].max() + 0.5
    if k_s1 > k_seg_top:
        raise ValueError(f'top of S1 is {(k_s1 - k_seg_top) * pz:.1f} mm above the cord segmentation')

    # Conus: (1) bottom edge of the cord segmentation (auto), (2) conus world-Z from the conus CSV
    length_auto = float(length_at(k_s1) - length_at(k_seg_bottom))
    length_csv, extrap, conus_csv_vs_seg = np.nan, np.nan, np.nan
    if conus_z_csv is not None:
        k_conus = float(slice_at_world_z(conus_z_csv))
        length_csv = float(length_at(k_s1) - length_at(k_conus))
        extrap = max(0.0, k_seg_bottom - k_conus) * pz
        conus_csv_vs_seg = (k_conus - k_seg_bottom) * pz

    flag = '  <-- re-check' if np.nan_to_num(extrap) > 3 or abs(np.nan_to_num(conus_csv_vs_seg)) > 3 else ''
    print(f'{subject}: conus CSV {length_csv:.1f} mm, auto {length_auto:.1f} mm (bottom area {bottom_area:.1f} mm2, '
          f'extrap {extrap:.1f} mm, conus CSV vs seg {conus_csv_vs_seg:+.1f} mm){flag}')
    return {'subject': subject, LENGTH_COL: round(length_csv, 2), LENGTH_AUTO_COL: round(length_auto, 2),
            'bottom_area_mm2': round(bottom_area, 2),
            'extrap_mm': round(extrap, 2),
            'conus_csv_vs_seg_mm': round(conus_csv_vs_seg, 2),
            'filename': f'{name}.nii.gz', 'fname_seg': fname_seg}


def main():
    parser = argparse.ArgumentParser(description='Compute the Balgrist sacral cord length (top of S1 to the conus) '
                                                 'along the cord centerline on the axial CISS.')
    parser.add_argument('-seg', required=True,
                        help='Folder with the cord segmentations of the CISS (*_acq-ciss_T2w<-seg-suffix>.nii.gz, '
                             'searched recursively).')
    parser.add_argument('-seg-suffix', default='_seg',
                        help="Suffix of the cord segmentation file names. Default: '_seg'.")
    parser.add_argument('-dseg', help='Folder with the spinal-level segmentations '
                                      '(<CISS name>_spinallevels_dseg.nii.gz). Default: same as -seg.')
    parser.add_argument('-xlsx', required=True,
                        help=f"Excel file with the subject sex and height ('{PARTICIPANTS_SHEET}' sheet).")
    parser.add_argument('-o', required=True, help='Output folder.')
    parser.add_argument('-conus-csv', required=True,
                        help="CSV with the conus world-Z read on the CISS ('subject', 'world_z_mm_ciss_tip').")
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline',
                        help='Algorithm for centerline fitting (as in sct_process_segmentation). Default: bspline.')
    parser.add_argument('-centerline-smooth', type=int, default=30,
                        help='Degree of smoothing for centerline fitting (bspline and linear only). Default: 30.')
    args = parser.parse_args()

    dir_seg = os.path.abspath(os.path.expanduser(args.seg))
    dir_dseg = os.path.abspath(os.path.expanduser(args.dseg or args.seg))
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    os.makedirs(dir_out, exist_ok=True)
    conus_z = read_conus_csv(os.path.expanduser(args.conus_csv))
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    rows = []
    pattern = f'*_acq-ciss_T2w{args.seg_suffix}.nii.gz'
    for fname_seg in sorted(glob.glob(os.path.join(dir_seg, '**', pattern), recursive=True)):
        name = os.path.basename(fname_seg)[:-len(f'{args.seg_suffix}.nii.gz')]     # CISS name
        subject = subject_id(name)
        fname_dseg = find_one(dir_dseg, f'{name}_spinallevels_dseg.nii.gz')
        if not fname_dseg:
            print(f'{subject}: no spinal-level segmentation ({name}_spinallevels_dseg.nii.gz), skipped')
            continue
        if subject not in conus_z:
            print(f'{subject}: not in {args.conus_csv}, only the automatic length is computed')
        try:
            rows.append(measure_subject(name, fname_seg, fname_dseg, dir_out, param_centerline,
                                        conus_z_csv=conus_z.get(subject)))
        except ValueError as e:
            print(f'{subject}: skipped ({e})')
    if not rows:
        raise FileNotFoundError(f'No {pattern} with a matching spinal-level segmentation found in {dir_seg} '
                                f'(-dseg {dir_dseg}).')

    df = add_sex_height(pd.DataFrame(rows), os.path.expanduser(args.xlsx))
    fname_csv = os.path.join(dir_out, 'sacral_length_S1-conus_CISS.csv')
    df.to_csv(fname_csv, index=False)
    print(f'\n{len(df)} subjects. -> {fname_csv}')
    for col, label in [(LENGTH_COL, 'conus CSV'), (LENGTH_AUTO_COL, 'auto')]:
        print(f'  {label}: mean {df[col].mean():.1f} +/- {df[col].std():.1f} mm, '
              f'range {df[col].min():.1f}-{df[col].max():.1f} mm (n={df[col].notna().sum()})')
    print('Run correlation_length_height.py -i <this CSV> for the sacral length vs height figure.')


if __name__ == '__main__':
    main()
