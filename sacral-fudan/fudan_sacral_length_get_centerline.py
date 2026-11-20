"""
Compare two centerlines for the Fudan sacral cord length (top of S1 to the conus tip):
    - centroid       : per-axial-slice centroid of the CISS cord segmentation + 10 mm moving average
                       (current method, fundan_sacral_length.py)
    - get_centerline : centerline fitted to the same cord segmentation with SCT's get_centerline()
                       (bspline, smooth=30, as in sct_process_segmentation and ../lumbar-fudan/lumbar_fudan_L1-L5_length.py)

For the get_centerline length, each manual bound (S1-top, conus) is converted from physical space to a (real-valued)
RPI slice index of the cord segmentation, and the length is the difference of the cumulative centerline length
(Centerline.incremental_length) interpolated at the two slices, as in the lumbar scripts. If a bound falls outside the
segmentation (typically the conus below the end of the cord mask), the cumulative length is linearly extrapolated;
the distance (mm) is reported in 'get_centerline_extrap_mm' (re-check if > 3 mm).

Input (-i: Fudan dataset root, same layout as fundan_sacral_length.py):
- derivatives/cord_seg_from_markers/sub-*/sub-*_cord_seg.nii.gz
- sacral_length/bounds/sub-*/[new/]sub-*_{S1top,conus}.mrk.json   (new/ preferred when present)

Output (-o, default <-i>/sacral_length):
- fudan_sacral_length_centroid_vs_get_centerline.csv: per subject, both lengths, their difference, the extrapolation
  QC columns and the bound sources
- centerline/sub-*_cord_seg_centerline.nii.gz: fitted centerline for visual QC

The script requires the SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python fudan_sacral_length_get_centerline.py -i ~/Desktop/fundan_dataset
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundan_sacral_length import (resolve_bound, mrk_pos, cord_centerline, arc_length_between,  # noqa: E402
                                  DEFAULT_ROOT, SMOOTH_MM)


def length_get_centerline(cord_file, pos_s1, pos_conus, param_centerline, dir_qc=None):
    """
    Length (mm) along the get_centerline() centerline between two physical points (RAS, mm).
    :return: (length_mm, extrap_mm)
    """
    im = Image(cord_file)
    orientation_native = im.orientation
    im.change_orientation('RPI')
    im.data = (im.data > 0).astype(np.uint8)

    im_ctl, arr_ctl, arr_der, _ = get_centerline(im, param=param_centerline, verbose=0, space='pix')
    points = im.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im.transfo_pix2phys(arr_der.T, mode='relative')
    centerline = Centerline(*points.T, *deriv.T)
    z_ctl = arr_ctl[2]
    if dir_qc:
        im_ctl.change_orientation(orientation_native).save(
            os.path.join(dir_qc, os.path.basename(cord_file).replace('.nii.gz', '_centerline.nii.gz')),
            mutable=True, verbose=0)

    # Bounds: physical space -> real-valued RPI slice index (accounts for oblique slices)
    k_s1, k_conus = im.transfo_phys2pix([pos_s1, pos_conus], real=False)[:, 2]
    length_at = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
    length = abs(float(length_at(k_s1) - length_at(k_conus)))

    k_lo, k_hi = min(k_s1, k_conus), max(k_s1, k_conus)
    extrap = (max(0.0, k_hi - z_ctl.max()) + max(0.0, z_ctl.min() - k_lo)) * im.dim[6]
    return length, float(extrap)


def main():
    ap = argparse.ArgumentParser(description='Compare the Fudan sacral cord length (S1-top to conus) along the '
                                             'centroid centerline vs. the SCT get_centerline() centerline.')
    ap.add_argument('-i', default=DEFAULT_ROOT, help=f'Fudan dataset root. Default: {DEFAULT_ROOT}')
    ap.add_argument('-o', default=None, help='Output folder. Default: <-i>/sacral_length')
    ap.add_argument('-smooth', type=int, default=SMOOTH_MM,
                    help=f'Centroid centerline moving average (mm). Default: {SMOOTH_MM}')
    ap.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline',
                    help='get_centerline() fitting algorithm. Default: bspline.')
    ap.add_argument('-centerline-smooth', type=int, default=30,
                    help='get_centerline() smoothing (bspline and linear only). Default: 30.')
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.i))
    bounds_dir = os.path.join(root, 'sacral_length', 'bounds')
    cord_dir = os.path.join(root, 'derivatives', 'cord_seg_from_markers')
    out_dir = os.path.abspath(os.path.expanduser(args.o)) if args.o else os.path.join(root, 'sacral_length')
    dir_qc = os.path.join(out_dir, 'centerline')
    os.makedirs(dir_qc, exist_ok=True)
    param_centerline = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(bounds_dir, 'sub-*'))):
        sub = os.path.basename(d)
        s1, s1src = resolve_bound(bounds_dir, sub, 'S1top')
        co, cosrc = resolve_bound(bounds_dir, sub, 'conus')
        cord = os.path.join(cord_dir, sub, f'{sub}_cord_seg.nii.gz')
        if not (s1 and co):
            print(f'  skip {sub}: missing bound')
            continue
        if not os.path.exists(cord):
            print(f'  skip {sub}: no cord_seg')
            continue
        p1, p2 = mrk_pos(s1), mrk_pos(co)

        len_centroid, extrap_centroid = arc_length_between(cord_centerline(cord), p1[2], p2[2], smooth_mm=args.smooth)
        len_gc, extrap_gc = length_get_centerline(cord, p1, p2, param_centerline, dir_qc=dir_qc)
        diff = len_gc - len_centroid
        flag = '  <-- re-check' if max(extrap_centroid, extrap_gc) > 3 else ''
        print(f'{sub}: centroid {len_centroid:.1f} mm, get_centerline {len_gc:.1f} mm, diff {diff:+.1f} mm '
              f'(extrap {extrap_centroid:.1f} / {extrap_gc:.1f} mm){flag}')
        rows.append({'subject': sub,
                     'length_centroid_mm': round(len_centroid, 2),
                     'length_get_centerline_mm': round(len_gc, 2),
                     'diff_mm': round(diff, 2),
                     'centroid_extrap_mm': round(extrap_centroid, 2),
                     'get_centerline_extrap_mm': round(extrap_gc, 2),
                     'S1top_source': s1src, 'conus_source': cosrc})
    if not rows:
        raise FileNotFoundError(f'No bounds/cord_seg found under {root}.')
    df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, 'fudan_sacral_length_centroid_vs_get_centerline.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} subjects. centroid {df['length_centroid_mm'].mean():.1f} +/- "
          f"{df['length_centroid_mm'].std():.1f} mm, get_centerline {df['length_get_centerline_mm'].mean():.1f} +/- "
          f"{df['length_get_centerline_mm'].std():.1f} mm, diff {df['diff_mm'].mean():+.1f} +/- "
          f"{df['diff_mm'].std():.1f} mm (range {df['diff_mm'].min():+.1f} to {df['diff_mm'].max():+.1f}). "
          f"-> {out_csv}")


if __name__ == '__main__':
    main()
