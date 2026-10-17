"""
Compute the whole spinal cord length from the pontomedullary junction (PMJ) to the conus medullaris tip, measured
along the spinal cord centerline (accounts for the cord curvature), on the whole-spine sagittal localizers.

Prerequisites (interactive, run once per subject, e.g. with manual_cord_length.sh):
    - the cord centerline traced on the localizer with SCT (sct_get_centerline -method viewer), saved as a
      *_label-centerline.csv (one point per axial slice, voxel coordinates);
    - the PMJ (label 50) placed manually and the conus (label 60) snapped onto the centerline at the CISS conus
      world-Z with snap_conus.py, saved in a *_label-PMJtip_dlabel.nii.gz.

For each subject the length is the arc length of the centerline between the PMJ and conus superior-inferior (S-I)
world-Z levels: the centerline x, y are smoothed with a 10 mm moving average (removes click tremor) and the two
ends are linearly interpolated/extrapolated to the exact PMJ and conus Z so that the length does not depend on
where the trace happened to start/stop.

Input:
- folder with, per subject, a *_label-centerline.csv and a *_label-PMJtip_dlabel.nii.gz (labels 50=PMJ, 60=conus).

Output (saved to the -o folder):
- cord_length.csv: cord length (mm) per subject, with sex and height; the columns si_distance_mm (straight
  PMJ-conus distance), centerline_extrap_mm (how far the trace was extrapolated beyond its range; re-trace if
  > 3 mm) and conus_vs_CISS_mm (distance of the snapped conus from the CISS conus Z) are kept for QC.

Author: Elia Rochiccioli
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from utils import OUTPUT_DIR, PARTICIPANTS_XLSX, add_sex_height

SMOOTH_MM = 10          # moving-average window for the centerline x, y (mm)


def centerline_length(ctl_csv, label_file, conus_z=np.nan, smooth_mm=SMOOTH_MM):
    """
    Arc length of the centerline between the PMJ (label 50) and the conus (label 60), in mm.
    :param ctl_csv: *_label-centerline.csv (voxel coordinates, one point per axial slice)
    :param label_file: *_label-PMJtip_dlabel.nii.gz with labels 50 (PMJ) and 60 (conus)
    :param conus_z: CISS conus world-Z (mm), only for the conus_vs_CISS_mm QC column
    :param smooth_mm: moving-average window (mm) applied to the centerline x, y
    :return: (length_mm, si_distance_mm, extrap_mm, conus_vs_CISS_mm)
    """
    ctl = np.loadtxt(ctl_csv, delimiter=',', ndmin=2)
    ld = np.asarray(nib.load(label_file).dataobj)
    aff = nib.load(label_file).affine
    if sorted(np.unique(ld[ld > 0]).astype(int).tolist()) != [50, 60]:
        raise ValueError('labels 50 (PMJ) and 60 (conus) are required')
    pmj = nib.affines.apply_affine(aff, np.argwhere(np.isclose(ld, 50))[0])
    tip = nib.affines.apply_affine(aff, np.argwhere(np.isclose(ld, 60))[0])

    # Sort the centerline along the S-I axis and smooth x, y (keep the ends raw)
    c = ctl[np.argsort(ctl[:, 2])]
    step = np.median(np.abs(np.diff(c[:, 2]))) or 1.0
    k = max(1, int(round(smooth_mm / step)))
    if len(c) > 2 * k + 1:
        kern = np.ones(k) / k
        cs = c.copy()
        for i in (0, 1):
            cs[:, i] = np.convolve(c[:, i], kern, mode='same')
        cs[:k] = c[:k]
        cs[-k:] = c[-k:]
        c = cs
    z = c[:, 2]

    def direction(seg):
        d = seg[-1] - seg[0]
        return d / (d[2] if d[2] != 0 else 1)

    def point_at(zt):
        if zt <= z[0]:
            return c[0] + direction(c[:5] if len(c) >= 5 else c) * (zt - z[0])
        if zt >= z[-1]:
            return c[-1] + direction(c[-5:] if len(c) >= 5 else c) * (zt - z[-1])
        i = np.searchsorted(z, zt)
        a, b = c[i - 1], c[i]
        return a + (zt - a[2]) / (b[2] - a[2]) * (b - a)

    z_pmj, z_tip = pmj[2], tip[2]
    z_hi, z_lo = max(z_pmj, z_tip), min(z_pmj, z_tip)
    inner = c[(z > z_lo) & (z < z_hi)]
    path = np.vstack([point_at(z_lo), inner, point_at(z_hi)])
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    extrap = max(0.0, z_hi - z.max()) + max(0.0, z.min() - z_lo)
    return length, abs(z_pmj - z_tip), extrap, abs(z_tip - conus_z)


def main():
    parser = argparse.ArgumentParser(description='Compute the whole spinal cord length (PMJ to conus) along the '
                                                 'centerline on the whole-spine localizers.')
    parser.add_argument('-i', required=True,
                        help='Folder with, per subject, a *_label-centerline.csv and a *_label-PMJtip_dlabel.nii.gz '
                             '(labels 50=PMJ, 60=conus).')
    parser.add_argument('-o', default=OUTPUT_DIR, help=f'Output folder for cord_length.csv. Default: {OUTPUT_DIR}')
    parser.add_argument('-conus-csv', default=None,
                        help='Optional CSV (subject, world_z_mm_ciss_tip) with the CISS conus world-Z per subject, '
                             'used only for the conus_vs_CISS_mm QC column.')
    parser.add_argument('-xlsx', default=PARTICIPANTS_XLSX,
                        help=f'Excel file with the subject sex and height. Default: {PARTICIPANTS_XLSX}')
    args = parser.parse_args()
    in_dir = os.path.abspath(os.path.expanduser(args.i))
    out_dir = os.path.abspath(os.path.expanduser(args.o))
    os.makedirs(out_dir, exist_ok=True)

    conus_z = {}
    if args.conus_csv:
        cz = pd.read_csv(args.conus_csv)
        conus_z = {f"sub-ltr{int(r['subject']):02d}": float(r['world_z_mm_ciss_tip']) for _, r in cz.iterrows()}

    rows = []
    for ctl_csv in sorted(glob.glob(os.path.join(in_dir, '**', '*_label-centerline.csv'), recursive=True)):
        subject = os.path.basename(ctl_csv).split('_')[0]
        label_file = glob.glob(os.path.join(os.path.dirname(ctl_csv), '*_label-PMJtip_dlabel.nii.gz'))
        if not label_file:
            print(f'{subject}: no PMJtip label, skipped')
            continue
        try:
            length, si, extrap, vs_ciss = centerline_length(ctl_csv, label_file[0], conus_z.get(subject, np.nan))
        except ValueError as e:
            print(f'{subject}: {e}, skipped')
            continue
        print(f'{subject}: {length:.1f} mm along the centerline')
        rows.append({'subject': subject, 'cord_length_mm': round(length, 2), 'si_distance_mm': round(si, 2),
                     'centerline_extrap_mm': round(extrap, 2), 'conus_vs_CISS_mm': round(vs_ciss, 2)})
    if not rows:
        raise FileNotFoundError(f'No *_label-centerline.csv found in {in_dir}. Check the -i path.')

    df = add_sex_height(pd.DataFrame(rows), args.xlsx)
    fname_csv = os.path.join(out_dir, 'cord_length.csv')
    df.to_csv(fname_csv, index=False)
    lengths = df['cord_length_mm']
    print(f'{len(df)} subjects, mean {lengths.mean():.1f} +/- {lengths.std():.1f} mm, '
          f'range {lengths.min():.0f}-{lengths.max():.0f}. -> {fname_csv}')


if __name__ == '__main__':
    main()
