"""
TL;DR: Compute the sacral spinal cord length (top of S1 to the conus medullaris) along the spinal cord centerline.

This script is adapted from lumbar_length_along_centerline.py and compute_cord_length.py to measure the SACRAL cord
length, i.e. from the top of the S1 spinal level down to the conus tip, along the spinal cord centerline (so the cord
curvature is accounted for and the length is properly aligned with the cord).

WHY TWO IMAGES:
The S1-S2 segmentation traced on the axial CISS does NOT reach the conus (it stops ~15-30 mm above it), so measuring
"S1 to bottom of S2" only covers S1-S2 and underestimates the sacral cord. The conus tip, however, is already known
from the whole-cord-length work: it is placed (label 60) on the cord centerline traced on the whole-spine localizer
(same pipeline as compute_cord_length.py, PMJ to conus). We therefore measure the arc length along the LOCALIZER
centerline (which does reach the conus, so no extrapolation is needed) between:
    - top of S1 : the superior-inferior (S-I) world-Z of the top edge of the S1 level, taken from the CISS
                  segmentation (1=L1 ... 5=L5, 6=S1, 7=S2);
    - conus     : label 60 (conus) of the localizer *_label-PMJtip_dlabel.nii.gz.
This is exactly the whole-cord-length method, with the top of S1 replacing the PMJ.

Input:
- -seg  : folder with the CISS spinal-level segmentations (sub-*/*spinallevels_dseg.nii[.gz]), 6=S1.
- -i    : folder with, per subject, the localizer *_label-centerline.csv and *_label-PMJtip_dlabel.nii.gz
          (labels 50=PMJ, 60=conus), e.g. ~/Desktop/Balgrist Dataset/derivatives/labels.

Output (saved to the -o folder):
    - sacral_length_S1-conus_centerline.csv : length along the centerline per subject, with sex and height, plus the
      centerline_extrap_mm QC column (how far S1/conus fell beyond the traced centerline; re-trace if > 3 mm).
    - figure_lumbar_length_S1-conus_centerline.png : the same length plotted per subject.

The script requires the SCT conda environment (for reading the CISS segmentation with the correct orientation):
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python sacral_length_along_centerline.py \
        -seg ~/Downloads/spinal_level_segmentation_L1-S2 \
        -i ~/Desktop/Balgrist Dataset/derivatives/labels \
        -xlsx ~/Downloads/spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx \
        -o ~/Downloads/sacral_length_results

Author: Elia Rochiccioli (adapted from Jan Valosek's lumbar length script and compute_cord_length.py).
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from spinalcordtoolbox.image import Image

PARTICIPANTS_SHEET = 'Spinal level heights (mm)'   # sheet with the 'sex' and 'Height (cm)' columns


def add_sex_height(df, fname_xlsx):
    """Add the subject sex ('F'/'M') and height (cm) from the participants Excel to a dataframe with a 'subject'
    column (the length-vs-height figure is produced by correlation_length_height.py)."""
    xl = pd.read_excel(fname_xlsx, sheet_name=PARTICIPANTS_SHEET)
    xl = xl[xl['Participants'].astype(str).str.startswith('sub-')]
    xl['sex'] = xl['sex'].str.upper()
    xl = xl.rename(columns={'Height (cm)': 'height'})
    return df.merge(xl[['Participants', 'sex', 'height']], how='left', left_on='subject',
                    right_on='Participants').drop(columns='Participants')


S1_VALUE = 6            # value of the S1 spinal level in the CISS segmentation
SMOOTH_MM = 10          # moving-average window for the localizer centerline x, y (as in compute_cord_length.py)

# Metric for the sacral length (the '_mm' column name written to the CSV)
SACRAL_METRIC = {'name': 'length_S1-conus_centerline', 'label': 'Length S1 to conus'}

DEFAULT_SEG = os.path.expanduser('~/Downloads/spinal_level_segmentation_L1-S2')
DEFAULT_LOC = os.path.expanduser('~/Desktop/Balgrist Dataset/derivatives/labels')
DEFAULT_XLSX = os.path.expanduser('~/Downloads/spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx')
DEFAULT_OUTPUT = os.path.expanduser('~/Desktop/Balgrist Dataset/sacral_length')
# Manually placed/checked bounds from the Slicer tool (make_sacral_place_bounds.py): sub-ltrNN_{S1top,conus}.mrk.json
DEFAULT_BOUNDS = os.path.expanduser('~/Desktop/Balgrist Dataset/sacral_length/bounds')


def s1_top_worldz(seg_file):
    """
    S-I world-Z (mm) of the top edge of the S1 level, from the CISS segmentation.
    The segmentation is reoriented to RPI; the top edge is the centroid of the most superior S1 slice plus half a
    slice along the S-I axis (so the whole top slice is included, as in the lumbar top-bottom metric).
    :return: world-Z in mm (NaN if S1 is missing)
    """
    im = Image(seg_file).change_orientation('RPI')
    aff = im.affine
    s1 = np.array(np.where(im.data == S1_VALUE))
    if s1.shape[1] == 0:
        return np.nan
    ktop = s1[2].max()
    centroid = s1[:, s1[2] == ktop].mean(1)             # center of the S1 top slice
    wz = (aff @ np.append(centroid, 1))[2]
    return wz + 0.5 * aff[2, 2]                          # top edge: half a slice more superior


def conus_z_from_label(label_file):
    """S-I world-Z (mm) of the conus (label 60) on the localizer PMJtip file."""
    ld = np.asarray(nib.load(label_file).dataobj)
    if not np.any(np.isclose(ld, 60)):
        raise ValueError('label 60 (conus) is required')
    aff = nib.load(label_file).affine
    return float(nib.affines.apply_affine(aff, np.argwhere(np.isclose(ld, 60))[0])[2])


def markup_z(path):
    """
    S-I world-Z (mm) of the first control point of a Slicer .mrk.json. The point is placed on the CISS; markups JSON
    stores positions in LPS or RAS, but the S-I value (3rd component) is identical in both, which is all we need here.
    """
    j = json.load(open(path))
    return float(j['markups'][0]['controlPoints'][0]['position'][2])


def manual_bound_z(bounds_dir, subject, name):
    """world-Z of a manually placed/checked bound (name = 'S1top' or 'conus'), or None if the file is absent."""
    p = os.path.join(bounds_dir, subject, f'{subject}_{name}.mrk.json')
    return markup_z(p) if os.path.exists(p) else None


def arc_length_between(ctl_csv, z_top, z_conus, smooth_mm=SMOOTH_MM):
    """
    Arc length (mm) of the localizer centerline between two S-I world-Z levels (z_top = top of S1, z_conus = conus),
    with the same smoothing and end interpolation as compute_cord_length.py.
    :param ctl_csv: localizer *_label-centerline.csv (world coordinates, one point per axial slice)
    :return: (length_mm, extrap_mm) where extrap_mm is how far the ends fell beyond the traced centerline
    """
    ctl = np.loadtxt(ctl_csv, delimiter=',', ndmin=2)

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

    z_hi, z_lo = max(z_top, z_conus), min(z_top, z_conus)
    inner = c[(z > z_lo) & (z < z_hi)]
    path = np.vstack([point_at(z_lo), inner, point_at(z_hi)])
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    extrap = max(0.0, z_hi - z.max()) + max(0.0, z.min() - z_lo)
    return length, extrap


def main():
    parser = argparse.ArgumentParser(description='Compute the sacral spinal cord length (top of S1 to the conus) along '
                                                 'the localizer cord centerline.')
    parser.add_argument('-seg', default=DEFAULT_SEG,
                        help=f'Folder with the CISS spinal-level segmentations (sub-*/*spinallevels_dseg.nii[.gz], '
                             f'6=S1). Default: {DEFAULT_SEG}')
    parser.add_argument('-i', default=DEFAULT_LOC,
                        help=f'Folder with, per subject, the localizer *_label-centerline.csv and '
                             f'*_label-PMJtip_dlabel.nii.gz (labels 50=PMJ, 60=conus). Default: {DEFAULT_LOC}')
    parser.add_argument('-o', default=DEFAULT_OUTPUT,
                        help=f'Output folder for the CSV file and the figure. Default: {DEFAULT_OUTPUT}')
    parser.add_argument('-smooth', type=int, default=SMOOTH_MM,
                        help=f'Moving-average window (mm) for the centerline x, y. Default: {SMOOTH_MM}.')
    parser.add_argument('-bounds', default=DEFAULT_BOUNDS,
                        help=f'Folder with the manually placed/checked bounds (sub-ltrNN/sub-ltrNN_{{S1top,conus}}'
                             f'.mrk.json from the Slicer tool). Used when present, otherwise the bounds are computed '
                             f'automatically. Default: {DEFAULT_BOUNDS}')
    parser.add_argument('-xlsx', default=DEFAULT_XLSX,
                        help=f'Excel file with the subject sex and height. Default: {DEFAULT_XLSX}')
    args = parser.parse_args()
    seg_dir = os.path.abspath(os.path.expanduser(args.seg))
    loc_dir = os.path.abspath(os.path.expanduser(args.i))
    bounds_dir = os.path.abspath(os.path.expanduser(args.bounds))
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    os.makedirs(dir_out, exist_ok=True)

    rows = []
    seg_files = sorted(glob.glob(os.path.join(seg_dir, '**', '*spinallevels_dseg.nii'), recursive=True) +
                       glob.glob(os.path.join(seg_dir, '**', '*spinallevels_dseg.nii.gz'), recursive=True))
    for seg in seg_files:
        subject = os.path.basename(seg).split('_')[0]
        ctl = glob.glob(os.path.join(loc_dir, subject, '**', '*_label-centerline.csv'), recursive=True)
        lab = glob.glob(os.path.join(loc_dir, subject, '**', '*_label-PMJtip_dlabel.nii.gz'), recursive=True)
        if not ctl:
            print(f'{subject}: no localizer centerline, skipped')
            continue

        # Top of S1: manually placed point if available, otherwise computed from the segmentation
        z_s1_manual = manual_bound_z(bounds_dir, subject, 'S1top')
        z_top = z_s1_manual if z_s1_manual is not None else s1_top_worldz(seg)
        if z_top is None or np.isnan(z_top):
            print(f'{subject}: no S1 top (manual or segmentation), skipped')
            continue

        # Conus: manually placed point if available, otherwise label 60 on the localizer
        z_co_manual = manual_bound_z(bounds_dir, subject, 'conus')
        if z_co_manual is not None:
            z_conus = z_co_manual
        elif lab:
            z_conus = conus_z_from_label(lab[0])
        else:
            print(f'{subject}: no conus (manual or label 60), skipped')
            continue

        source = ('manual' if (z_s1_manual is not None and z_co_manual is not None)
                  else 'auto' if (z_s1_manual is None and z_co_manual is None) else 'mixed')
        length, extrap = arc_length_between(ctl[0], z_top, z_conus, smooth_mm=args.smooth)
        flag = '  <-- re-check conus/centerline' if extrap > 3 else ''
        print(f'{subject}: {length:.1f} mm along the centerline (extrap {extrap:.1f} mm, bounds: {source}){flag}')
        rows.append({'subject': subject,
                     f"{SACRAL_METRIC['name']}_mm": round(length, 2),
                     'centerline_extrap_mm': round(extrap, 2),
                     'bounds_source': source})
    if not rows:
        raise FileNotFoundError(f'No matching segmentations/localizer labels found. Check -seg {seg_dir} and -i {loc_dir}.')
    df = pd.DataFrame(rows)

    # Add the subject sex and height (also used to color and sort the subjects in the figure)
    df = add_sex_height(df, args.xlsx)

    fname_csv = os.path.join(dir_out, 'sacral_length_S1-conus_centerline.csv')
    df.to_csv(fname_csv, index=False)
    lengths = df[f"{SACRAL_METRIC['name']}_mm"]
    print(f"{len(df)} subjects, mean {lengths.mean():.1f} +/- {lengths.std():.1f} mm, "
          f"range {lengths.min():.0f}-{lengths.max():.0f}. -> {fname_csv}")
    print('Run correlation_length_height.py on this CSV for the sacral length vs height figure.')


if __name__ == '__main__':
    main()
