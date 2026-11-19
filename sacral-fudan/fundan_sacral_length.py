"""
Compute the sacral spinal cord length (top of S1 to the conus medullaris) ALONG THE CORD CENTERLINE for the
Fudan (Shanghai) lumbar cohort, to match ../sacral-balgrist/sacral_length_along_centerline.py (and hence the
whole-cord / lumbar centerline metric). The cord curvature is accounted for (arc length), not a straight line.

Unlike Balgrist, the Fudan dataset does not ship a localizer centerline, so the centerline substrate differs:
- centerline = per-axial-slice centroid of the CISS cord segmentation
  (derivatives/cord_seg_from_markers/sub-*/sub-*_cord_seg.nii.gz, rasterized from the manual cord_* contours; it reaches the conus);
- bounds     = manually placed S1-top and conus points (sacral_length/bounds/sub-*/sub-*_{S1top,conus}.mrk.json).

The conus of the cord segmentation stops a few mm above the true conus for the larger subjects, so the conus was
re-checked and raised on the SAGITTAL view; corrected points live in a `new/` subfolder next to each subject's
original bounds. This script prefers `new/<file>` when present (falling back to the original point), and records the
choice in the S1top_source / conus_source columns.

Input (-i: Fudan dataset root):
- derivatives/cord_seg_from_markers/sub-*/sub-*_cord_seg.nii.gz           # cord mask
- sacral_length/bounds/sub-*/sub-*_{S1top,conus}.mrk.json   # manual bounds (original)
- sacral_length/bounds/sub-*/new/sub-*_{S1top,conus}.mrk.json  # corrected bounds (preferred when present)
- derivatives/L1-L5/L1-L5_lengths.csv                       # sex, height_cm, age, weight_kg

Output (-o, default <-i>/sacral_length):
- fundan_sacral_length.csv : subject, sex, height_cm, length_S1-conus_centerline_mm, centerline_extrap_mm (QC;
  re-check if > 3 mm), S1top_source, conus_source.

Author: Elia Rochiccioli
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

LENGTH_COL = 'length_S1-conus_centerline_mm'   # same naming as ../sacral-balgrist
SMOOTH_MM = 10

DEFAULT_ROOT = os.path.expanduser('~/Desktop/fundan_dataset')


def mrk_pos(path):
    """RAS position of the first control point (LPS->RAS flips x,y; S-I=z is identical in both)."""
    j = json.load(open(path))
    mk = j['markups'][0]
    pos = np.array(mk['controlPoints'][0]['position'], float)
    if mk.get('coordinateSystem', 'LPS').upper().startswith('L'):
        pos = pos * np.array([-1, -1, 1])
    return pos


def resolve_bound(bounds_dir, subject, name):
    """Prefer the corrected point in bounds/sub/new/<file>, else the original bounds/sub/<file>. Returns (path, source)."""
    fname = f'{subject}_{name}.mrk.json'
    new = os.path.join(bounds_dir, subject, 'new', fname)
    old = os.path.join(bounds_dir, subject, fname)
    if os.path.exists(new):
        return new, 'new'
    if os.path.exists(old):
        return old, 'original'
    return None, 'missing'


def cord_centerline(cord_file):
    """Per-axial-slice centroid of the cord mask -> Nx3 world (RAS) points, sorted along S-I."""
    img = nib.as_closest_canonical(nib.load(cord_file))   # RAS+: axis 2 = S-I
    data = np.asarray(img.dataobj) > 0
    aff = img.affine
    ii, jj, kk = np.nonzero(data)
    pts = [nib.affines.apply_affine(aff, np.array([ii[kk == k].mean(), jj[kk == k].mean(), k]))
           for k in np.unique(kk)]
    pts = np.array(pts)
    return pts[np.argsort(pts[:, 2])]


def arc_length_between(c, z_top, z_conus, smooth_mm=SMOOTH_MM):
    """Arc length (mm) of the centerline between two S-I world-Z levels, same smoothing + end interpolation
    as ../sacral-balgrist/sacral_length_along_centerline.py."""
    c = c[np.argsort(c[:, 2])]
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
    ap = argparse.ArgumentParser(description='Compute the Fudan sacral cord length (top of S1 to the conus) along '
                                             'the cord centerline.')
    ap.add_argument('-i', default=DEFAULT_ROOT, help=f'Fudan dataset root. Default: {DEFAULT_ROOT}')
    ap.add_argument('-o', default=None, help='Output folder. Default: <-i>/sacral_length')
    ap.add_argument('-smooth', type=int, default=SMOOTH_MM, help=f'Centerline smoothing (mm). Default: {SMOOTH_MM}')
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.i))
    bounds_dir = os.path.join(root, 'sacral_length', 'bounds')
    cord_dir = os.path.join(root, 'derivatives', 'cord_seg_from_markers')
    demo_csv = os.path.join(root, 'derivatives', 'L1-L5', 'L1-L5_lengths.csv')
    out_dir = os.path.abspath(os.path.expanduser(args.o)) if args.o else os.path.join(root, 'sacral_length')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(bounds_dir, 'sub-*'))):
        sub = os.path.basename(d)
        s1, s1src = resolve_bound(bounds_dir, sub, 'S1top')
        co, cosrc = resolve_bound(bounds_dir, sub, 'conus')
        cord = os.path.join(cord_dir, sub, f'{sub}_cord_seg.nii.gz')
        if not (s1 and co):
            print(f'  skip {sub}: missing bound'); continue
        if not os.path.exists(cord):
            print(f'  skip {sub}: no cord_seg'); continue
        p1, p2 = mrk_pos(s1), mrk_pos(co)
        cl, extrap = arc_length_between(cord_centerline(cord), p1[2], p2[2], smooth_mm=args.smooth)
        flag = '  <-- re-check conus/centerline' if extrap > 3 else ''
        print(f'{sub}: {cl:.1f} mm (extrap {extrap:.1f} mm, bounds S1={s1src}/conus={cosrc}){flag}')
        rows.append({'subject': sub, LENGTH_COL: round(cl, 2),
                     'centerline_extrap_mm': round(extrap, 2),
                     'S1top_source': s1src, 'conus_source': cosrc})
    if not rows:
        raise FileNotFoundError(f'No bounds/cord_seg found under {root}.')
    df = pd.DataFrame(rows)

    # subject sex and height from the L1-L5 demographics table
    demo = pd.read_csv(demo_csv)[['subject', 'sex', 'height_cm', 'age', 'weight_kg']]
    df = df.merge(demo, on='subject', how='left')
    df = df[['subject', 'sex', 'height_cm', 'age', 'weight_kg', LENGTH_COL,
             'centerline_extrap_mm', 'S1top_source', 'conus_source']]

    out_csv = os.path.join(out_dir, 'fundan_sacral_length.csv')
    df.to_csv(out_csv, index=False)
    L = df[LENGTH_COL]
    print(f"\n{len(df)} subjects, mean {L.mean():.1f} +/- {L.std():.1f} mm, range {L.min():.0f}-{L.max():.0f} "
          f"(max extrap {df['centerline_extrap_mm'].max():.1f} mm). -> {out_csv}")
    print('Run correlation_length_height.py on this CSV for the sacral length vs height figure.')


if __name__ == '__main__':
    main()
