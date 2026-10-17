"""
Place the conus label (60) on the traced centerline at the conus superior-inferior (S-I) world-Z taken from the
high-resolution lumbar CISS.

The conus tip is hard to see on the whole-spine localizer but is read reliably on the lumbar CISS. This script
takes the CISS conus world-Z (from -conus-csv or -z), finds the point on the traced centerline at that exact Z
(interpolated within the trace, or extrapolated along the cord direction if the trace stops short), and writes a
single voxel with label 60 there. The PMJ (label 50) is left untouched. compute_cord_length.py then measures the
length between labels 50 and 60.

Only the conus Z matters for the length (the x, y are taken from the centerline); the centerline must therefore be
traced from the PMJ down to at least the conus level.

Input (per subject):
- *_label-centerline.csv: the traced centerline (voxel coordinates, one point per axial slice)
- *_label-PMJtip_dlabel.nii.gz: label file that already contains the PMJ (label 50)

Example:
    python snap_conus.py -ctl sub-ltr21_..._label-centerline.csv -label sub-ltr21_..._label-PMJtip_dlabel.nii.gz -z -320.76

Author: Elia Rochiccioli
"""

import argparse
import numpy as np
import nibabel as nib


def snap_conus(ctl_csv, label_file, conus_z):
    """Write label 60 onto the centerline at the conus world-Z (conus_z, in mm)."""
    ctl = np.loadtxt(ctl_csv, delimiter=',', ndmin=2)
    img = nib.load(label_file)
    data = np.asarray(img.dataobj).copy()
    aff = img.affine
    inv = np.linalg.inv(aff)

    # Sort the centerline along the S-I axis (z ascending; the conus is the caudal, low-Z end)
    c = ctl[np.argsort(ctl[:, 2])]
    z = c[:, 2]
    short = 0.0
    if conus_z < z[0]:        # conus below the trace -> extrapolate down the bottom segment
        short = z[0] - conus_z
        a, b = c[0], c[1]
    elif conus_z > z[-1]:     # conus above the top (unexpected) -> extrapolate up the top segment
        short = conus_z - z[-1]
        a, b = c[-2], c[-1]
    else:                     # within the traced range -> interpolate (on-cord, exact Z)
        i = np.searchsorted(z, conus_z)
        a, b = c[i - 1], c[i]
    world = a + (conus_z - a[2]) / (b[2] - a[2]) * (b - a)
    vox = np.clip(np.rint(nib.affines.apply_affine(inv, world)).astype(int), 0, np.array(data.shape) - 1)
    if short > 1:
        print(f'  [!] centerline fell {short:.0f} mm short of the conus; label 60 was extrapolated to the exact '
              f'CISS level (re-trace lower for max precision).')

    old = np.argwhere(data == 60)
    data[data == 60] = 0
    data[tuple(vox)] = 60
    nib.save(nib.Nifti1Image(data.astype(img.get_data_dtype()), aff, img.header), label_file)
    print(f'conus (60) set to world={np.round(world, 1)} (Z={conus_z}), voxel {vox.tolist()}'
          + (f' (was at {old[0].tolist()})' if len(old) else ''))


def main():
    parser = argparse.ArgumentParser(description='Place the conus label (60) on the centerline at the CISS conus '
                                                 'world-Z.')
    parser.add_argument('-ctl', required=True, help='*_label-centerline.csv (traced centerline, voxel coordinates).')
    parser.add_argument('-label', required=True, help='*_label-PMJtip_dlabel.nii.gz (already contains the PMJ=50).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-z', type=float, help='Conus world-Z in mm (S-I axis).')
    group.add_argument('-conus-csv', help='CSV (subject, world_z_mm_ciss_tip) + -subject to look up the conus Z.')
    parser.add_argument('-subject', help='Subject id (e.g. sub-ltr21), used with -conus-csv.')
    args = parser.parse_args()

    if args.z is not None:
        conus_z = args.z
    else:
        import pandas as pd
        cz = pd.read_csv(args.conus_csv)
        num = int(''.join(ch for ch in args.subject if ch.isdigit()))
        conus_z = float(cz.loc[cz['subject'].astype(int) == num, 'world_z_mm_ciss_tip'].iloc[0])
    snap_conus(args.ctl, args.label, conus_z)


if __name__ == '__main__':
    main()
