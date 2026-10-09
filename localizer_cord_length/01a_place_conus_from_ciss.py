#!/usr/bin/env python
"""Set label 60 (conus) of a subject to the CISS-known conus world-Z.

Replaces whatever (possibly rough) label 60 is in the saved PMJ/tip file with a
single voxel placed on the traced centerline at the conus world-Z taken from
tip_conus_CISS_worldZ.csv (interpolated/extrapolated along the centerline to
that exact S-I level). Label 50 (PMJ) is left untouched.

Usage:  snap_conus.py 03      (or sub-ltr03)
"""
import sys, os, csv, glob
import numpy as np, nibabel as nib

BIDS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV  = "/Users/eliarochiccioli/PhD/Lumbar registration /tip_conus_CISS_worldZ.csv"

only_if_missing = "--only-if-missing" in sys.argv   # skip if a conus (60) already exists
# optional explicit conus world-Z (for the 4 subjects not in Silvan's CSV,
# derived here from an SCT cord segmentation of their CISS): --z=-375.2
zopt = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--z=")), None)
arg = [a for a in sys.argv[1:] if not a.startswith("--")][0]
sub = arg if arg.startswith("sub-ltr") else f"sub-ltr{int(arg):02d}"
num = f"{int(sub.replace('sub-ltr','')):02d}"
conus = {f"{int(r['subject']):02d}": float(r['world_z_mm_ciss_tip']) for r in csv.DictReader(open(CSV))}
# local extra conus values for the 4 subjects absent from Silvan's CSV
# (11/17/21/23), derived here from an SCT cord segmentation of their CISS
EXTRA = os.path.join(os.path.dirname(__file__), "conus_extra.csv")
if os.path.exists(EXTRA):
    conus.update({f"{int(r['subject']):02d}": float(r['world_z_mm_ciss_tip']) for r in csv.DictReader(open(EXTRA))})
if zopt is not None:
    zc = zopt
elif num in conus:
    zc = conus[num]
else:
    sys.exit(f"{sub}: no CISS conus -> leave label 60 as manually placed")

cg = glob.glob(f"{BIDS}/derivatives/labels/{sub}/ses-*/anat/*_label-centerline.csv")
lg = glob.glob(f"{BIDS}/derivatives/labels/{sub}/ses-*/anat/*_label-PMJtip_dlabel.nii.gz")
if not cg or not lg:
    sys.exit(f"{sub}: no centerline/label yet, skipped")
ctl_csv, lblf = cg[0], lg[0]
ctl = np.loadtxt(ctl_csv, delimiter=",", ndmin=2)
img = nib.load(lblf); data = np.asarray(img.dataobj).copy(); aff = img.affine; inv = np.linalg.inv(aff)
if only_if_missing and np.any(np.isclose(data, 60)):
    sys.exit(f"{sub}: conus already present, left unchanged")

# place the conus on the centerline (= on the cord) at the EXACT conus S-I level zc.
# z is ascending; the conus is the caudal (low-Z) end. If the traced centerline
# does not reach zc, extrapolate along the end segment to the exact zc (same as
# the length calc), so the conus sits at the true CISS level -- `short` flags how
# far the centerline fell short so it can be re-traced if the gap is large.
c = ctl[np.argsort(ctl[:, 2])]; z = c[:, 2]
short = 0.0
if zc < z[0]:        # conus below the traced centerline -> extrapolate down the bottom segment
    short = z[0] - zc; a, b = c[0], c[1]
elif zc > z[-1]:     # conus above the top (unexpected) -> extrapolate up the top segment
    short = zc - z[-1]; a, b = c[-2], c[-1]
else:                # conus within the traced range -> interpolate (on-cord, exact Z)
    i = np.searchsorted(z, zc); a, b = c[i-1], c[i]
t = (zc - a[2]) / (b[2] - a[2]); world = a + t * (b - a)
vox = np.rint(nib.affines.apply_affine(inv, world)).astype(int)
vox = np.clip(vox, 0, np.array(data.shape) - 1)
if short > 1:
    print(f"  [!] NOTE: centerline fell {short:.0f} mm short of the conus; the conus (60) was")
    print(f"      EXTRAPOLATED along the cord direction to the exact CISS level (length includes")
    print(f"      a {short:.0f} mm straight-line extrapolation -> re-trace lower for max precision).")

old = np.argwhere(data == 60)
data[data == 60] = 0
data[tuple(vox)] = 60
nib.save(nib.Nifti1Image(data.astype(img.get_data_dtype()), aff, img.header), lblf)
print(f"{sub}: conus (60) set to world={np.round(world,1)} (Z={zc}), voxel {vox.tolist()}")
if len(old): print(f"  (was at voxel {old[0].tolist()})")
