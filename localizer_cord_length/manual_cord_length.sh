#!/bin/bash
#
# Manual cord length (PMJ -> cord tip) on the sagittal localizer.
# See: https://github.com/sct-pipeline/spinal-levels/issues/4
#
# For each subject:
#   1. sct_get_centerline -method viewer   -> manual centerline (click the cord every <gap> mm)
#   2. sct_label_utils -create-viewer      -> PMJ (label 50) and cord tip (label 60)
#   3. length = arc length of the centerline between PMJ and tip (mm)
#
# Outputs go to derivatives/labels/sub-*/ses-*/anat/ ; lengths to derivatives/labels/cord_length.csv
# Existing labels are reused (delete the file to redo it).
#
# Usage (needs a display, run in your own terminal, not via Claude):
#   bash code/manual_cord_length.sh                 # all subjects
#   bash code/manual_cord_length.sh sub-ltr01 sub-ltr02

set -e

# Auto-detect the SCT install (this machine has it at ~/sct_7.3)
if [ -z "$SCT_DIR" ]; then
  for cand in "$HOME/sct_7.3" "$HOME/spinalcordtoolbox"; do
    [ -x "$cand/bin/sct_get_centerline" ] && SCT_DIR="$cand" && break
  done
fi
SCT_DIR=${SCT_DIR:-$HOME/spinalcordtoolbox}
export PATH="$SCT_DIR/bin:$PATH"
unset LD_LIBRARY_PATH  # avoids FSL/CUDA lib conflicts with SCT

BIDS="$(cd "$(dirname "$0")/.." && pwd)"
DERIV="$BIDS/derivatives/labels"
GAP=10  # mm between manually clicked centerline points
OUT_CSV="$DERIV/cord_length.csv"
CONUS_CSV="/Users/eliarochiccioli/PhD/Lumbar registration /tip_conus_CISS_worldZ.csv"
EXTRA_CSV="$(dirname "$0")/conus_extra.csv"
# subjects whose conus world-Z is known (Silvan's CISS CSV + our conus_extra.csv
# for 11/17/21/23) -> label 60 is auto-placed on the centerline at that Z and
# only the PMJ is clicked by hand.
COVERED=$("$SCT_DIR/python/envs/venv_sct/bin/python" -c "
import csv,sys
subs=set()
for p in ['$CONUS_CSV','$EXTRA_CSV']:
    try:
        for r in csv.DictReader(open(p)): subs.add(int(r['subject']))
    except FileNotFoundError: pass
print(' '.join(f'sub-ltr{n:02d}' for n in sorted(subs)))" 2>/dev/null)

cd "$BIDS"
if [ $# -gt 0 ]; then SUBJECTS="$@"; else SUBJECTS=$(ls -d sub-* | sort); fi

for SUB in $SUBJECTS; do
  for IMG in "$BIDS"/$SUB/ses-*/anat/${SUB}_ses-*_acq-localizerSag_T2w.nii.gz; do
    [ -f "$IMG" ] || { echo "No localizer for $SUB"; continue; }
    REL=$(dirname "${IMG#$BIDS/}")
    NAME=$(basename "$IMG" .nii.gz)
    OUT="$DERIV/$REL"
    mkdir -p "$OUT"
    CTL="$OUT/${NAME}_label-centerline.nii.gz"
    LBL="$OUT/${NAME}_label-PMJtip_dlabel.nii.gz"

    echo "=== $NAME ==="
    if [ ! -f "$CTL" ]; then
      sct_get_centerline -i "$IMG" -method viewer -gap $GAP -space phys -o "$CTL"
    fi
    if [ ! -f "$LBL" ]; then
      if echo " $COVERED " | grep -q " $SUB "; then
        # conus known from the CISS -> click ONLY the PMJ; 60 is auto-placed.
        # NB the SCT viewer often exits non-zero on close -> "|| true" so that
        # snap_conus still runs (set -e would otherwise abort here).
        sct_label_utils -i "$IMG" -create-viewer 50 \
          -msg "Click 50 = PMJ (pontomedullary junction). The conus (60) is placed automatically from the CISS." \
          -o "$LBL" || true
        "$SCT_DIR/python/envs/venv_sct/bin/python" "$BIDS/code/snap_conus.py" "$SUB" || true
      else
        # no CISS conus for this subject (ltr11/17/21/23) -> click both PMJ and conus
        sct_label_utils -i "$IMG" -create-viewer 50,60 \
          -msg "Click 50 = PMJ (pontomedullary junction), then 60 = tip of the spinal cord (conus)" \
          -o "$LBL" || true
      fi
    fi
  done
done

# Safety net: auto-place the conus (60) on any CISS-covered subject that still
# has only the PMJ (labelled before this existed, or if the inline snap was skipped).
for SUB in $COVERED; do
  "$SCT_DIR/python/envs/venv_sct/bin/python" "$BIDS/code/snap_conus.py" "$SUB" --only-if-missing 2>/dev/null || true
done

# Compute lengths for every subject with both files
"$SCT_DIR/python/envs/venv_sct/bin/python" -I - "$DERIV" "$OUT_CSV" <<'EOF'
import sys, glob, os
import numpy as np
import nibabel as nib

deriv, out_csv = sys.argv[1], sys.argv[2]
rows = []
for ctl_nii in sorted(glob.glob(os.path.join(deriv, "sub-*", "ses-*", "anat", "*_label-centerline.nii.gz"))):
    name = os.path.basename(ctl_nii).replace("_label-centerline.nii.gz", "")
    ctl_csv = ctl_nii.replace(".nii.gz", ".csv")
    lbl = ctl_nii.replace("_label-centerline.nii.gz", "_label-PMJtip_dlabel.nii.gz")
    if not (os.path.exists(ctl_csv) and os.path.exists(lbl)):
        print(f"{name}: missing centerline csv or PMJ/tip labels, skipped")
        continue
    # Centerline in physical space (RAS+, mm): one point per axial slice
    ctl = np.loadtxt(ctl_csv, delimiter=",", ndmin=2)
    # PMJ / tip in physical space (RAS+, mm)
    img = nib.load(lbl)
    data = np.asarray(img.dataobj)
    pts = {}
    for val in (50, 60):
        vox = np.argwhere(np.isclose(data, val))
        if len(vox) != 1:
            print(f"{name}: expected 1 voxel with label {val}, found {len(vox)}, skipped")
            break
        pts[val] = nib.affines.apply_affine(img.affine, vox[0])
    else:
        # Length ALONG the centerline between the PMJ-Z and the tip-Z.
        # We clip/extrapolate the centerline exactly to those two S-I levels
        # (interpolating a point at each boundary Z) rather than jumping to the
        # label's 3D position -- the labels may sit on a different L-R sagittal
        # slice than the traced centerline, and bridging to them in 3D would add
        # a spurious left-right offset that is not real cord length.
        zlo, zhi = sorted((pts[50][2], pts[60][2]))
        c = ctl[np.argsort(ctl[:, 2])]          # centerline, ascending Z
        z = c[:, 2]

        def pt_at(zt):                          # point on the centerline at S-I level zt
            if zt <= z[0]:    a, b = c[0], c[1]        # extrapolate below
            elif zt >= z[-1]: a, b = c[-2], c[-1]      # extrapolate above (e.g. centerline stops below PMJ)
            else:
                i = np.searchsorted(z, zt); a, b = c[i - 1], c[i]
            t = (zt - a[2]) / (b[2] - a[2]); return a + t * (b - a)

        inner = c[(z > zlo) & (z < zhi)]
        pts_path = np.vstack([pt_at(zlo), inner, pt_at(zhi)])
        length = np.sum(np.linalg.norm(np.diff(pts_path, axis=0), axis=1))
        # flag if the centerline did not actually span up to the PMJ (top extrapolated)
        extrap = max(0.0, zhi - z[-1]) + max(0.0, z[0] - zlo)
        sub, ses = name.split("_")[0], name.split("_")[1]
        rows.append((sub, ses, length, abs(pts[50][2] - pts[60][2]), extrap))
        note = f"  [!] centerline extrapolated {extrap:.0f} mm beyond its traced range" if extrap > 3 else ""
        print(f"{name}: cord length = {length:.1f} mm (straight S-I distance {abs(pts[50][2]-pts[60][2]):.1f} mm){note}")

with open(out_csv, "w") as f:
    f.write("participant_id,session_id,cord_length_mm,si_distance_mm,centerline_extrap_mm\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f}\n")
print(f"Saved {out_csv}")
EOF
