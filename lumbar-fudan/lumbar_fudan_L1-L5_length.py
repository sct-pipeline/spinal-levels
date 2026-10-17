"""
TL;DR: Compute the lumbar spinal cord L1-L5 length along the centerline for the Fudan
(Shanghai) lumbar dataset, straight from the provided Slicer annotations.

Unlike the Balgrist/EPFL lumbar scripts, the Fudan dataset does not ship a per-level
`*_spinallevels_dseg.nii.gz`; it ships Slicer markups instead. This script builds the
inputs it needs and measures the length, in three steps:

    1. Spinal cord mask: the `cord_*` ClosedCurve markers (one closed contour per CISS
       axial slice, LPS) are rasterized onto the CISS grid -> binary cord mask.
    2. L1-L5 end boundaries along the superior-inferior (S-I) axis, each defined by the
       rootlet zone of entry (ZEP) on the cord:
         - L1 top    = ZEP of L1   (top edge of the most superior L1 slice)
         - L5 bottom = ZEP(S1) - 1 slice = last slice of L5 (bottom edge included)
       Taken from manually placed Slicer points (`sub-*_L1top.mrk.json`,
       `sub-*_L5S1.mrk.json`) when available, otherwise estimated automatically as the
       most superior (max world-Z) point of the `nerveroots_L1` / `nerveroots_S1` markers.
    3. Length: a labeled seg is built (whole cord = 2, value 1 on the L1-top slice, value
       5 on the L5-bottom slice) and the centerline is fitted with SCT's get_centerline()
       + ParamCenterline (bspline, smooth=30, as in sct_process_segmentation). The length
       is the arc length of the centerline between the two end slices (both included),
       computed from SCT's Centerline.incremental_length -- i.e., the exact same
       measurement as lumbar_length_along_centerline.py (-metric top-bottom), so the Fudan
       numbers are comparable to the Balgrist/EPFL cohorts.

Input: a BIDS-like Fudan dataset with
    rawdata/sub-*/anat/sub-*_CISS.nii.gz
    markers/sub-*/sub-*_{cord_*,nerveroots_<level>_<L|R>}.json   (Slicer markups, LPS)
    rawdata/participants.tsv                                     (sex, height, ...)
and, optionally, manually placed boundary points under -bounds:
    <bounds>/sub-*/sub-*_{L1top,L5S1}.mrk.json

Output (-o):
    L1-L5_lengths.csv / .xlsx   length per subject with sex, height, age, weight
    L1-L5_vs_height.png         length vs height, coloured by sex, with the correlation
    sub-*_spinallevels_dseg.nii.gz + centerline/  for visual QC

The script requires the SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh && conda activate venv_sct

Example:
    python lumbar_fudan_L1-L5_length.py -i /path/to/fudan -bounds /path/to/fudan/derivatives/route_b_boundary

Centerline method & metric after Jan Valosek (lumbar_length_along_centerline.py).
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib.path import Path

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.types import Centerline

SEX_STYLE = {'F': {'color': '#2a78d6', 'marker': 'o', 'label': 'Female'},
             'M': {'color': '#eb6834', 'marker': 's', 'label': 'Male'}}


def lps_points(fname):
    """All control points of a Slicer markups json, (N,3) in LPS (as stored)."""
    j = json.load(open(fname))
    return np.array([c['position'] for m in j['markups'] for c in m['controlPoints']], float)


def rasterize_cord(markers_dir, sub, ref_img):
    """Rasterize the cord ClosedCurve contours onto the reference grid -> binary mask."""
    inv = np.linalg.inv(ref_img.affine)
    cord = np.zeros(ref_img.shape, np.uint8)
    for f in sorted(glob.glob(os.path.join(markers_dir, f'{sub}_cord_*.json'))):
        pts = lps_points(f)
        ras = pts.copy(); ras[:, 0] *= -1; ras[:, 1] *= -1     # LPS -> RAS
        vox = nib.affines.apply_affine(inv, ras)
        k = int(round(vox[:, 2].mean()))
        if not 0 <= k < ref_img.shape[2]:
            continue
        poly = vox[:, :2]
        i0, j0 = np.floor(poly.min(0)).astype(int); i1, j1 = np.ceil(poly.max(0)).astype(int)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, ref_img.shape[0] - 1), min(j1, ref_img.shape[1] - 1)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
        inside = Path(poly).contains_points(np.c_[ii.ravel(), jj.ravel()]).reshape(ii.shape)
        cord[ii[inside], jj[inside], k] = 1
    return cord


def nerveroot_zep(markers_dir, sub, level):
    """Auto zone of entry of a level = max world-Z of its L/R nerveroots points."""
    zs = []
    for side in ('L', 'R'):
        for f in glob.glob(os.path.join(markers_dir, f'{sub}_nerveroots_{level}_{side}.json')):
            zs.append(lps_points(f)[:, 2])                     # LPS z == world z
    return float(np.concatenate(zs).max()) if zs else np.nan


def point_world_z(path):
    """World-Z (mm) of a single-point Slicer markups file."""
    j = json.load(open(path))
    return float(j['markups'][0]['controlPoints'][0]['position'][2])


def build_dseg(cord, ref_img, z_l1top, z_l5bot):
    """Labeled seg for the top-bottom metric: whole cord = 2, L1-top slice = 1, L5-bottom slice = 5."""
    aff = ref_img.affine
    kz = nib.affines.apply_affine(
        aff, np.c_[np.zeros(ref_img.shape[2]), np.zeros(ref_img.shape[2]), np.arange(ref_img.shape[2])])[:, 2]
    k1, k5 = int(np.argmin(np.abs(kz - z_l1top))), int(np.argmin(np.abs(kz - z_l5bot)))
    dseg = cord.astype(np.uint8) * 2
    dseg[:, :, k1][cord[:, :, k1] > 0] = 1
    dseg[:, :, k5][cord[:, :, k5] > 0] = 5
    return dseg


def length_along_centerline(dseg_path, param_centerline, dir_qc=None):
    """Arc length (mm) along the fitted cord centerline between the top of L1 and the bottom of L5.

    Same computation as lumbar_length_along_centerline.py (Jan Valosek): fit the centerline to
    the binarized seg, then interpolate the cumulative centerline length at the two end slices.
    """
    im_seg = Image(dseg_path).change_orientation('RPI')
    data = im_seg.data

    def level_slice(value, point):
        z = np.where(data == value)[2]
        return {'top': z.max() + 0.5, 'bottom': z.min() - 0.5}[point]

    z_first, z_last = level_slice(1, 'top'), level_slice(5, 'bottom')

    im_bin = im_seg.copy(); im_bin.data = (im_seg.data > 0).astype(np.uint8)
    im_ctl, arr_ctl, arr_der, _ = get_centerline(im_bin, param=param_centerline, verbose=0, space='pix')
    points = im_bin.transfo_pix2phys(arr_ctl.T, mode='absolute')
    deriv = im_bin.transfo_pix2phys(arr_der.T, mode='relative')
    centerline = Centerline(*points.T, *deriv.T)
    z_ctl = arr_ctl[2]
    if dir_qc:
        im_ctl.change_orientation(Image(dseg_path).orientation).save(
            os.path.join(dir_qc, os.path.basename(dseg_path).replace('.nii.gz', '_centerline.nii.gz')),
            mutable=True, verbose=0)
    from scipy.interpolate import interp1d
    length_at = interp1d(z_ctl, centerline.incremental_length, fill_value='extrapolate')
    return float(length_at(z_first) - length_at(z_last))


def make_figure(df, out_png):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy import stats
    x = df['height_cm'].to_numpy(float); y = df['L1-L5_length_mm'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for sex, s in SEX_STYLE.items():
        m = (df['sex'] == sex).to_numpy()
        if m.any():
            ax.scatter(x[m], y[m], c=s['color'], marker=s['marker'], s=70, zorder=3,
                       edgecolor='white', linewidth=0.5, label=f"{s['label']} (n={m.sum()})")
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min() - 2, x.max() + 2, 50)
    ax.plot(xx, slope * xx + intercept, color='0.4', lw=1.5, ls='--', zorder=2)
    r, pr = stats.pearsonr(x, y); rho, ps = stats.spearmanr(x, y)
    ax.text(0.03, 0.97, f"Pearson r = {r:.2f} (p = {pr:.3f})\nSpearman ρ = {rho:.2f} (p = {ps:.3f})\nn = {len(x)}",
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', ec='0.8'))
    for _, row in df.iterrows():
        ax.annotate(row['subject'].replace('sub-', ''), (row['height_cm'], row['L1-L5_length_mm']),
                    textcoords='offset points', xytext=(5, 4), fontsize=8, color='0.3')
    ax.set_xlabel('Height [cm]', fontsize=13); ax.set_ylabel('Lumbar cord L1–L5 length [mm]', fontsize=13)
    ax.set_title('Lumbar spinal cord L1–L5 length vs height', fontsize=13)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11); plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    return r, pr, rho, ps


def main():
    parser = argparse.ArgumentParser(description='Compute the lumbar cord L1-L5 length along the centerline '
                                                 'for the Fudan lumbar dataset from its Slicer annotations.')
    parser.add_argument('-i', required=True, help='Fudan dataset root (with rawdata/ and markers/).')
    parser.add_argument('-bounds', default=None,
                        help='Folder with manual boundary points sub-*/sub-*_{L1top,L5S1}.mrk.json. '
                             'If a point is missing it is estimated from the nerveroots markers.')
    parser.add_argument('-o', default=None, help='Output folder. Default: <-i>/derivatives/L1-L5.')
    parser.add_argument('-centerline-algo', choices=['polyfit', 'bspline', 'linear', 'nurbs'], default='bspline')
    parser.add_argument('-centerline-smooth', type=int, default=30)
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.i))
    bounds = os.path.abspath(os.path.expanduser(args.bounds)) if args.bounds else None
    out = os.path.abspath(os.path.expanduser(args.o)) if args.o else os.path.join(root, 'derivatives', 'L1-L5')
    dir_qc = os.path.join(out, 'centerline')
    os.makedirs(dir_qc, exist_ok=True)
    pc = ParamCenterline(algo_fitting=args.centerline_algo, smooth=args.centerline_smooth, minmax=True)
    part = pd.read_csv(os.path.join(root, 'rawdata', 'participants.tsv'), sep='\t')

    rows = []
    for ciss in sorted(glob.glob(os.path.join(root, 'rawdata', 'sub-*', 'anat', 'sub-*_CISS.nii.gz'))):
        sub = os.path.basename(ciss).split('_')[0]
        mdir = os.path.join(root, 'markers', sub)
        ref = nib.load(ciss)
        cord = rasterize_cord(mdir, sub, ref)
        if cord.sum() == 0:
            print(f'{sub}: no cord markers, skipped'); continue

        def boundary(name, level):
            f = os.path.join(bounds, sub, f'{sub}_{name}.mrk.json') if bounds else None
            if f and os.path.exists(f):
                return point_world_z(f), 'manual'
            return nerveroot_zep(mdir, sub, level), 'auto'

        z_l1, s1src = boundary('L1top', 'L1')
        z_l5, s5src = boundary('L5S1', 'S1')
        dseg = build_dseg(cord, ref, z_l1, z_l5)
        dseg_path = os.path.join(out, f'{sub}_spinallevels_dseg.nii.gz')
        hdr = ref.header.copy(); hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(dseg, ref.affine, hdr), dseg_path)
        L = length_along_centerline(dseg_path, pc, dir_qc)

        p = part[part.participant_id == sub]
        p = p.iloc[0] if len(p) else {}
        rows.append({'subject': sub, 'sex': p.get('sex', ''), 'height_cm': p.get('height', np.nan),
                     'age': p.get('age', np.nan), 'weight_kg': p.get('weight', np.nan),
                     'L1-L5_length_mm': round(L, 2),
                     'L1_boundary': s1src, 'L5_boundary': s5src,
                     'L1top_z_mm': round(z_l1, 1), 'L5bot_z_mm': round(z_l5, 1)})
        print(f"{sub}: L1-L5 = {L:5.1f} mm  (sex {p.get('sex','?')}, height {p.get('height','?')} cm, "
              f"L1 {s1src}, L5 {s5src})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'L1-L5_lengths.csv'), index=False)
    try:
        df.to_excel(os.path.join(out, 'L1-L5_lengths.xlsx'), index=False)
    except Exception as e:
        print('xlsx skipped (install openpyxl/xlsxwriter):', e)

    y = df['L1-L5_length_mm'].to_numpy(float)
    print(f"\nL1-L5: mean {y.mean():.1f} ± {y.std(ddof=0):.1f} mm, range {y.min():.0f}-{y.max():.0f}")
    if df['height_cm'].notna().all():
        r, pr, rho, ps = make_figure(df, os.path.join(out, 'L1-L5_vs_height.png'))
        print(f"vs height: Pearson r={r:.2f} (p={pr:.3f}), Spearman ρ={rho:.2f} (p={ps:.3f})")
    print('Saved to', out)


if __name__ == '__main__':
    main()
