"""
Get lumbar spinal levels from manual rootlets and spinal cord segmentations (lumbar-marseille dataset).

Follows the "SOP Rootlet segmentation" (Balgrist Campus, v1.0 2026):
    Step 3 - ZEP (Zone of Entry of the rootlets): the slice where the rootlet merges with the cord. Estimated as the
             most rostral (superior) slice of the rootlet label, separately for the left and right side (side = voxel
             position relative to the cord centre of mass in that slice).
             A rootlet that reaches the top slice of the image is cut by the FOV -> its ZEP is unknown (NaN).
    Step 4 - left/right asymmetry:
                 difference of 1 slice -> take the more rostral slice
                 difference of 2 slices -> take the middle slice
                 difference > 2 slices -> not covered by the SOP; take the middle slice (rounded rostrally), flag it
             If only one side is present, its ZEP is used and the level is flagged.
    Step 5 - spinal level X spans from the ZEP-X slice (rostral boundary) down to the slice just above the
             ZEP-(X+1) slice (caudal boundary). The most caudal labelled rootlet has no next ZEP, so it only serves
             as the caudal boundary of the level above.
    Only the L1-L5 spinal levels are saved (the S1 ZEP is needed for the caudal boundary of L5).

Rootlet label values follow the SCT spinal levels convention
(https://spinalcordtoolbox.com/stable/overview/concepts/pam50.html#spinal-levels):
    C1 = 1, ..., C8 = 8, T1 = 9, ..., T12 = 20, L1 = 21, ..., L5 = 25, S1 = 26, ..., S5 = 30

Outputs (in -ofolder), per subject:
    <rootlets>_spinal_levels.nii.gz : the spinal cord segmentation labelled with the spinal levels (same values as
                                      the rootlets)
    <rootlets>_spinal_levels.csv    : per-level ZEPs (left, right, combined), level start/end slices and length
and, across subjects:
    spinal_levels_lumbar-marseille.csv

Slices are 0-indexed in RPI orientation (slice 0 = most inferior).

The script requires the SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh && conda activate venv_sct

Example:
    python rootlets_to_spinal_levels.py -i ~/data/data.neuro.polymtl.ca/lumbar-marseille/derivatives/labels -ofolder results
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

from spinalcordtoolbox.image import Image, zeros_like

# SCT spinal levels: C1-C8 = 1-8, T1-T12 = 9-20, L1-L5 = 21-25, S1-S5 = 26-30
LABEL_TO_LEVEL = {label: f'{region}{i}' for label, (region, i) in enumerate(
    [('C', i) for i in range(1, 9)] + [('T', i) for i in range(1, 13)] +
    [('L', i) for i in range(1, 6)] + [('S', i) for i in range(1, 6)], start=1)}
# Spinal levels saved in the outputs: L1-L5
LEVELS_TO_KEEP = list(range(21, 26))
ROOTLETS_SUFFIX = '_label-rootlets_dseg.nii.gz'
SEG_SUFFIX = '_sc_seg.nii.gz'


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True,
                        help='Path to derivatives/labels. Every *' + ROOTLETS_SUFFIX + ' with a matching *' +
                             SEG_SUFFIX + ' is processed.')
    parser.add_argument('-ofolder', default='results', help='Output folder. Default: results')
    return parser


def get_cord_center_x(seg):
    """Cord centre of mass along x (R-L) for every slice; slices without cord are filled by the nearest slice."""
    nz = seg.shape[2]
    z_cord = [z for z in range(nz) if seg[:, :, z].any()]
    cx_cord = [np.where(seg[:, :, z])[0].mean() for z in z_cord]
    return np.interp(np.arange(nz), z_cord, cx_cord)   # np.interp holds the edge values outside the cord


def combine_left_right(zep_l, zep_r):
    """SOP step 4. Returns (combined ZEP, flag)."""
    if np.isnan(zep_l) and np.isnan(zep_r):
        return np.nan, 'no ZEP (rootlet cut by the FOV)'
    if np.isnan(zep_l) or np.isnan(zep_r):
        return np.nanmax([zep_l, zep_r]), 'one side only'
    diff = abs(zep_l - zep_r)
    if diff <= 1:
        return max(zep_l, zep_r), ''         # identical, or 1 slice -> more rostral (RPI: larger z)
    middle = int(np.ceil((zep_l + zep_r) / 2))  # 2 slices -> exact middle; > 2 -> rounded rostrally
    return middle, '' if diff == 2 else f'L/R difference of {int(diff)} slices'


def get_spinal_levels(fname_rootlets, fname_seg, ofolder):
    rootlets_img = Image(fname_rootlets)
    orientation = rootlets_img.orientation
    rootlets = rootlets_img.change_orientation('RPI').data.round().astype(int)
    seg_img = Image(fname_seg).change_orientation('RPI')
    seg = seg_img.data > 0.5
    if rootlets.shape != seg.shape:
        raise ValueError(f'Shape mismatch: {fname_rootlets} {rootlets.shape} vs {fname_seg} {seg.shape}')
    nz = rootlets.shape[2]
    pz = seg_img.dim[6]
    cord_cx = get_cord_center_x(seg)

    # Step 3 + 4: ZEP per rootlet label (RPI: x increases towards the left)
    rows = []
    for label in np.unique(rootlets[rootlets > 0]):
        x, _, z = np.where(rootlets == label)
        is_left = x > cord_cx[z]
        zep = {}
        for side, mask in (('L', is_left), ('R', ~is_left)):
            top = z[mask].max() if mask.any() else np.nan
            zep[side] = np.nan if top == nz - 1 else top   # touches the top of the FOV -> real ZEP is above
        zep_combined, flag = combine_left_right(zep['L'], zep['R'])
        rows.append({'label': label, 'level': LABEL_TO_LEVEL.get(label, f'?{label}'),
                     'zep_L': zep['L'], 'zep_R': zep['R'], 'zep': zep_combined, 'flag': flag})
    df = pd.DataFrame(rows).sort_values('label').reset_index(drop=True)

    # Step 5: level X = [ZEP-(X+1) + 1, ZEP-X]
    df['start_slice'] = df['zep']                       # rostral boundary
    df['end_slice'] = df['zep'].shift(-1) + 1           # caudal boundary (next level's ZEP, exclusive)
    invalid = df['end_slice'] > df['start_slice']
    if invalid.any():
        print(f'WARNING: ZEPs not in rostro-caudal order for {df.loc[invalid, "level"].tolist()}')
        df.loc[invalid, 'flag'] = (df.loc[invalid, 'flag'] + '; ZEP below next level ZEP').str.strip('; ')
        df.loc[invalid, ['start_slice', 'end_slice']] = np.nan
    df['n_slices'] = df['start_slice'] - df['end_slice'] + 1
    df['length_mm'] = df['n_slices'] * pz

    # Keep only L1-L5 (the S1 ZEP is still used above as the caudal boundary of L5)
    df = df[df['label'].isin(LEVELS_TO_KEEP)].reset_index(drop=True)
    missing = sorted(set(LEVELS_TO_KEEP) - set(df['label']))
    if missing:
        print(f'WARNING: missing rootlets for {[LABEL_TO_LEVEL[label] for label in missing]}')

    # Spinal levels image: cord voxels labelled with the level value
    levels_img = zeros_like(seg_img, dtype=np.uint8)
    for _, row in df.dropna(subset=['n_slices']).iterrows():
        z0, z1 = int(row['end_slice']), int(row['start_slice'])
        levels_img.data[:, :, z0:z1 + 1][seg[:, :, z0:z1 + 1]] = row['label']
    fname_out = os.path.join(ofolder, os.path.basename(fname_rootlets).replace('.nii.gz', '_spinal_levels.nii.gz'))
    levels_img.change_orientation(orientation).save(fname_out)
    df.to_csv(fname_out.replace('.nii.gz', '.csv'), index=False)
    return df


def main():
    args = get_parser().parse_args()
    os.makedirs(args.ofolder, exist_ok=True)
    dfs = []
    for fname_rootlets in sorted(glob.glob(os.path.join(args.i, 'sub-*', '**', '*' + ROOTLETS_SUFFIX), recursive=True)):
        fname_seg = fname_rootlets.replace(ROOTLETS_SUFFIX, SEG_SUFFIX)
        if not os.path.isfile(fname_seg):
            print(f'Skipping {fname_rootlets}: no spinal cord segmentation')
            continue
        subject = os.path.basename(fname_rootlets).split('_')[0]
        print(f'Processing {subject}')
        df = get_spinal_levels(fname_rootlets, fname_seg, args.ofolder)
        df.insert(0, 'subject', subject)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    fname_csv = os.path.join(args.ofolder, 'spinal_levels_lumbar-marseille.csv')
    df_all.to_csv(fname_csv, index=False)
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(df_all)
    print(f'Saved: {fname_csv}')


if __name__ == '__main__':
    main()
