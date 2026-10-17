# Lumbar Fudan

Lumbar spinal cord length from the top of L1 to the bottom of L5, measured **along the cord centerline**, for the
Fudan (Shanghai) lumbar dataset. Unlike the Balgrist and EPFL cohorts, the Fudan dataset does not ship a per-level
`*_spinallevels_dseg.nii.gz`; it ships Slicer markups (a cord contour per CISS slice and nerve-root points), so the
length script builds the inputs it needs from those markups and then uses the **same centerline measurement** as
[`../lumbar-epfl`](../lumbar-epfl) / [`../lumbar-balgrist`](../lumbar-balgrist) (`-metric top-bottom`), making the
Fudan numbers comparable to the other cohorts.

## Method

1. **Cord mask.** The `cord_*` closed-curve markers (one closed contour per CISS axial slice) are rasterized onto
   the CISS grid → binary cord mask.
2. **End boundaries** (superior-inferior axis), each at the rootlet zone of entry (ZEP) on the cord:
   - L1 top = ZEP of L1 (top edge of the most superior L1 slice);
   - L5 bottom = ZEP(S1) − 1 slice (bottom edge of the last L5 slice, included).
   Taken from manually placed Slicer points (`sub-*_L1top.mrk.json`, `sub-*_L5S1.mrk.json`) when available,
   otherwise estimated automatically from the `nerveroots_L1` / `nerveroots_S1` markers.
3. **Length.** A labeled segmentation is built (whole cord = 2, value 1 on the L1-top slice, value 5 on the
   L5-bottom slice) and the centerline is fitted with SCT (`ParamCenterline`, bspline, smooth = 30). The length is
   the arc length of the centerline between the two end slices (both included).
4. **Correlation.** `correlation_length_height.py` correlates the length with subject height, coloured by sex,
   reporting the Spearman rank correlation (ρ with 95 % CI) and a least-squares linear fit (visual guide), same
   style as the other cohorts.

## Scripts

| Script | What it does |
|---|---|
| `lumbar_fudan_L1-L5_length.py` | Build the cord mask + boundaries from the Slicer markups and measure the L1–L5 length along the centerline → `L1-L5_lengths.csv`. |
| `correlation_length_height.py` | Correlation between the L1–L5 length and subject height, by sex. |
| `utils.py` | Shared plotting style and statistics (mirrors `../lumbar-balgrist/utils.py`). |

```bash
python lumbar_fudan_L1-L5_length.py -i <fudan root> -bounds <fudan root>/derivatives/route_b_boundary
python correlation_length_height.py -i <fudan root>/derivatives/L1-L5/L1-L5_lengths.csv
```

`lumbar_fudan_L1-L5_length.py` requires SCT (`conda activate venv_sct`). The subject sex and height are read from
the dataset `participants.tsv` and carried into `L1-L5_lengths.csv`.
