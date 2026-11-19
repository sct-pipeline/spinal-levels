# Sacral Fudan

Sacral spinal cord length from the top of the **S1** spinal level to the **conus medullaris** tip, measured
**along the cord centerline**, for the Fudan (Shanghai) lumbar cohort. Same metric as
[`../sacral-balgrist`](../sacral-balgrist), so the numbers are comparable across cohorts. Unlike Balgrist, the Fudan
dataset does not ship a localizer centerline; it ships manual cord contours, so the centerline substrate differs.

## Method

1. **Centerline.** Per-axial-slice centroid of the CISS cord segmentation
   (`derivatives/cord_seg_from_markers/sub-*/sub-*_cord_seg.nii.gz`, rasterized from the manual `cord_*` contours); it reaches
   the conus.
2. **Bounds.** Manually placed top-of-S1 and conus points
   (`sacral_length/bounds/sub-*/sub-*_{S1top,conus}.mrk.json`).
3. **Corrected conus.** The cord segmentation stops a few mm above the true conus for the larger subjects, so the
   first conus points sat too low (in the cauda equina), inflating the length by 5 to 9 mm. The conus was re-checked
   and raised on the **sagittal** view; corrected points live in a `new/` subfolder next to each subject's original
   bounds. The length script prefers `new/` when present (original otherwise) and records the choice in the
   `S1top_source` / `conus_source` columns. After the fix the centerline reaches the corrected conus
   (`centerline_extrap_mm` ~ 0 to 2 mm); no re-segmentation is needed.
4. **Length.** The centerline x, y are smoothed and the arc length is taken between the two world-Z levels, with
   exact interpolation at each level (same as `../sacral-balgrist`).
5. **Correlation.** `correlation_length_height.py` correlates the length with subject height, coloured by sex,
   reporting the Spearman rank correlation (rho with 95 % CI) and a least-squares linear fit, same style as the
   other cohorts.

## Scripts

| Script | What it does |
|---|---|
| `fundan_sacral_length.py` | Build the per-slice centroid centerline, read the S1-top and conus bounds (preferring `new/`) and measure the S1-to-conus length along the centerline -> `fundan_sacral_length.csv`. Needs only numpy / pandas / nibabel (no SCT). |
| `correlation_length_height.py` | Correlation between the sacral S1-conus length and subject height, using the shared figure `../visualisation_utils.py` (`plot_correlation`: sex by color, dataset by marker, linear fit + 95 % CI, Spearman rho [95 % CI]), so the figure matches the cervical and combined panels. |

```bash
python fundan_sacral_length.py -i <fudan dataset root>
python correlation_length_height.py -i <fudan root>/sacral_length/fundan_sacral_length.csv
```

## Notes

- The subject sex and height are read from the dataset's L1-L5 demographics table and carried into the CSV.
