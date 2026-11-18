# Sacral Balgrist

Sacral spinal cord length from the top of the **S1** spinal level to the **conus medullaris** tip, measured
**along the cord centerline**, for the Balgrist lumbar (14_LTR) cohort. Same metric structure as the lumbar
[`../lumbar-balgrist`](../lumbar-balgrist) / [`../lumbar-fudan`](../lumbar-fudan) scripts, with the top of S1
replacing the PMJ of the whole-cord length, so the numbers are comparable across measures and cohorts.

## Method

The sacral length is the arc length along the **whole-spine localizer** cord centerline (the same centerline as the
total PMJ-to-conus cord length, which already reaches the conus), between two superior-inferior world-Z levels:

1. **Top of S1.** The top edge of the S1 level from the CISS spinal-level segmentation (`6 = S1`), or a manually
   placed/checked point (`sub-*_S1top.mrk.json`) when available.
2. **Conus.** Label `60` (conus) placed on the localizer centerline during the whole-cord-length work, or a manually
   placed/checked point (`sub-*_conus.mrk.json`).
3. **Length.** The centerline x, y are smoothed (moving average) and the arc length is taken between the two world-Z
   levels, with exact interpolation at each level. Because the localizer centerline reaches the conus, no
   extrapolation is needed (`centerline_extrap_mm` ~ 0; re-check if > 3 mm).
4. **Correlation.** `correlation_length_height.py` correlates the length with subject height, coloured by sex,
   reporting the Spearman rank correlation (rho with 95 % CI) and a least-squares linear fit (visual guide), same
   style as the other cohorts.

## Scripts

| Script | What it does |
|---|---|
| `sacral_length_along_centerline.py` | Read the S1-top (segmentation or manual) and conus (label 60 or manual) levels and measure the S1-to-conus length along the localizer centerline -> `sacral_length_S1-conus_centerline.csv`. Requires SCT (`conda activate venv_sct`). |
| `correlation_length_height.py` | Correlation between the sacral S1-conus length and subject height, using the shared figure `../visualisation_utils.py` (`plot_correlation`: sex by color, dataset by marker, linear fit + 95 % CI, Spearman rho [95 % CI]), so the figure matches the cervical and combined panels. |

```bash
conda activate venv_sct
python sacral_length_along_centerline.py -seg <seg> -i <localizer labels> -bounds <bounds> -o <out>
python correlation_length_height.py -i <out>/sacral_length_S1-conus_centerline.csv
```

If a boundary point is missing for a subject, that boundary is taken from the segmentation (S1) or label 60 (conus).

## Notes

- `sacral_length_along_centerline.py` requires SCT. The subject sex and height are read from the participants Excel
  and carried into the CSV.
